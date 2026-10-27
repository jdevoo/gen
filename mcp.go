package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/shlex"
	"github.com/jdevoo/gen/core"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"
)

type ElicitError struct {
	mu  sync.Mutex
	err error
}

func (t *ElicitError) SetError(err error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.err == nil {
		t.err = err
	}
}

func (t *ElicitError) Error() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.err
}

// initMCPSessions starts the MCP server processes and connects clients.
// TODO timeout hardcoded
func initMCPSessions(ctx context.Context, params *core.Parameters) error {
	if len(params.MCPServers) == 0 {
		return nil
	}

	var spinner *Spinner
	if !params.OutRedirected && !params.Verbose {
		spinner = NewSpinner("%s")
		spinner.Start()
		defer spinner.Stop()
	}

	// Get the current working directory for AddRoots
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("failed to get current working directory: %v", err)
	}

	// Parallel startup of sessions
	g, gCtx := errgroup.WithContext(ctx)
	var mu sync.Mutex

	for _, srv := range params.MCPServers {
		srvStr := srv // capture for closure
		g.Go(func() error {
			var session *mcp.ClientSession
			var connErr error

			options := mcp.ClientOptions{
				// deprecated under SEP-2577 but supported during the transition window
				Capabilities: &mcp.ClientCapabilities{
					RootsV2:  &mcp.RootCapabilities{ListChanged: false},
					Sampling: &mcp.SamplingCapabilities{},
				},
				CreateMessageHandler: genSampling,
				ElicitationHandler:   genElicitation,
			}
			if params.Verbose && params.Tool {
				// deprecated under SEP-2577 but kept for verbose output
				options.LoggingMessageHandler = genLoggingHandler
			}

			client := mcp.NewClient(
				&mcp.Implementation{Name: filepath.Base(os.Args[0]), Version: Version},
				&options,
			)

			client.AddRoots(&mcp.Root{
				Name: "gen",
				URI:  "file://" + filepath.ToSlash(cwd),
			})

			parts, err := shlex.Split(srvStr)
			if err != nil || len(parts) == 0 {
				return fmt.Errorf("invalid MCP entry '%s': %v", srvStr, err)
			}

			endpoint := parts[0]
			isStreamableServer := strings.HasPrefix(endpoint, "http://") ||
				strings.HasPrefix(endpoint, "https://")
			mcpCtx, cancel := context.WithTimeout(gCtx, 30*time.Second)
			defer cancel()

			if isStreamableServer {
				var clientID, clientSecret string
				if len(parts) > 1 {
					clientID = parts[1]
				}
				if len(parts) > 2 {
					clientSecret = parts[2]
				}

				httpClient := &http.Client{
					Transport: http.DefaultTransport.(*http.Transport).Clone(),
				}
				oauthHandler, oerr := buildOAuthHandler(endpoint, clientID, clientSecret, httpClient, spinner)
				if oerr != nil {
					return fmt.Errorf("failed to setup OAuth for MCP server '%s': %v", endpoint, oerr)
				}
				session, connErr = client.Connect(mcpCtx, &mcp.StreamableClientTransport{
					Endpoint:     endpoint,
					HTTPClient:   httpClient,
					OAuthHandler: oauthHandler,
				}, nil)
			} else {
				cmdPath, err := exec.LookPath(parts[0])
				if err != nil {
					return fmt.Errorf("cannot find MCP server '%s': %v", parts[0], err)
				}
				cmd := exec.Command(cmdPath, parts[1:]...)
				if params.Verbose {
					cmd.Stderr = os.Stderr
				} else {
					cmd.Stderr = io.Discard
				}
				session, connErr = client.Connect(mcpCtx, &mcp.CommandTransport{
					Command: cmd,
				}, nil)
			}

			if connErr != nil {
				return fmt.Errorf("MCP connect error: %v", connErr)
			}
			mu.Lock()
			params.MCPSessions = append(params.MCPSessions, session)
			mu.Unlock()

			return nil
		})
	}
	return g.Wait()
}

// registerMCPTools declares tools of MCP servers in genai.FunctionDeclaration format.
func registerMCPTools(ctx context.Context, config *genai.GenerateContentConfig) error {
	params, _ := ctx.Value(core.ParamsKey).(*core.Parameters)

	for _, sess := range params.MCPSessions {
		// TODO risk of name collision
		mcpDecls := []*genai.FunctionDeclaration{}
		for tool, err := range sess.Tools(ctx, nil) {
			if err != nil {
				return fmt.Errorf("failed to list MCP tools: %v", err)
			}
			params.ToolRegistry[tool.Name] = sess
			decl := &genai.FunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
			}
			if tool.InputSchema != nil {
				decl.ParametersJsonSchema = tool.InputSchema
			}
			mcpDecls = append(mcpDecls, decl)
		}

		if len(mcpDecls) > 0 {
			var funcTool *genai.Tool
			for _, t := range config.Tools {
				if t.FunctionDeclarations != nil {
					funcTool = t
					break
				}
			}
			if funcTool == nil {
				funcTool = &genai.Tool{}
				config.Tools = append(config.Tools, funcTool)
			}
			funcTool.FunctionDeclarations = append(funcTool.FunctionDeclarations, mcpDecls...)
		}
	}

	return nil
}

// sigMCPTool parses the input schema of an MCP Tool and returns its signature.
func sigMCPTool(tool *mcp.Tool) string {
	schemaBytes, err := json.Marshal(tool.InputSchema)
	if err != nil {
		return fmt.Sprintf("  • %s ?", tool.Name)
	}
	var schemaMap map[string]any
	if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
		return fmt.Sprintf("  • %s ?", tool.Name)
	}

	props, _ := schemaMap["properties"].(map[string]any)
	reqs, _ := schemaMap["required"].([]any)
	requiredSet := make(map[string]bool)
	for _, r := range reqs {
		if s, ok := r.(string); ok {
			requiredSet[s] = true
		}
	}

	var keys []string
	for k := range props {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var params []string
	for _, k := range keys {
		prop, ok := props[k].(map[string]any)
		if !ok {
			continue
		}
		pType, _ := prop["type"].(string)
		if pType == "" {
			pType = "any"
		}
		reqStr := ""
		if requiredSet[k] {
			reqStr = "*"
		}
		if pType == "string" {
			params = append(params, fmt.Sprintf("%s%s", k, reqStr))
		} else {
			params = append(params, fmt.Sprintf("%s%s (%s)", k, reqStr, pType))
		}
	}

	if len(params) == 0 {
		return fmt.Sprintf("  • %s", tool.Name)
	}
	return fmt.Sprintf("  • %s %s", tool.Name, strings.Join(params, ", "))
}

func genLoggingHandler(_ context.Context, r *mcp.LoggingMessageRequest) {
	fmt.Fprintf(os.Stderr, infos("[MCP %v] %+v\n"), r.Params.Level, r.Params.Data)
}

// invokeMCPTool looks for a tool across MCP sessions matching the provided FunctionCall signature.
func invokeMCPTool(ctx context.Context, fc *genai.FunctionCall) (*genai.Part, error) {
	params, _ := ctx.Value(core.ParamsKey).(*core.Parameters)

	// Lookup tool
	sess, ok := params.ToolRegistry[fc.Name]
	if !ok {
		return nil, nil
	}

	ctr, err := sess.CallTool(ctx, &mcp.CallToolParams{
		Name:      fc.Name,
		Arguments: fc.Args,
	})
	if t, ok := ctx.Value(core.ElicitErrKey).(*ElicitError); ok {
		if elicitErr := t.Error(); elicitErr != nil {
			if params.ChatMode {
				err = elicitErr
			} else {
				return nil, elicitErr
			}
		}
	}
	if err != nil {
		return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"error": fmt.Sprintf("invokeMCPTool: transport error: %s", err.Error()),
		}), nil
	}
	if ctr.IsError {
		var errText []string
		for _, c := range ctr.Content {
			if tc, ok := c.(*mcp.TextContent); ok {
				errText = append(errText, tc.Text)
			}
		}
		return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"error": fmt.Sprintf("invokeMCPTool: tool execution failed: %s", strings.Join(errText, "\n")),
		}), nil
	}

	var parts []*genai.FunctionResponsePart
	var errStrings []string
	var textStrings []string

	for _, c := range ctr.Content {
		switch v := c.(type) {
		case *mcp.TextContent:
			textStrings = append(textStrings, v.Text)
		case *mcp.ResourceLink:
			if res, err := sess.ReadResource(ctx, &mcp.ReadResourceParams{URI: v.URI}); err != nil {
				textStrings = append(textStrings, fmt.Sprintf("[Resource link: %s (%s)]", v.URI, v.MIMEType))
			} else {
				for _, rc := range res.Contents {
					if len(rc.Text) > 0 {
						textStrings = append(textStrings, rc.Text)
					}
					if len(rc.Blob) > 0 {
						parts = append(parts, genai.NewFunctionResponsePartFromBytes(rc.Blob, rc.MIMEType))
					}
				}
			}
		case *mcp.ImageContent:
			stripper := &PNGAncillaryChunkStripper{Reader: bytes.NewReader(v.Data)}
			strippedData, err := io.ReadAll(stripper)
			if err != nil {
				errStrings = append(errStrings, "invokeMCPTool: error in PNG ancillary chunk stripper")
				continue
			}
			parts = append(parts, genai.NewFunctionResponsePartFromBytes(strippedData, v.MIMEType))
		case *mcp.AudioContent:
			errStrings = append(errStrings, "invokeMCPTool: audio content not supported")
		case *mcp.EmbeddedResource:
			if v.Resource != nil {
				if len(v.Resource.Blob) > 0 {
					parts = append(parts, genai.NewFunctionResponsePartFromBytes(v.Resource.Blob, v.Resource.MIMEType))
				}
				if len(v.Resource.Text) > 0 {
					textStrings = append(textStrings, v.Resource.Text)
				}
			}
		}
	}

	if len(errStrings) > 0 {
		return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"error": errStrings,
		}), nil
	}

	resMap := map[string]any{
		"output": "SUCCESS",
	}

	if ctr.StructuredContent != nil {
		resMap["result"] = ctr.StructuredContent
	}

	if len(textStrings) > 0 {
		var parsedItems []any
		for _, s := range textStrings {
			var parsed any
			if err := json.Unmarshal([]byte(s), &parsed); err == nil {
				parsedItems = append(parsedItems, parsed)
			} else {
				parsedItems = append(parsedItems, s)
			}
		}

		if params.JSON {
			if len(parsedItems) == 1 {
				resMap["result"] = parsedItems[0]
			} else {
				resMap["result"] = parsedItems
			}
		} else {
			if len(textStrings) == 1 {
				resMap["text"] = textStrings[0]
			} else {
				resMap["text"] = textStrings
			}
		}
	} else if ctr.StructuredContent == nil {
		resMap["text"] = ""
	}

	return genai.NewPartFromFunctionResponseWithParts(
		fc.Name,
		resMap,
		parts,
	), nil
}

// convertMCPType attempts to convert a string value to a target type as defined in the JSON schema.
func convertMCPType(val string, t string) (any, error) {
	switch strings.ToLower(t) {
	case "string":
		return val, nil
	case "integer":
		i, err := strconv.ParseInt(val, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse '%s' as integer: %v", val, err)
		}
		return i, nil
	case "number":
		f, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse '%s' as number: %v", val, err)
		}
		return f, nil
	case "boolean":
		b, err := strconv.ParseBool(val)
		if err != nil {
			return nil, fmt.Errorf("failed to parse '%s' as boolean: %v", val, err)
		}
		return b, nil
	case "array":
		var v []any
		if strings.HasPrefix(val, "[") {
			if err := json.Unmarshal([]byte(val), &v); err != nil {
				return nil, fmt.Errorf("failed to unmarshal '%s' as JSON %s: %v", val, t, err)
			}
		} else { // Fallback: treat comma separated as array of strings
			for p := range strings.SplitSeq(val, ",") {
				v = append(v, strings.TrimSpace(p))
			}
		}
		return v, nil
	case "object":
		var v any
		err := json.Unmarshal([]byte(val), &v)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal '%s' as JSON %s: %v", val, t, err)
		}
		return v, nil
	}
	return nil, fmt.Errorf("unsupported MCP type %s", t)
}

// genSampling message callback for MCP servers.
func genSampling(ctx context.Context, req *mcp.CreateMessageRequest) (*mcp.CreateMessageResult, error) {
	params, _ := ctx.Value(core.ParamsKey).(*core.Parameters)

	client, err := core.GetGenClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("genSampling: failed to get genai client: %w", err)
	}

	p := req.Params
	if p == nil || len(p.Messages) == 0 {
		return nil, fmt.Errorf("genSampling: prompt missing")
	}

	var contents []*genai.Content
	for _, msg := range p.Messages {
		if msg == nil || msg.Content == nil {
			continue
		}
		role := genai.Role(genai.RoleUser)
		if msg.Role == "assistant" {
			role = genai.Role(genai.RoleModel)
		}
		var parts []*genai.Part
		switch c := msg.Content.(type) {
		case *mcp.TextContent:
			parts = append(parts, genai.NewPartFromText(c.Text))
		case *mcp.ImageContent:
			parts = append(parts, genai.NewPartFromBytes(c.Data, c.MIMEType))
		case *mcp.EmbeddedResource:
			if c.Resource != nil {
				if len(c.Resource.Text) > 0 {
					parts = append(parts, genai.NewPartFromText(c.Resource.Text))
				} else if len(c.Resource.Blob) > 0 {
					parts = append(parts, genai.NewPartFromBytes(c.Resource.Blob, c.Resource.MIMEType))
				}
			}
		}
		if len(parts) > 0 {
			contents = append(contents, genai.NewContentFromParts(parts, role))
		}
	}

	if len(contents) == 0 {
		return nil, fmt.Errorf("genSampling: no valid text or media parts found in messages")
	}

	modelName := params.GenModel
	cfg := &genai.GenerateContentConfig{}
	if p.SystemPrompt != "" {
		cfg.SystemInstruction = genai.NewContentFromText(p.SystemPrompt, genai.RoleUser)
	}
	if p.Temperature > 0 {
		t := float32(p.Temperature)
		cfg.Temperature = &t
	}
	if p.MaxTokens > 0 {
		cfg.MaxOutputTokens = int32(p.MaxTokens)
	}
	if len(p.StopSequences) > 0 {
		cfg.StopSequences = p.StopSequences
	}

	res, err := client.Models.GenerateContent(ctx, modelName, contents, cfg)
	if err != nil {
		return nil, err
	}

	var outText strings.Builder
	if len(res.Candidates) > 0 && res.Candidates[0].Content != nil {
		for _, pt := range res.Candidates[0].Content.Parts {
			if !pt.Thought && pt.Text != "" {
				outText.WriteString(pt.Text)
			}
		}
	}

	return &mcp.CreateMessageResult{
		Content: &mcp.TextContent{
			Text: outText.String(),
		},
		Model: modelName,
		Role:  "assistant",
	}, nil
}

// genElicitation callback for MCP servers that request inputs not supplied via -p.
func genElicitation(ctx context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
	res := mcp.ElicitResult{
		Action:  "accept",
		Content: map[string]any{},
	}

	if (*req.Params).RequestedSchema == nil {
		return nil, fmt.Errorf("elicitation request has no schema (mode: %s, url: %s)", (*req.Params).Mode, (*req.Params).URL)
	}

	schemaMap, ok := (*req.Params).RequestedSchema.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("expected map[string]any but got %T", (*req.Params).RequestedSchema)
	}

	propsMap, ok := schemaMap["properties"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("'properties' not found in requested schema")
	}

	keyVals, _ := ctx.Value(core.KeyValsKey).(core.ParamMap)

	var propNames []string
	for propName := range propsMap {
		propNames = append(propNames, propName)
	}
	sort.Strings(propNames)

	var out []string
	for _, propName := range propNames {
		propSchemaIf := propsMap[propName]
		if propSchema, ok := propSchemaIf.(map[string]any); ok {
			propType, _ := propSchema["type"].(string)
			if valString, ok := keyVals[propName]; ok {
				propVal, err := convertMCPType(valString, propType)
				if err != nil {
					out = append(out, fmt.Sprintf("  -p %s: %v", propName, err))
				} else if err := validateMCPValue(propVal, propSchema); err != nil {
					out = append(out, fmt.Sprintf("  -p %s: %v", propName, err))
				} else {
					res.Content[propName] = propVal
				}
			} else if defVal, ok := propSchema["default"]; ok {
				res.Content[propName] = defVal
			} else {
				out = append(out, describeElicitField(propName, propSchema))
			}
		}
	}

	if len(out) > 0 {
		out = append([]string{(*req.Params).Message}, out...)
		paramErr := &core.ParamError{
			Message: strings.Join(out, "\n"),
		}
		if t, ok := ctx.Value(core.ElicitErrKey).(*ElicitError); ok {
			t.SetError(paramErr)
		}
		return nil, paramErr
	}

	return &res, nil
}

// describeElicitField parses an elicitation property schema.
func describeElicitField(name string, schema map[string]any) string {
	pType, _ := schema["type"].(string)
	if pType == "" {
		pType = "any"
	}

	// Capture formats (e.g., email, uri, date)
	if format, ok := schema["format"].(string); ok && format != "" {
		pType = pType + ":" + format
	}

	var description string
	if desc, ok := schema["description"].(string); ok {
		description = desc
	}

	var choices []string

	parseOfList := func(ofList []any) {
		for _, itemIf := range ofList {
			if item, ok := itemIf.(map[string]any); ok {
				var constVal string
				if cv, ok := item["const"]; ok {
					constVal = fmt.Sprintf("%v", cv)
				}
				var title string
				if t, ok := item["title"].(string); ok {
					title = t
				}
				if constVal != "" {
					if title != "" {
						choices = append(choices, fmt.Sprintf("%s (%s)", constVal, title))
					} else {
						choices = append(choices, constVal)
					}
				}
			}
		}
	}

	if enumVals, ok := schema["enum"].([]any); ok {
		var enumNames []string
		if names, ok := schema["enumNames"].([]any); ok {
			for _, n := range names {
				enumNames = append(enumNames, fmt.Sprintf("%v", n))
			}
		}
		for i, v := range enumVals {
			valStr := fmt.Sprintf("%v", v)
			if i < len(enumNames) {
				choices = append(choices, fmt.Sprintf("%s (%s)", valStr, enumNames[i]))
			} else {
				choices = append(choices, valStr)
			}
		}
	}

	if oneOfList, ok := schema["oneOf"].([]any); ok {
		parseOfList(oneOfList)
	}
	if anyOfList, ok := schema["anyOf"].([]any); ok {
		parseOfList(anyOfList)
	}

	if pType == "array" {
		if itemsIf, ok := schema["items"].(map[string]any); ok {
			itemType, _ := itemsIf["type"].(string)
			if itemType != "" {
				pType = fmt.Sprintf("array[%s]", itemType)
			}
			if itemEnums, ok := itemsIf["enum"].([]any); ok {
				for _, v := range itemEnums {
					choices = append(choices, fmt.Sprintf("%v", v))
				}
			}
			if itemOneOf, ok := itemsIf["oneOf"].([]any); ok {
				parseOfList(itemOneOf)
			}
			if itemAnyOf, ok := itemsIf["anyOf"].([]any); ok {
				parseOfList(itemAnyOf)
			}
		}
	}

	var limits []string
	if minVal, ok := schema["minimum"]; ok {
		limits = append(limits, fmt.Sprintf("min: %v", minVal))
	}
	if maxVal, ok := schema["maximum"]; ok {
		limits = append(limits, fmt.Sprintf("max: %v", maxVal))
	}
	if minItems, ok := schema["minItems"]; ok {
		limits = append(limits, fmt.Sprintf("minItems: %v", minItems))
	}
	if maxItems, ok := schema["maxItems"]; ok {
		limits = append(limits, fmt.Sprintf("maxItems: %v", maxItems))
	}

	var defaultStr string
	if defVal, ok := schema["default"]; ok {
		if arr, ok := defVal.([]any); ok {
			var arrStrs []string
			for _, el := range arr {
				arrStrs = append(arrStrs, fmt.Sprintf("%v", el))
			}
			defaultStr = "[" + strings.Join(arrStrs, ", ") + "]"
		} else {
			defaultStr = fmt.Sprintf("%v", defVal)
		}
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, "  -p %s <%s>", name, pType)
	if len(limits) > 0 {
		fmt.Fprintf(&sb, " (%s)", strings.Join(limits, ", "))
	}
	if description != "" {
		sb.WriteString(" - ")
		sb.WriteString(description)
	}
	if len(choices) > 0 {
		fmt.Fprintf(&sb, " (options: %s)", strings.Join(choices, ", "))
	}
	if defaultStr != "" {
		fmt.Fprintf(&sb, " [default: %s]", defaultStr)
	}

	return sb.String()
}

// validateMCPValue checks if the converted value satisfies schema constraints.
func validateMCPValue(val any, schema map[string]any) error {
	if enumVals, ok := schema["enum"].([]any); ok {
		found := false
		for _, ev := range enumVals {
			if fmt.Sprintf("%v", val) == fmt.Sprintf("%v", ev) {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("value '%v' is not one of the allowed options", val)
		}
	}

	checkOfList := func(ofList []any) (bool, []string) {
		var allowed []string
		hasConst := false
		for _, itemIf := range ofList {
			if item, ok := itemIf.(map[string]any); ok {
				if cv, ok := item["const"]; ok {
					hasConst = true
					constStr := fmt.Sprintf("%v", cv)
					allowed = append(allowed, constStr)
					if fmt.Sprintf("%v", val) == constStr {
						return true, nil
					}
				}
			}
		}
		return !hasConst, allowed
	}

	if oneOfList, ok := schema["oneOf"].([]any); ok {
		if ok, allowed := checkOfList(oneOfList); !ok {
			return fmt.Errorf("value '%v' is not one of the allowed options: %s", val, strings.Join(allowed, ", "))
		}
	}
	if anyOfList, ok := schema["anyOf"].([]any); ok {
		if ok, allowed := checkOfList(anyOfList); !ok {
			return fmt.Errorf("value '%v' is not one of the allowed options: %s", val, strings.Join(allowed, ", "))
		}
	}

	if min, ok := schema["minimum"]; ok {
		if minF, err := strconv.ParseFloat(fmt.Sprintf("%v", min), 64); err == nil {
			if valF, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 64); err == nil {
				if valF < minF {
					return fmt.Errorf("value %v is less than minimum %v", val, min)
				}
			}
		}
	}
	if max, ok := schema["maximum"]; ok {
		if maxF, err := strconv.ParseFloat(fmt.Sprintf("%v", max), 64); err == nil {
			if valF, err := strconv.ParseFloat(fmt.Sprintf("%v", val), 64); err == nil {
				if valF > maxF {
					return fmt.Errorf("value %v is greater than maximum %v", val, max)
				}
			}
		}
	}

	return nil
}
