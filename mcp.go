package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/shlex"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/sync/errgroup"
	"google.golang.org/genai"
)

// SessionArray holds a list of MCP client session
type SessionArray []*mcp.ClientSession

// ToolRegistry maps tool names to the session
type ToolMap map[string]*mcp.ClientSession

// initMCPSessions starts the MCP server processes and connects clients.
// TODO timeout hardcoded
func initMCPSessions(ctx context.Context, params *Parameters) error {
	if len(params.MCPServers) == 0 {
		return nil
	}

	if !isRedirected(os.Stdout) && !params.Verbose {
		spinner := NewSpinner("%s")
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
			parts, err := shlex.Split(srvStr)
			if err != nil || len(parts) == 0 {
				return fmt.Errorf("invalid MCP command '%s': %v", srvStr, err)
			}

			cmdPath, err := exec.LookPath(parts[0])
			if err != nil {
				return fmt.Errorf("cannot find MCP server '%s': %v", parts[0], err)
			}

			options := mcp.ClientOptions{
				Capabilities: &mcp.ClientCapabilities{
					RootsV2:     &mcp.RootCapabilities{ListChanged: false},
					Elicitation: &mcp.ElicitationCapabilities{},
					Sampling:    &mcp.SamplingCapabilities{},
				},
				CreateMessageHandler: genSampling,
				ElicitationHandler:   genElicitation,
			}

			if params.Verbose && params.Tool {
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

			// NOTE 30s connection timeout
			mcpCtx, cancel := context.WithTimeout(gCtx, 30*time.Second)
			defer cancel()

			session, err := client.Connect(mcpCtx, &mcp.CommandTransport{
				Command: exec.Command(cmdPath, parts[1:]...),
			}, nil)
			if err != nil {
				return fmt.Errorf("MCP connect error: %v", err)
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
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return fmt.Errorf("registerMcpTools: params not found in context")
	}

	for _, sess := range params.MCPSessions {
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to list MCP tools: %v", err)
		}

		mcpDecls := []*genai.FunctionDeclaration{}
		for _, tool := range ltr.Tools {
			params.ToolRegistry[tool.Name] = sess
			if tool.InputSchema == nil {
				return fmt.Errorf("no input schema for MCP tool: '%s'", tool.Name)
			}
			jsonBytes, err := json.Marshal(tool.InputSchema)
			if err != nil {
				return fmt.Errorf("failed to marshal input schema for MCP tool '%s': %v", tool.Name, err)
			}
			var mcpInputSchema genai.Schema
			if err = json.Unmarshal(jsonBytes, &mcpInputSchema); err != nil {
				return fmt.Errorf("failed to unmarshal JSON bytes for MCP tool '%s': %v", tool.Name, err)
			}
			mcpDecls = append(mcpDecls, &genai.FunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  &mcpInputSchema,
			})
		}
		if len(mcpDecls) > 0 {
			config.Tools = append(config.Tools, &genai.Tool{
				FunctionDeclarations: mcpDecls,
			})
		}
	}
	return nil
}

func genLoggingHandler(_ context.Context, r *mcp.LoggingMessageRequest) {
	fmt.Fprintf(os.Stderr, "\033[36m[MCP %v] %+v\033[0m\n", r.Params.Level, r.Params.Data)
}

// invokeMCPTool looks for a tool across MCP sessions matching the provided FunctionCall signature.
func invokeMCPTool(ctx context.Context, fc *genai.FunctionCall) []*genai.Part {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return []*genai.Part{
			genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
				"output": "",
				"error":  "invokeTool: params not found in context",
			}),
		}
	}

	// Lookup tool
	sess, ok := params.ToolRegistry[fc.Name]
	if !ok {
		return []*genai.Part{}
	}

	ctr, err := sess.CallTool(ctx, &mcp.CallToolParams{
		Name:      fc.Name,
		Arguments: fc.Args,
	})
	if err != nil {
		return []*genai.Part{
			genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
				"output": "",
				"error":  fmt.Sprintf("invokeMcpTool: %s", err.Error()),
			}),
		}
	}

	var parts []*genai.Part
	var outputStrings []string
	var errorStrings []string

	for _, c := range ctr.Content {
		switch v := c.(type) {
		case *mcp.TextContent:
			outputStrings = append(outputStrings, v.Text)
		case *mcp.ResourceLink:
			outputStrings = append(outputStrings, fmt.Sprintf("%+v", v))
		case *mcp.ImageContent:
			stripper := &PNGAncillaryChunkStripper{Reader: bytes.NewReader(v.Data)}
			strippedData, err := io.ReadAll(stripper)
			if err != nil {
				errorStrings = append(errorStrings, "invokeMcpTool: error in PNG ancillary chunk stripper")
				continue
			}
			parts = append(parts, genai.NewPartFromBytes(strippedData, c.(*mcp.ImageContent).MIMEType))
			parts = append(parts, genai.NewPartFromText("\n"))
		case *mcp.AudioContent:
			errorStrings = append(errorStrings, "invokeMcpTool: audio content not supported")
		case *mcp.EmbeddedResource:
			outputStrings = append(outputStrings, v.Resource.Text)
		}
	}
	parts = append(parts,
		genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"output": strings.Join(outputStrings, "\n"),
			"error":  strings.Join(errorStrings, "\n"),
		}))
	return parts
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
			parts := strings.Split(val, ",")
			for _, p := range parts {
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
	return nil, fmt.Errorf("Unsupported MCP type %s", t)
}

// genSampling message callback for MCP servers.
func genSampling(ctx context.Context, req *mcp.CreateMessageRequest) (*mcp.CreateMessageResult, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return nil, fmt.Errorf("genSampling: params not found in context")
	}
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("genSampling: failed to create genai client")
	}
	if len((*req.Params).Messages) == 0 || (*req.Params).Messages[0].Content == nil {
		return nil, fmt.Errorf("genSampling: prompt missing")
	}
	prompt := genai.Text((*req.Params).Messages[0].Content.(*mcp.TextContent).Text)
	res, err := client.Models.GenerateContent(ctx, params.GenModel, prompt, nil)
	if err != nil {
		return nil, err
	}
	return &mcp.CreateMessageResult{
		Content: &mcp.TextContent{
			Text: string(res.Candidates[0].Content.Parts[0].Text),
		},
		Role: "assistant",
	}, nil
}

// genElicitation callback for MCP servers that request inputs not supplied via -p.
func genElicitation(ctx context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		return nil, fmt.Errorf("genElicitation: keyVals not found in context")
	}
	res := mcp.ElicitResult{
		Action:  "",
		Content: map[string]any{},
	}
	schemaMap, ok := (*req.Params).RequestedSchema.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("expected map[string]any but got %T", (*req.Params).RequestedSchema)
	}
	propsMap, ok := schemaMap["properties"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("'properties' not found in RequestedSchema")
	}
	reqIf, ok := schemaMap["required"].([]any)
	var reqProps []string
	if ok {
		for _, i := range reqIf {
			if p, ok := i.(string); ok {
				reqProps = append(reqProps, p)
			}
		}
	}
	for _, p := range reqProps {
		if _, val := keyVals[p]; !val {
			res.Action = "cancel"
		}
	}
	var out []string
	for propName, propSchemaIf := range propsMap {
		if propSchema, ok := propSchemaIf.(map[string]any); ok {
			propType, _ := propSchema["type"].(string)
			if valString, ok := keyVals[propName]; ok {
				propVal, _ := convertMCPType(valString, propType)
				res.Content[propName] = propVal
			} else {
				propDesc, _ := propSchema["description"]
				out = append(out, fmt.Sprintf("  -p %s=<%s> %s", propName, propType, propDesc))
			}
		}
	}
	if len(out) > 0 {
		res.Action = "cancel"
		out = append([]string{(*req.Params).Message}, out...)
		return nil, fmt.Errorf("missing information\n%s", strings.Join(out, "\n"))
	}
	res.Action = "accept"
	return &res, nil
}
