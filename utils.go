package main

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"

	_ "github.com/lib/pq"
	"github.com/modelcontextprotocol/go-sdk/mcp"

	"google.golang.org/api/googleapi"
	"google.golang.org/genai"
)

// ParamMap holds key-value pairs for string replacement.
type ParamMap map[string]string

// String implements the flag.Value interface for ParamMap.
func (*ParamMap) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
func (m *ParamMap) Set(kv string) error {
	parts := strings.SplitN(kv, "=", 2) // limit splits to 2
	if len(parts) != 2 {
		return fmt.Errorf("invalid parameter %s", kv)
	}
	(*m)[parts[0]] = parts[1]
	return nil
}

// ParamArray holds a list of strings.
type ParamArray []string

// SessionArray holds a list of MCP client session
type SessionArray []*mcp.ClientSession

// String implements the flag.Value interface for ParamMap.
func (*ParamArray) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
func (a *ParamArray) Set(val string) error {
	*a = append(*a, val)
	return nil
}

// Property for MCP JSON unmarshal
type Property struct {
	Description string `json:"description"`
	Type        string `json:"type"`
}

// GenSchema for MCP JSON unmarshal
type GenSchema struct {
	Schema     string              `json:"$schema"`
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required"`
	Type       string              `json:"type"`
}

// derefParts is a kludge for SendMessageStream
func derefParts(parts []*genai.Part) []genai.Part {
	res := []genai.Part{}
	for _, p := range parts {
		res = append(res, *p)
	}
	return res
}

// conjoin returns a single text resulting from concatenation of all original parts.
// TODO handle other part types
func conjTexts(parts *[]*genai.Part) {
	if len(*parts) == 0 {
		return
	}
	var buf bytes.Buffer
	for i, p := range *parts {
		if p.Text != "" {
			if i > 0 && buf.Len() > 0 {
				buf.WriteString(" ")
			}
			buf.WriteString(string(p.Text))
		}
	}
	*parts = []*genai.Part{{Text: buf.String()}}
}

// searchReplace performs string replacement based on key-value pairs.
func searchReplace(prompt string, pm ParamMap) string {
	res := prompt
	for k, v := range pm {
		searchRegex := regexp.MustCompile("(?i){" + k + "}")
		res = searchRegex.ReplaceAllString(res, v)
	}
	return res
}

// partWithKey searches prompt parts for occurrence of key and returns index.
func partWithKey(parts []*genai.Part, key string) int {
	for idx, part := range parts {
		if strings.Contains(string(part.Text), key) {
			return idx
		}
	}
	return -1
}

// replacePart returns new array with updated entry at idx.
func replacePart(parts *[]*genai.Part, idx int, key string, selection []QueryResult) {
	var keyVal string
	for _, s := range selection {
		keyVal += s.doc.content
	}
	text := (*parts)[idx].Text
	(*parts)[idx] = &genai.Part{Text: strings.Replace(string(text), key, keyVal, 1)}
}

// prependToParts extends prompts with digest selection.
func prependToParts(parts *[]*genai.Part, selection []QueryResult) {
	var res []*genai.Part
	for _, s := range selection {
		res = append(res, &genai.Part{Text: s.doc.content})
	}
	*parts = append(res, (*parts)...)
}

// appendToSelection extends selection with a query result in decreasing order of MMR up to k chunks.
func appendToSelection(selection []QueryResult, item QueryResult, k int) []QueryResult {
	result := selection
	result = append(result, item)
	sort.Slice(result[:], func(i, j int) bool {
		return result[i].mmr > result[j].mmr
	})
	if len(result) > k {
		return result[0:k]
	}
	return result
}

// knownTools returns string of comma-separated function names.
func knownTools(params *Parameters) string {
	var res []string

	// gen tools
	genTool := reflect.TypeOf(Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		res = append(res, fmt.Sprintf("  * %s", genTool.Method(i).Name))
	}

	// MCP tools
	for _, sess := range params.McpSessions {
		ctx := context.Background()
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			res = append(res, fmt.Sprintf("  * %v", err))
			continue
		}
		for _, tool := range ltr.Tools {
			res = append(res, fmt.Sprintf("  * %v", tool.Name))
		}
	}

	return strings.Join(res, "\n")
}

// registerTools declares functions of type Tool in genai.FunctionDeclaration format.
// TODO add support for arrays and objects
func registerGenTools(config *genai.GenerateContentConfig) {
	genTool := reflect.TypeOf(Tool{})
	n := genTool.NumMethod()
	genDecls := make([]*genai.FunctionDeclaration, n)
	for i := 0; i < n; i++ {
		m := genTool.Method(i)
		f := reflect.ValueOf(Tool{}).MethodByName(m.Name)
		t := f.Type()
		argMap := make(map[string]*genai.Schema)
		if t.NumIn() > 0 {
			for j := 0; j < t.NumIn(); j++ {
				switch t.In(j).Kind() {
				case reflect.String:
					argMap[fmt.Sprintf("arg%d", j)] = &genai.Schema{Type: genai.TypeString}
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
					argMap[fmt.Sprintf("arg%d", j)] = &genai.Schema{Type: genai.TypeInteger}
				case reflect.Float32, reflect.Float64:
					argMap[fmt.Sprintf("arg%d", j)] = &genai.Schema{Type: genai.TypeNumber}
				case reflect.Bool:
					argMap[fmt.Sprintf("arg%d", j)] = &genai.Schema{Type: genai.TypeBoolean}
				}
			}
			genDecls[i] = &genai.FunctionDeclaration{
				Name: m.Name,
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: argMap,
				},
			}
		} else {
			genDecls[i] = &genai.FunctionDeclaration{
				Name: m.Name,
			}
		}
	}
	if len(genDecls) > 0 {
		config.Tools = append(config.Tools, &genai.Tool{
			FunctionDeclarations: genDecls,
		})
	}
}

// registerMcpTools declares tools of MCP servers in genai.FunctionDeclaration format.
func registerMcpTools(ctx context.Context, config *genai.GenerateContentConfig, params *Parameters) error {
	for _, sess := range params.McpSessions {
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to list MCP tools: %w", err)
		}
		mcpDecls := []*genai.FunctionDeclaration{}
		// MCP tools for this server
		for _, tool := range ltr.Tools {
			jsonBytes, err := json.Marshal(tool.InputSchema)
			if err != nil {
				return fmt.Errorf("failed to marshal input schema for MCP tool '%s': %w", tool.Name, err)
			}
			var mcpInputSchema GenSchema
			if err = json.Unmarshal(jsonBytes, &mcpInputSchema); err != nil {
				return fmt.Errorf("failed to unmarshal JSON bytes for MCP tool '%s': %w", tool.Name, err)
			}
			argMap := make(map[string]*genai.Schema)
			for name, def := range mcpInputSchema.Properties {
				switch strings.ToLower(def.Type) {
				case "string":
					argMap[name] = &genai.Schema{
						Type:        genai.TypeString,
						Description: def.Description,
					}
				case "number":
					argMap[name] = &genai.Schema{
						Type:        genai.TypeNumber,
						Description: def.Description,
					}
				case "integer":
					argMap[name] = &genai.Schema{
						Type:        genai.TypeInteger,
						Description: def.Description,
					}
				case "boolean":
					argMap[name] = &genai.Schema{
						Type:        genai.TypeBoolean,
						Description: def.Description,
					}
				case "object":
					argMap[name] = &genai.Schema{
						Type:        genai.TypeObject,
						Description: def.Description,
					}
				case "array":
					argMap[name] = &genai.Schema{
						Type:        genai.TypeArray,
						Description: def.Description,
					}
				}
			}
			mcpDecls = append(mcpDecls, &genai.FunctionDeclaration{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: argMap,
				},
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

// invokeTool calls tool identified by genai.FunctionCall using anonymous argument names.
func invokeTool(ctx context.Context, params *Parameters, fc genai.FunctionCall) string {
	f := reflect.ValueOf(Tool{}).MethodByName(fc.Name)
	if !f.IsValid() {
		for _, sess := range params.McpSessions {
			res, err := sess.CallTool(ctx, &mcp.CallToolParams{
				Name:      fc.Name,
				Arguments: fc.Args,
			})
			if err != nil {
				return fmt.Sprintf("%s error: %v", fc.Name, err)
			}
			return res.Content[0].(*mcp.TextContent).Text
		}
		return "NO TOOL FOUND"
	}
	var args []reflect.Value
	for i := 0; i < len(fc.Args); i++ {
		t := f.Type().In(i)
		v := reflect.New(t).Elem()
		arg := fc.Args[fmt.Sprintf("arg%d", i)]
		switch t.Kind() {
		case reflect.String:
			v.SetString(arg.(string))
		}
		args = append(args, v)
	}
	vals := f.Call(args)
	if err := vals[1].Interface(); err != nil {
		return fmt.Sprintf("%s error: %v", fc.Name, err)
	}
	return vals[0].String()
}

// hasInvokedTool checks for a suggested function call, invokes tool and returns response to model.
func hasInvokedTool(ctx context.Context, params *Parameters, resp *genai.GenerateContentResponse) (bool, *genai.FunctionResponse) {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return false, &genai.FunctionResponse{}
	}
	for _, fc := range resp.FunctionCalls() {
		res := invokeTool(ctx, params, *fc)
		return true, &genai.FunctionResponse{
			Name: fc.Name,
			Response: map[string]any{
				"Response": res,
			},
		}
	}
	return false, &genai.FunctionResponse{}
}

// genLogFatal refines the error if available and exits with 1
func genLogFatal(err error) {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		log.Fatal(gerr)
	} else {
		log.Fatal(err)
	}
}

// anyMatches returns true if any of the match candidates are found in
func anyMatches(strArray []string, candidates ...string) bool {
	for _, s := range strArray {
		for _, c := range candidates {
			if strings.Contains(strings.ToLower(s), strings.ToLower(c)) {
				return true
			}
		}
	}
	return false
}

// allMatch returns true if all list elements match.
func allMatch(strArray []string, cand string) bool {
	if len(strArray) == 0 {
		return false
	}
	for _, s := range strArray {
		if !strings.Contains(strings.ToLower(s), strings.ToLower(cand)) {
			return false
		}
	}
	return true
}

// oneMatches returns true if one and only one matches.
func oneMatches(strArray []string, cand string) bool {
	var res int
	for _, s := range strArray {
		if strings.Contains(strings.ToLower(s), strings.ToLower(cand)) {
			res += 1
		}
	}
	return res == 1
}

// QueryPostgres submits query to database set by DSN parameter.
func queryPostgres(query string) (string, error) {
	var res []string
	dsn, ok := keyVals["DSN"]
	if !ok || len(dsn) == 0 {
		return "", fmt.Errorf("DSN parameter missing")
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return "", fmt.Errorf("opening DSN '%s': %w", dsn, err)
	}
	defer db.Close()
	rows, err := db.Query(query)
	if err != nil {
		return "", fmt.Errorf("for query '%s': %w", query, err)
	}
	defer rows.Close()
	cols, _ := rows.Columns()
	row := make([]interface{}, len(cols))
	rowPtr := make([]interface{}, len(cols))
	for i := range row {
		rowPtr[i] = &row[i]
	}
	for rows.Next() {
		err := rows.Scan(rowPtr...)
		if err != nil {
			return "", err
		}
		res = append(res, fmt.Sprintf("%v", row))
	}
	if err := rows.Err(); err != nil {
		return "", err
	}
	return strings.Join(res, "\n"), nil
}

// isFlagSet visits the flags passed to the command at runtime.
func isFlagSet(name string) bool {
	res := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			res = true
		}
	})
	return res
}

// loadPrefs reads and parses .genrc from the user's home directory
func loadPrefs(params *Parameters) error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %w", err)
	}
	prefsPath := filepath.Join(homeDir, DotGenRc)
	prefsFile, err := os.Open(prefsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to open at %s: %w", prefsPath, err)
	}
	defer prefsFile.Close()

	scanner := bufio.NewScanner(prefsFile)
	currentSection := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) == 0 || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue // skip empty line or line with comments
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			currentSection = strings.ToLower(line[1 : len(line)-1])
			continue
		}
		switch currentSection {
		case "flags":
			parts := strings.SplitN(line, "=", 2)
			if len(parts) != 2 {
				return fmt.Errorf("flag error on line: %s", line)
			}
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			switch strings.ToLower(key) {
			case "k":
				if val, err := strconv.Atoi(value); err == nil {
					params.K = val
				}
			case "lambda":
				if val, err := strconv.ParseFloat(value, 64); err == nil {
					params.Lambda = val
				}
			case "temp":
				if val, err := strconv.ParseFloat(value, 64); err == nil {
					params.Temp = val
				}
			case "topp":
				if val, err := strconv.ParseFloat(value, 64); err == nil {
					params.TopP = val
				}
			case "embmodel":
				params.EmbModel = value
			case "genmodel":
				params.GenModel = value
			default:
				return fmt.Errorf("unknown key value %s", key)
			}
		case "digestpaths":
			params.DigestPaths = append(params.DigestPaths, line)
		case "mcpservers":
			params.McpServers = append(params.McpServers, line)
		default:
			return fmt.Errorf("unknown section: %s", currentSection)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}
	return nil
}
