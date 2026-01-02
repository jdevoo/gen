package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

// SessionArray holds a list of MCP client session
type SessionArray []*mcp.ClientSession

// registerMCPTools declares tools of MCP servers in genai.FunctionDeclaration format.
func registerMCPTools(ctx context.Context, config *genai.GenerateContentConfig) error {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return fmt.Errorf("registerMcpTools: params not found in context")
	}
	for _, sess := range params.MCPSessions {
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to list MCP tools: %w", err)
		}
		mcpDecls := []*genai.FunctionDeclaration{}
		// MCP tools for this server
		for _, tool := range ltr.Tools {
			if tool.InputSchema == nil {
				return fmt.Errorf("no input schema for MCP tool: '%s'", tool.Name)
			}
			jsonBytes, err := json.Marshal(tool.InputSchema)
			if err != nil {
				return fmt.Errorf("failed to marshal input schema for MCP tool '%s': %w", tool.Name, err)
			}
			var mcpInputSchema genai.Schema
			if err = json.Unmarshal(jsonBytes, &mcpInputSchema); err != nil {
				return fmt.Errorf("failed to unmarshal JSON bytes for MCP tool '%s': %w", tool.Name, err)
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
	fmt.Fprintf(os.Stderr, "\033[36m%+v\033[0m\n\n", r.Params.Data)
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
	for _, sess := range params.MCPSessions {
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			return []*genai.Part{
				genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
					"output": "",
					"error":  fmt.Sprintf("invokeMcpTool: %s", err.Error()),
				}),
			}
		}
		for _, tool := range ltr.Tools {
			if tool.Name == fc.Name {
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
				for _, c := range ctr.Content {
					switch c.(type) {
					case *mcp.TextContent:
						parts = append(parts, genai.NewPartFromText(c.(*mcp.TextContent).Text))
					case *mcp.ResourceLink:
						parts = append(parts,
							genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
								"output": fmt.Sprintf("%+v", c),
								"error":  "",
							}))
					case *mcp.ImageContent:
						img := c.(*mcp.ImageContent)
						stripper := &PNGAncillaryChunkStripper{Reader: bytes.NewReader(img.Data)}
						strippedData, err := io.ReadAll(stripper)
						if err != nil {
							parts = append(parts,
								genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
									"output": "",
									"error":  "invokeMcpTool: error in PNG ancillary chunk stripper",
								}))
							continue
						}
						parts = append(parts, genai.NewPartFromBytes(strippedData, c.(*mcp.ImageContent).MIMEType))
						parts = append(parts, genai.NewPartFromText("\n"))
					case *mcp.AudioContent:
						parts = append(parts,
							genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
								"output": "",
								"error":  "invokeMcpTool: audio content not supported",
							}))
					case *mcp.EmbeddedResource:
						parts = append(parts,
							genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
								"output": c.(*mcp.EmbeddedResource).Resource.Text,
								"error":  "",
							}))
					}
				}
				return parts
			} // if tool.Name == fc.Name {
		} // for _, tool := range ltr.Tools {
	} // for _, sess := range params.McpSessions {
	return []*genai.Part{}
}

// convertMCPType attempts to convert a string value to a target type as defined in the JSON schema.
func convertMCPType(val string, t string) (any, error) {
	switch strings.ToLower(t) {
	case "string":
		return val, nil
	case "integer":
		i, err := strconv.ParseInt(val, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse '%s' as integer: %w", val, err)
		}
		return i, nil
	case "number":
		f, err := strconv.ParseFloat(val, 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse '%s' as number: %w", val, err)
		}
		return f, nil
	case "boolean":
		b, err := strconv.ParseBool(val)
		if err != nil {
			return nil, fmt.Errorf("failed to parse '%s' as boolean: %w", val, err)
		}
		return b, nil
	case "object", "array":
		var v any
		err := json.Unmarshal([]byte(val), &v)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal '%s' as JSON %s: %w", val, t, err)
		}
		return v, nil
	}
	return nil, fmt.Errorf("Unsupported MCP type %s", t)
}

// genSampling message callback for MCP servers
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

// genElicitation callback for MCP servers that request inputs not supplied via -p
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
