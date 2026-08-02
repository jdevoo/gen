package toolbox

import (
	"context"
	"fmt"
	"strings"

	"github.com/jdevoo/gen/core"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

type GetMCPPromptArgs struct {
	Name string `json:"name"`
}

// GetPrompt retrieves a specific prompt by name from available MCP servers.
func (t Tool) GetMCPPrompt(ctx context.Context, args GetMCPPromptArgs) (*genai.Part, error) {
	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("GetMCPPrompt: params not found in context")
	}
	keyVals, ok := ctx.Value(core.KeyValsKey).(core.ParamMap)
	if !ok {
		return nil, fmt.Errorf("GetMCPPrompt: keyVals not found in context")
	}
	for _, sess := range params.MCPSessions {
		for p, err := range sess.Prompts(ctx, nil) {
			if err != nil {
				continue // skip this MCP server
			}
			if args.Name == p.Name {
				if err := argsUnsatisfied(p.Arguments, keyVals); err != nil {
					return nil, err
				}
				prompt, err := sess.GetPrompt(ctx, &mcp.GetPromptParams{
					Name:      p.Name,
					Arguments: keyVals,
				})
				if err != nil {
					return nil, err
				}
				var textStrings []string
				var parts []*genai.FunctionResponsePart
				for _, msg := range prompt.Messages {
					switch c := msg.Content.(type) {
					case *mcp.TextContent:
						textStrings = append(textStrings, c.Text)
					case *mcp.ResourceLink:
						parts = append(parts, genai.NewFunctionResponsePartFromURI(c.URI, c.MIMEType))
					case *mcp.EmbeddedResource:
						if c.Resource != nil {
							if len(c.Resource.Blob) > 0 {
								parts = append(parts, genai.NewFunctionResponsePartFromBytes(c.Resource.Blob, c.Resource.MIMEType))
							}
							if len(c.Resource.Text) > 0 {
								textStrings = append(textStrings, c.Resource.Text)
							}
						}
					}
				}
				return genai.NewPartFromFunctionResponseWithParts(
					"GetMCPPrompt",
					map[string]any{
						"output": "SUCCESS",
						"text":   textStrings,
					},
					parts,
				), nil
			}
		}
	}
	return genai.NewPartFromFunctionResponse(
		"GetMCPPrompt",
		map[string]any{"output": fmt.Sprintf("GetPrompt: '%s' not found", args.Name)},
	), nil
}

// argsUnsatisfied returns a list of prompt parameters missing from keyVals.
func argsUnsatisfied(args []*mcp.PromptArgument, keyVals core.ParamMap) *core.ParamError {
	var out []string
	for _, arg := range args {
		if _, val := keyVals[arg.Name]; !val {
			out = append(out, fmt.Sprintf("  -p %s", arg.Name))
		}
	}
	if len(out) > 0 {
		return &core.ParamError{
			Message: strings.Join(out, "\n"),
		}
	}
	return nil
}
