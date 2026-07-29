package toolbox

import (
	"context"
	"fmt"
	"strings"

	"github.com/jdevoo/gen/core"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

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

// ListPrompts lists available prompts from available MCP servers.
func (t Tool) ListMCPPrompts(ctx context.Context) (*genai.Part, error) {
	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("ListMCPPrompts: params not found in context")
	}
	var res []string
	for _, sess := range params.MCPSessions {
		for p, err := range sess.Prompts(ctx, nil) {
			if err != nil {
				continue // skip this MCP server
			}
			desc := fmt.Sprintf("%s: %s", p.Name, p.Description)
			if len(p.Arguments) > 0 {
				desc += " ("
			}
			var args []string
			for _, arg := range p.Arguments {
				name := arg.Name
				if !arg.Required {
					name += " optional"
				}
				args = append(args, name)
			}
			if len(p.Arguments) > 0 {
				desc += fmt.Sprintf("%s) ", strings.Join(args, ","))
			}
			res = append(res, desc)
		}
	}
	return genai.NewPartFromFunctionResponse(
		"ListMCPPrompts",
		map[string]any{
			"output": "SUCCESS",
			"text":   res,
		},
	), nil
}

type GetPromptArgs struct {
	Name string `json:"name"`
}

// GetPrompt retrieves a specific prompt by name from available MCP servers.
func (t Tool) GetMCPPrompt(ctx context.Context, args GetPromptArgs) (*genai.Part, error) {
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

// ListResources returns resources available from MCP servers.
func (t Tool) ListMCPResources(ctx context.Context) (*genai.Part, error) {
	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("ListResources: params not found in context")
	}
	var res []string
	for _, sess := range params.MCPSessions {
		for r, err := range sess.Resources(ctx, nil) {
			if err != nil {
				continue // skip this MCP server
			}
			res = append(res, r.URI)
		}
	}
	return genai.NewPartFromFunctionResponse(
		"ListMCPResources",
		map[string]any{
			"output": "SUCCESS",
			"text":   res,
		}), nil
}

type GetResourceArgs struct {
	Name string `json:"name"`
}

// GetResource retrieves a specific resource by name from available MCP servers.
func (t Tool) GetMCPResource(ctx context.Context, args GetResourceArgs) (*genai.Part, error) {
	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("GetResource: params not found in context")
	}
	for _, sess := range params.MCPSessions {
		for r, err := range sess.Resources(ctx, nil) {
			if err != nil {
				continue // skip this MCP server
			}
			if args.Name == r.URI {
				res, err := sess.ReadResource(ctx, &mcp.ReadResourceParams{
					URI: args.Name,
				})
				if err != nil {
					return nil, err
				}
				var textStrings []string
				var parts []*genai.FunctionResponsePart
				for _, c := range res.Contents {
					if len(c.Text) > 0 {
						textStrings = append(textStrings, c.Text)
					}
					if len(c.Blob) > 0 {
						parts = append(parts, genai.NewFunctionResponsePartFromBytes(c.Blob, c.MIMEType))
					}
				}
				return genai.NewPartFromFunctionResponseWithParts(
					"GetMCPResource",
					map[string]any{
						"output": "SUCCESS",
						"text":   textStrings,
					},
					parts,
				), nil
			}
		}
	}
	return nil, fmt.Errorf("GetMCPResource: '%s' not found", args.Name)
}