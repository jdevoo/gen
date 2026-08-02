package toolbox

import (
	"context"
	"fmt"
	"strings"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

// ListMCPPrompts lists available prompts from available MCP servers.
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
