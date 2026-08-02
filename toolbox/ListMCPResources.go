package toolbox

import (
	"context"
	"fmt"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

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
