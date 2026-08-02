package toolbox

import (
	"context"
	"fmt"

	"github.com/jdevoo/gen/core"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

type GetMCPResourceArgs struct {
	Name string `json:"name"`
}

// GetResource retrieves a specific resource by name from available MCP servers.
func (t Tool) GetMCPResource(ctx context.Context, args GetMCPResourceArgs) (*genai.Part, error) {
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
