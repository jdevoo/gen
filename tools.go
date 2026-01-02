package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

type Tool struct{}

// ListKnownGeminiModels retrieves the list of available Gemini models.
func (t Tool) ListKnownGeminiModels(ctx context.Context) (string, error) {
	var res string
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		genLogFatal(err)
	}
	for m, err := range client.Models.All(ctx) {
		if err != nil {
			return "", err
		}
		res += fmt.Sprintf("%s %s\n", m.Name, m.Description)
	}
	return res, nil
}

// ListAWSServices returns a list of services via Steampipe.
func (t Tool) ListAWSServices(ctx context.Context) (string, error) {
	return queryPostgres(ctx, "SELECT DISTINCT foreign_table_name FROM information_schema.foreign_tables WHERE foreign_table_schema='aws'")
}

// GetPrompt lists available prompts from available MCP servers.
func (t Tool) ListPrompts(ctx context.Context) (string, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return "", fmt.Errorf("ListPrompts: params not found in context")
	}
	var res []string
	for _, sess := range params.MCPSessions {
		for p, err := range sess.Prompts(ctx, nil) {
			if err != nil {
				return "", err
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
	return strings.Join(res, "\n"), nil
}

// GetPrompt retrieve a specific prompt by name from available MCP servers.
func (t Tool) GetPrompt(ctx context.Context, name string) (string, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return "", fmt.Errorf("GetPrompt: params not found in context")
	}
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		return "", fmt.Errorf("GetPrompt: keyVals not found in context")
	}
	for _, sess := range params.MCPSessions {
		for p, err := range sess.Prompts(ctx, nil) {
			if err != nil {
				return "", err
			}
			if name == p.Name {
				prompt, err := sess.GetPrompt(ctx, &mcp.GetPromptParams{
					Name:      name,
					Arguments: keyVals,
				})
				if err != nil {
					return "", err
				}
				var res string
				for _, msg := range prompt.Messages {
					switch msg.Content.(type) {
					case *mcp.TextContent:
						res += fmt.Sprintf("%s\n", msg.Content.(*mcp.TextContent).Text)
					}
				}
				return res, nil
			}
		}
	}
	return fmt.Sprintf("GetPrompt: prompt '%s' not found", name), nil
}

// ListResources returns resources available from MCP servers.
func (t Tool) ListResources(ctx context.Context) (string, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return "", fmt.Errorf("ListResources: params not found in context")
	}
	var res []string
	for _, sess := range params.MCPSessions {
		for r, err := range sess.Resources(ctx, nil) {
			if err != nil {
				return "", err
			}
			res = append(res, r.URI)
		}
	}
	return strings.Join(res, "\n"), nil
}
