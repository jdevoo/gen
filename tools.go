package main

import (
	"bytes"
	"context"
	"fmt"
	"image/jpeg"
	"strings"

	"github.com/kbinani/screenshot"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

type Tool struct{}

// GetKnownGeminiModels retrieves the list of available Gemini models.
func (t Tool) GetKnownGeminiModels(ctx context.Context) (*genai.Part, error) {
	var res []string
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		return nil, err
	}
	for m, err := range client.Models.All(ctx) {
		if err != nil {
			return nil, err
		}
		res = append(res, fmt.Sprintf("%s %s", m.Name, m.Description))
	}
	return genai.NewPartFromFunctionResponse(
		"GetKnownGeminiModels",
		map[string]any{"output": strings.Join(res, "\n")},
	), nil
}

// GetAWSServices returns a list of services via Steampipe.
func (t Tool) GetAWSServices(ctx context.Context) (*genai.Part, error) {
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		return nil, fmt.Errorf("queryPostgres: keyVals not found in context")
	}
	dsn, ok := keyVals["DSN"]
	if !ok || len(dsn) == 0 {
		return nil, fmt.Errorf("DSN parameter missing")
	}
	res, err := queryPostgres(ctx, "SELECT DISTINCT foreign_table_name FROM information_schema.foreign_tables WHERE foreign_table_schema='aws'", dsn)
	if err != nil {
		return nil, err
	}
	return genai.NewPartFromFunctionResponse(
		"GetAWSServices",
		map[string]any{"output": res},
	), nil
}

// ListPrompts lists available prompts from available MCP servers.
func (t Tool) ListPrompts(ctx context.Context) (*genai.Part, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return nil, fmt.Errorf("ListPrompts: params not found in context")
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
		"ListPrompts",
		map[string]any{"output": strings.Join(res, "\n")},
	), nil
}

// GetPrompt retrieves a specific prompt by name from available MCP servers.
func (t Tool) GetPrompt(ctx context.Context, name string) (*genai.Part, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return nil, fmt.Errorf("GetPrompt: params not found in context")
	}
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		return nil, fmt.Errorf("GetPrompt: keyVals not found in context")
	}
	for _, sess := range params.MCPSessions {
		for p, err := range sess.Prompts(ctx, nil) {
			if err != nil {
				continue // skip this MCP server
			}
			if name == p.Name {
				prompt, err := sess.GetPrompt(ctx, &mcp.GetPromptParams{
					Name:      name,
					Arguments: keyVals,
				})
				if err != nil {
					return nil, err
				}
				var res []string
				var parts []*genai.FunctionResponsePart
				for _, msg := range prompt.Messages {
					switch content := msg.Content.(type) {
					case *mcp.TextContent:
						res = append(res, content.Text)
					case *mcp.ResourceLink:
						parts = append(parts, genai.NewFunctionResponsePartFromURI(content.URI, content.MIMEType))
					case *mcp.EmbeddedResource:
						if content.Resource != nil {
							parts = append(parts, &genai.FunctionResponsePart{
								FileData: &genai.FunctionResponseFileData{
									FileURI:     content.Resource.URI,
									MIMEType:    content.Resource.MIMEType,
									DisplayName: content.Resource.Text,
								},
							})
						}
					}
				}
				return genai.NewPartFromFunctionResponseWithParts(
					"GetPrompt",
					map[string]any{"output": strings.Join(res, "\n")},
					parts,
				), nil
			}
		}
	}
	return genai.NewPartFromFunctionResponse(
		"GetPrompt",
		map[string]any{"output": fmt.Sprintf("GetPrompt: '%s' not found", name)},
	), nil
}

// ListResources returns resources available from MCP servers.
func (t Tool) ListResources(ctx context.Context) (*genai.Part, error) {
	params, ok := ctx.Value("params").(*Parameters)
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
		"ListResources",
		map[string]any{"output": strings.Join(res, "\n")},
	), nil
}

// GetResource retrieves a specific resource by name from available MCP servers.
func (t Tool) GetResource(ctx context.Context, name string) (*genai.Part, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return nil, fmt.Errorf("GetResource: params not found in context")
	}
	for _, sess := range params.MCPSessions {
		for r, err := range sess.Resources(ctx, nil) {
			if err != nil {
				continue // skip this MCP server
			}
			if name == r.URI {
				res, err := sess.ReadResource(ctx, &mcp.ReadResourceParams{
					URI: name,
				})
				if err != nil {
					return nil, err
				}
				for _, c := range res.Contents {
					if len(c.Text) > 0 {
						return genai.NewPartFromFunctionResponse(
							"GetResource",
							map[string]any{"output": c.Text},
						), nil
					}
					if len(c.Blob) > 0 {
						return genai.NewPartFromFunctionResponse(
							"GetResource",
							map[string]any{"output": string(c.Blob)},
						), nil
					}
				}
			}
		}
	}
	return nil, fmt.Errorf("GetResource: '%s' not found", name)
}

// Capture uses the kbinani library to capture a given screen to image
func (t Tool) CaptureScreen(ctx context.Context, n int) (*genai.Part, error) {
	bounds := screenshot.GetDisplayBounds(n)
	img, err := screenshot.CaptureRect(bounds)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, nil); err != nil {
		return nil, err
	}
	parts := []*genai.FunctionResponsePart{
		{
			InlineData: &genai.FunctionResponseBlob{
				MIMEType: "image/jpeg",
				Data:     buf.Bytes(),
			},
		},
	}
	return genai.NewPartFromFunctionResponseWithParts(
		"CaptureScreen",
		map[string]any{
			"output": "Success",
		},
		parts,
	), nil
}
