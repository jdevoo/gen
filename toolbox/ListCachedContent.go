package toolbox

import (
	"context"
	"fmt"
	"time"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

// ListCachedContent lists files previously uploaded to the Gemini Files API cache.
func (t Tool) ListCachedContent(ctx context.Context) (*genai.Part, error) {
	client, err := core.GetGenClient(ctx)
	if err != nil {
		return nil, err
	}

	params, _ := ctx.Value(core.ParamsKey).(*core.Parameters)
	wantJSON := params != nil && params.JSON

	var items []map[string]any
	var textLines []string

	for file, err := range client.Files.All(ctx) {
		if err != nil {
			return nil, err
		}
		ttl := time.Until(file.ExpirationTime).Round(time.Minute)
		size := int64(0)
		if file.SizeBytes != nil {
			size = *file.SizeBytes
		}

		if wantJSON {
			items = append(items, map[string]any{
				"name":           file.Name,
				"displayName":    file.DisplayName,
				"mimeType":       file.MIMEType,
				"sizeBytes":      size,
				"uri":            file.URI,
				"expirationTime": file.ExpirationTime.Format(time.RFC3339),
			})
		} else {
			textLines = append(textLines, fmt.Sprintf("%s (%s, %d bytes, expires in %v)",
				file.DisplayName, file.Name, size, ttl))
		}
	}

	if wantJSON {
		return genai.NewPartFromFunctionResponse("ListCachedContent", map[string]any{
			"output": "SUCCESS",
			"files":  items,
		}), nil
	}

	if len(textLines) == 0 {
		return genai.NewPartFromFunctionResponse("ListCachedContent", map[string]any{
			"output": "SUCCESS",
			"text":   "Cache is empty.",
		}), nil
	}

	return genai.NewPartFromFunctionResponse("ListCachedContent", map[string]any{
		"output": "SUCCESS",
		"text":   textLines,
	}), nil
}
