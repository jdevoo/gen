package toolbox

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// GetKnownGeminiModels retrieves the list of available Gemini models.
func (t Tool) ListGeminiModels(ctx context.Context) (*genai.Part, error) {
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
		"ListGeminiModels",
		map[string]any{
			"output": "SUCCESS",
			"text":   res,
		},
	), nil
}
