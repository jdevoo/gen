package toolbox

import (
	"context"
	"fmt"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

// FlushCache deletes all files previously uploaded to the Gemini Files API cache.
func (t Tool) FlushCache(ctx context.Context) (*genai.Part, error) {
	client, err := core.GetGenClient(ctx)
	if err != nil {
		return genai.NewPartFromFunctionResponse("FlushCache", map[string]any{
			"output": "ERROR",
			"error":  fmt.Sprintf("FlushCache: client initialization failed: %v", err),
		}), nil
	}

	params, _ := ctx.Value(core.ParamsKey).(*core.Parameters)

	var deleted []string
	var delErrors []string

	for file, listErr := range client.Files.All(ctx) {
		if listErr != nil {
			delErrors = append(delErrors, fmt.Sprintf("listing files error: %v", listErr))
			break
		}

		if _, err := client.Files.Delete(ctx, file.Name, nil); err != nil {
			delErrors = append(delErrors, fmt.Sprintf("failed to delete %s (%s): %v", file.DisplayName, file.Name, err))
		} else {
			deleted = append(deleted, file.DisplayName)
		}
	}

	// so cleanup doesn't re-attempt deleted files
	if params != nil && params.FileURIs != nil {
		for _, name := range deleted {
			delete(params.FileURIs, name)
		}
	}

	outputStatus := "SUCCESS"
	if len(delErrors) > 0 {
		outputStatus = "ERROR"
	}

	resp := map[string]any{
		"output":       outputStatus,
		"deletedCount": len(deleted),
		"deleted":      deleted,
	}
	if len(delErrors) > 0 {
		resp["errors"] = delErrors
	}

	return genai.NewPartFromFunctionResponse("FlushCache", resp), nil
}
