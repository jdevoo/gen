package toolbox

import (
	"bytes"
	"context"
	"image/jpeg"

	"github.com/kbinani/screenshot"
	"google.golang.org/genai"
)

type CaptureScreenArgs struct {
	N *int `json:"n,omitempty"`
}

// Capture uses the kbinani library to capture a given screen to image
func (t Tool) CaptureScreen(ctx context.Context, args CaptureScreenArgs) (*genai.Part, error) {
	n := 0
	if args.N != nil {
		n = *args.N
	}
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
			"output": "SUCCESS",
		},
		parts,
	), nil
}
