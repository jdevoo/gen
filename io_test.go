package main

import (
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"testing"

	"google.golang.org/genai"
)

// TestReadLine tests user input in chat mode.
func TestReadLine(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{
			input:   "Hello, world!\n",
			want:    "Hello, world!",
			wantErr: false,
		},
		{
			input:   "\n",
			want:    "",
			wantErr: false,
		},
	}

	for _, test := range tests {
		reader := strings.NewReader(test.input)
		res, err := readLine(reader)
		if (err != nil) != test.wantErr {
			t.Errorf("Did not expect error for '%s'", test.input)
			continue
		}
		if res != test.want {
			t.Errorf("Expected '%s', got '%s' for '%s'", test.want, res, test.input)
		}
	}
}

func TestEmitCandidates(t *testing.T) {
	base64Data := `/9j/4AAQSkZJRgABAQIAHAAcAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdA
SFxOQERXRTc4UG1RV19iZ2hnPk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2Nj
Y2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2P/wAARCABnAJYDASIAAhEBAxEB/8QA
HwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIh
MUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW
V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG
x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQF
BgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAV
YnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE
hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq
8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDlwKMD0pwzSiuK57QzGDxS7D6in8Y5ximnAPUfSlcq4m3ilUYp
2OKXHvRcVxnTtS7c07HNFK4DQPakC4PNOA+tOx70XAjK/So5gBGP94fzqfvUVx/qxx/EP51UXqRP4WSE
cmgjilP3jSEZqS0IO/NGDnpUiocDg/McDjvV6HTPOdVWYgsM5KcfzzQ2JySM2jp6VYu7SWzmMUwG4cgj
kMPUVBjjtTGtRu0Zopw+lFFxhinrGzuqqMsxAA9yaXFSRv5cqSEcIwYj6GpuZ30O30fSLKzhUpbpNMv3
5XGTn29BV28jt7pPLuIVljPBBFVreYx+VbqAjycgt3x14zRcNOxGyVFHQkIc/wA61exyKLbuzjdZ046d
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
`
	imgBytes, _ := base64.StdEncoding.DecodeString(base64Data)
	tests := []struct {
		name        string
		part        genai.Part
		imgModality bool
	}{
		{
			name: "Text",
			part: genai.Part{
				Text: "Text Part",
			},
			imgModality: false,
		},
		{
			name: "FunctionResponse",
			part: genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					Name:     "Function Name",
					Response: map[string]any{"Response": "Function Response"},
				},
			},
			imgModality: false,
		},
		{
			name: "InlineData",
			part: genai.Part{
				InlineData: &genai.Blob{
					Data:     imgBytes,
					MIMEType: "image/jpeg",
				},
			},
			imgModality: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cp := make([]*genai.Part, 1)
			cp[0] = &test.part
			c := make([]*genai.Candidate, 1)
			c[0] = &genai.Candidate{
				Content: &genai.Content{
					Parts: cp,
				},
				Index: 1,
			}
			_ = emitCandidates(os.Stdout, c, test.imgModality)
			fmt.Println()
		})
	}
}
