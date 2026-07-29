package main

import (
	"bytes"
	"encoding/base64"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/jdevoo/gen/core"
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
	stdout := os.Stdout
	defer func() { os.Stdout = stdout }()
	os.Stdout = os.NewFile(0, os.DevNull)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cp := make([]*genai.Part, 1)
			cp[0] = &test.part
			c := &genai.Candidate{
				Content: &genai.Content{Parts: cp},
			}
			_ = emitCandidate(os.Stdout, c, false, test.imgModality, false, nil, nil, "")
		})
	}
}

func TestLoadPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	subPromptPath := filepath.Join(tmpDir, "child"+PExt)
	subContent := "I am the child content."
	if err := os.WriteFile(subPromptPath, []byte(subContent), 0644); err != nil {
		t.Fatal(err)
	}

	mainPromptPath := filepath.Join(tmpDir, "parent"+PExt)
	mainContent := "Hello, @child" + PExt
	if err := os.WriteFile(mainPromptPath, []byte(mainContent), 0644); err != nil {
		t.Fatal(err)
	}
	t.Run("Successful Recursive Loading", func(t *testing.T) {
		seen := make(map[string]bool)
		result, err := loadPrompt(mainPromptPath, seen)
		if err != nil {
			t.Fatalf("Expected no error, got %v", err)
		}
		expected := "Hello, I am the child content."
		if strings.TrimSpace(result) != expected {
			t.Errorf("Expected %q, got %q", expected, result)
		}
	})

	t.Run("Circular Reference Detection", func(t *testing.T) {
		pathA := filepath.Join(tmpDir, "a"+PExt)
		pathB := filepath.Join(tmpDir, "b"+PExt)
		os.WriteFile(pathA, []byte("File A calls @b"+PExt), 0644)
		os.WriteFile(pathB, []byte("File B calls @a"+PExt), 0644)
		seen := make(map[string]bool)
		_, err := loadPrompt(pathA, seen)
		if err == nil {
			t.Error("Expected a circular reference error, but got nil")
		} else if !strings.Contains(err.Error(), "circular reference detected") {
			t.Errorf("Expected circular reference error message, got: %v", err)
		}
	})

	t.Run("File Not Found", func(t *testing.T) {
		seen := make(map[string]bool)
		_, err := loadPrompt(filepath.Join(tmpDir, "nonexistent.prompt"), seen)
		if err == nil {
			t.Error("Expected error for non-existent file, got nil")
		}
	})

	t.Run("Multiple References", func(t *testing.T) {
		multiPath := filepath.Join(tmpDir, "multi"+PExt)
		content := "@child" + PExt + " and again @child" + PExt
		os.WriteFile(multiPath, []byte(content), 0644)

		seen := make(map[string]bool)
		result, err := loadPrompt(multiPath, seen)
		if err != nil {
			t.Fatal(err)
		}
		expected := subContent + " and again " + subContent
		if strings.TrimSpace(result) != expected {
			t.Errorf("Expected %q, got %q", expected, result)
		}
	})
}

// TestIsEmpty tests detection of empty files.
func TestIsEmpty(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "test-empty-*.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(tmpFile.Name())
	defer tmpFile.Close()

	if !isEmpty(tmpFile) {
		t.Errorf("Expected empty file to be detected as empty")
	}

	if _, err := tmpFile.WriteString("non-empty"); err != nil {
		t.Fatal(err)
	}

	if isEmpty(tmpFile) {
		t.Errorf("Expected non-empty file to NOT be detected as empty")
	}

	// Non-*os.File should return false
	if isEmpty(nil) {
		t.Errorf("Expected non-*os.File to return false")
	}
}

// TestIsValidPath tests directory validity evaluation.
func TestIsValidPath(t *testing.T) {
	tmpDir := t.TempDir()
	if !isValidPath(tmpDir) {
		t.Errorf("Expected directory %s to be valid", tmpDir)
	}

	tmpFile := filepath.Join(tmpDir, "file.txt")
	if err := os.WriteFile(tmpFile, []byte("test"), 0644); err != nil {
		t.Fatal(err)
	}

	if isValidPath(tmpFile) {
		t.Errorf("Expected file %s NOT to be a directory (valid path in gen-cli terms means IsDir)", tmpFile)
	}
}

// TestPersistChatAndRetrieveHistory tests serialized chat history persistence.
func TestPersistChatAndRetrieveHistory(t *testing.T) {
	tmpDir := t.TempDir()

	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldWd)

	if err := os.Chdir(tmpDir); err != nil {
		t.Fatal(err)
	}

	hist := []*genai.Content{
		{
			Role: "user",
			Parts: []*genai.Part{
				{Text: "Hello!"},
			},
		},
	}

	if err := persistChat(hist); err != nil {
		t.Fatalf("persistChat failed: %v", err)
	}

	var loadedHist []*genai.Content
	if err := retrieveHistory(&loadedHist); err != nil {
		t.Fatalf("retrieveHistory failed: %v", err)
	}

	if len(loadedHist) != 1 || loadedHist[0].Role != "user" || loadedHist[0].Parts[0].Text != "Hello!" {
		t.Errorf("Retrieved history mismatch: %+v", loadedHist)
	}
}

// TestLoadPrefs tests user preferences parsing and parameter assignment.
func TestLoadPrefs(t *testing.T) {
	tmpDir := t.TempDir()
	t.Setenv("HOME", tmpDir)

	// Clean up environment variables set during this test
	t.Cleanup(func() {
		os.Unsetenv("TEST_GENRC_ENV_VAR")
	})

	rcContent := `[flags]
k = 5
lambda = 0.7
thinkinglevel = MEDIUM
temp = 1.5
timeout = 10s
topp = 0.8
embmodel = custom-emb
genmodel = custom-gen

[digestpaths]
path/to/digest1
path/to/digest2

[mcpservers]
cmd1
cmd2

[env]
TEST_GENRC_ENV_VAR = hello_from_genrc
`
	err := os.WriteFile(filepath.Join(tmpDir, DotGenRc), []byte(rcContent), 0644)
	if err != nil {
		t.Fatal(err)
	}

	params := &core.Parameters{}
	if err := loadPrefs(params); err != nil {
		t.Fatalf("loadPrefs failed: %v", err)
	}

	if params.K != 5 {
		t.Errorf("Expected K=5, got %d", params.K)
	}
	if params.Lambda != 0.7 {
		t.Errorf("Expected Lambda=0.7, got %v", params.Lambda)
	}
	if params.ThinkingLevel != genai.ThinkingLevelMedium {
		t.Errorf("Expected ThinkingLevel=MEDIUM, got %v", params.ThinkingLevel)
	}
	if params.Temp != 1.5 {
		t.Errorf("Expected Temp=1.5, got %v", params.Temp)
	}
	if params.Timeout != 10*time.Second {
		t.Errorf("Expected Timeout=10s, got %v", params.Timeout)
	}
	if params.TopP != 0.8 {
		t.Errorf("Expected TopP=0.8, got %v", params.TopP)
	}
	if params.EmbModel != "custom-emb" {
		t.Errorf("Expected EmbModel=custom-emb, got %q", params.EmbModel)
	}
	if params.GenModel != "custom-gen" {
		t.Errorf("Expected GenModel=custom-gen, got %q", params.GenModel)
	}
	if len(params.DigestPaths) != 2 || params.DigestPaths[0] != "path/to/digest1" {
		t.Errorf("Unexpected DigestPaths: %v", params.DigestPaths)
	}
	if len(params.MCPServers) != 2 || params.MCPServers[0] != "cmd1" {
		t.Errorf("Unexpected MCPServers: %v", params.MCPServers)
	}
	if val := os.Getenv("TEST_GENRC_ENV_VAR"); val != "hello_from_genrc" {
		t.Errorf("Expected TEST_GENRC_ENV_VAR=hello_from_genrc, got %q", val)
	}
}

// TestPNGAncillaryChunkStripper tests stripping ancillary chunks from PNG data streams.
func TestPNGAncillaryChunkStripper(t *testing.T) {
	t.Run("Not PNG", func(t *testing.T) {
		input := []byte("Not a PNG file")
		stripper := &PNGAncillaryChunkStripper{Reader: bytes.NewReader(input)}
		out, err := io.ReadAll(stripper)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(out, input) {
			t.Errorf("Expected identical output for non-PNG, got %s", string(out))
		}
	})

	t.Run("PNG with ancillary chunk", func(t *testing.T) {
		// Prepare a mock PNG
		magic := []byte("\x89PNG\x0D\x0A\x1A\x0A")

		// IHDR chunk (non-ancillary)
		ihdrLength := []byte{0, 0, 0, 4}
		ihdrType := []byte("IHDR")
		ihdrData := []byte("head")
		ihdrCrc := []byte{0, 0, 0, 0}

		// gAMA chunk (ancillary, type starts with lowercase 'g', chunkTypeAncillaryBit is set)
		gamaLength := []byte{0, 0, 0, 4}
		gamaType := []byte("gAMA")
		gamaData := []byte("gama")
		gamaCrc := []byte{0, 0, 0, 0}

		// IDAT chunk (non-ancillary)
		idatLength := []byte{0, 0, 0, 4}
		idatType := []byte("IDAT")
		idatData := []byte("body")
		idatCrc := []byte{0, 0, 0, 0}

		var mockPNG []byte
		mockPNG = append(mockPNG, magic...)
		mockPNG = append(mockPNG, ihdrLength...)
		mockPNG = append(mockPNG, ihdrType...)
		mockPNG = append(mockPNG, ihdrData...)
		mockPNG = append(mockPNG, ihdrCrc...)
		mockPNG = append(mockPNG, gamaLength...)
		mockPNG = append(mockPNG, gamaType...)
		mockPNG = append(mockPNG, gamaData...)
		mockPNG = append(mockPNG, gamaCrc...)
		mockPNG = append(mockPNG, idatLength...)
		mockPNG = append(mockPNG, idatType...)
		mockPNG = append(mockPNG, idatData...)
		mockPNG = append(mockPNG, idatCrc...)

		stripper := &PNGAncillaryChunkStripper{Reader: bytes.NewReader(mockPNG)}
		out, err := io.ReadAll(stripper)
		if err != nil {
			t.Fatal(err)
		}

		// Gama chunk (16 bytes: length(4), type(4), data(4), crc(4)) should be stripped
		expectedLength := len(mockPNG) - 16
		if len(out) != expectedLength {
			t.Errorf("Expected stripped length %d, got %d", expectedLength, len(out))
		}

		if bytes.Contains(out, gamaData) {
			t.Errorf("gAMA data should have been stripped")
		}
		if !bytes.Contains(out, idatData) {
			t.Errorf("IDAT data should be preserved")
		}
	})
}
