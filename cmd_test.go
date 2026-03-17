package main

import (
	"flag"
	"os"
	"testing"
	"time"

	"google.golang.org/genai"
)

func SetupFlags(fs *flag.FlagSet, params *Parameters, keyVals *ParamMap) {
	fs.BoolVar(&params.Verbose, "V", false, "")
	fs.BoolVar(&params.SegmentBackground, "b", false, "")
	fs.BoolVar(&params.ChatMode, "c", false, "")
	fs.BoolVar(&params.CodeGen, "code", false, "")
	fs.Var(&params.DigestPaths, "d", "")
	fs.BoolVar(&params.Embed, "e", false, "")
	fs.Var(&params.FilePaths, "f", "")
	fs.BoolVar(&params.GoogleSearch, "g", false, "")
	fs.BoolVar(&params.Help, "h", false, "")
	fs.BoolVar(&params.ImgModality, "img", false, "")
	fs.BoolVar(&params.JSON, "json", false, "")
	fs.IntVar(&params.K, "k", 3, "")
	fs.Float64Var(&params.Lambda, "l", 0.5, "")
	fs.Func("think", "", func(v string) error {
		params.ThinkingLevel = genai.ThinkingLevelUnspecified
		return nil
	})
	fs.StringVar(&params.GenModel, "m", "gemini-2.0-flash", "")
	fs.Var(&params.MCPServers, "mcp", "")
	fs.BoolVar(&params.OnlyKvs, "o", false, "")
	fs.Var(keyVals, "p", "")
	fs.BoolVar(&params.SystemInstruction, "s", false, "")
	fs.BoolVar(&params.Segment, "seg", false, "")
	fs.BoolVar(&params.TokenCount, "t", false, "")
	fs.Float64Var(&params.Temp, "temp", 1.0, "")
	fs.DurationVar(&params.Timeout, "timeout", 90*time.Second, "")
	fs.BoolVar(&params.Tool, "tool", false, "")
	fs.Float64Var(&params.TopP, "top_p", 0.95, "")
	fs.BoolVar(&params.Unsafe, "unsafe", false, "")
	fs.BoolVar(&params.Version, "v", false, "")
	fs.BoolVar(&params.Walk, "w", false, "")
}

func TestArgsInvalid(t *testing.T) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	testCases := []struct {
		name        string
		args        []string
		interactive bool
		expected    bool
	}{
		{
			name:        "piped content not referenced",
			args:        []string{"-f", "example.xml", "what does this code do?"},
			interactive: false,
			expected:    true,
		},
		{
			name:        "pipe content into gen",
			args:        []string{"-f", "-", "what does this code do?"},
			interactive: false,
			expected:    false,
		},
		{
			name:        "system prompt from stdin and prompt as argument",
			args:        []string{"-s", "-f", "-", "ten names for flowers"},
			interactive: false,
			expected:    false,
		},
		{
			name:        "system prompt as file and argument from pipe",
			args:        []string{"-f", "editor.sprompt", "-"},
			interactive: false,
			expected:    false,
		},
		{
			name:        "chat with input from pipe",
			args:        []string{"-c", "-"},
			interactive: false,
			expected:    false,
		},
		{
			name:        "two files and prompt",
			args:        []string{"-t", "-f", "../twitter/img/123497680.jpg", "-f", "../twitter/img/123406895.jpg", "what are the differences between these photos?"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "no flag no prompt",
			args:        []string{},
			interactive: true,
			expected:    true,
		},
		{
			name:        "-tool with prompt",
			args:        []string{"-tool", "list available models"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "system flag no prompt, no stdin",
			args:        []string{"-s"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "chat mode no prompt",
			args:        []string{"-c"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "chat mode with prompt",
			args:        []string{"-c", "ten names for money"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "-c and -s with prompt",
			args:        []string{"-s", "-c", "ten names for money"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "path to system prompt and prompt argument",
			args:        []string{"-f", "french.sprompt", "ten names for flowers"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "-code with prompt",
			args:        []string{"-code", "Levenshtein distance between Paris and Praha"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "embed and digest path but nothing to embed",
			args:        []string{"-e", "-d", "/path/to/digest"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "path to regular prompt with parameter",
			args:        []string{"-f", "/path/to/some.prompt", "-p", "key=value"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "parameterized prompt",
			args:        []string{"-p", "a=1", "-p", "b=2", "complete this sentence: replace {a} apple with {a} banana and {b} oranges for a good ..."},
			interactive: true,
			expected:    false,
		},
	}

	var params *Parameters
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset params for each test case.
			fs := flag.NewFlagSet(os.Args[0], flag.ExitOnError)

			params = &Parameters{} // Re-initialize the params struct
			keyVals := ParamMap{}  // Reset the keyVals map
			params.FilePaths = ParamArray{}
			params.DigestPaths = ParamArray{}
			SetupFlags(fs, params, &keyVals)

			if err := fs.Parse(tc.args); err != nil {
				t.Fatalf("Parse failed: %v", err)
			}
			params.Args = fs.Args()
			params.Interactive = tc.interactive

			actual := isArgsInvalid(params, keyVals)
			if (actual != nil) != tc.expected {
				t.Errorf("For test case '%s', expected %t, but got %t, Args: %v", tc.name, tc.expected, actual, tc.args)
			}
		})
	}
}
