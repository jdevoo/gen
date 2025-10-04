package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

func TestParams(t *testing.T) {
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
			expected:    false,
		},
		{
			name:        "pipe content into gen",
			args:        []string{"-f", "-", "what does this code do?"},
			interactive: false,
			expected:    true,
		},
		{
			name:        "system prompt from stdin and prompt as argument",
			args:        []string{"-s", "-f", "-", "ten names for flowers"},
			interactive: false,
			expected:    true,
		},
		{
			name:        "system prompt as file and argument from pipe",
			args:        []string{"-f", "editor.sprompt", "-"},
			interactive: false,
			expected:    true,
		},
		{
			name:        "chat with input from pipe",
			args:        []string{"-c", "-"},
			interactive: false,
			expected:    true,
		},
		{
			name:        "two files and prompt",
			args:        []string{"-t", "-f", "../twitter/img/123497680.jpg", "-f", "../twitter/img/123406895.jpg", "what are the differences between these photos?"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "no flag no prompt",
			args:        []string{},
			interactive: true,
			expected:    false,
		},
		{
			name:        "-tool with prompt",
			args:        []string{"-tool", "list available models"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "system flag no prompt, no stdin",
			args:        []string{"-s"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "chat mode no prompt",
			args:        []string{"-c"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "chat mode with prompt",
			args:        []string{"-c", "ten names for money"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "-c and -s with prompt",
			args:        []string{"-s", "-c", "ten names for money"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "path to system prompt and prompt argument",
			args:        []string{"-f", "french.sprompt", "ten names for flowers"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "-code with prompt",
			args:        []string{"-code", "Levenshtein distance between Paris and Praha"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "embed and digest path but nothing to embed",
			args:        []string{"-e", "-d", "/path/to/digest"},
			interactive: true,
			expected:    false,
		},
		{
			name:        "path to regular prompt with parameter",
			args:        []string{"-f", "/path/to/some.prompt", "-p", "key=value"},
			interactive: true,
			expected:    true,
		},
		{
			name:        "parameterized prompt",
			args:        []string{"-p", "a=1", "-p", "b=2", "complete this sentence: replace {a} apple with {a} banana and {b} oranges for a good ..."},
			interactive: true,
			expected:    true,
		},
	}

	var params *Parameters
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset params for each test case.
			flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)
			params = &Parameters{} // Re-initialize the params struct
			keyVals = ParamMap{}   // Reset the keyVals map
			params.FilePaths = ParamArray{}
			params.DigestPaths = ParamArray{}
			flag.BoolVar(&params.Verbose, "V", false, "")
			flag.BoolVar(&params.ChatMode, "c", false, "")
			flag.BoolVar(&params.CodeGen, "code", false, "")
			flag.Var(&params.DigestPaths, "d", "")
			flag.BoolVar(&params.Embed, "e", false, "")
			flag.Var(&params.FilePaths, "f", "")
			flag.BoolVar(&params.GoogleSearch, "g", false, "")
			flag.BoolVar(&params.Help, "h", false, "")
			flag.BoolVar(&params.ImgModality, "img", false, "")
			flag.BoolVar(&params.JSON, "json", false, "")
			flag.IntVar(&params.K, "k", 3, "")
			flag.Float64Var(&params.Lambda, "l", 0.5, "")
			flag.StringVar(&params.GenModel, "m", "gemini-2.0-flash", "")
			flag.BoolVar(&params.OnlyKvs, "o", false, "")
			flag.Var(&keyVals, "p", "")
			flag.BoolVar(&params.SystemInstruction, "s", false, "")
			flag.BoolVar(&params.TokenCount, "t", false, "")
			flag.Float64Var(&params.Temp, "temp", 1.0, "")
			flag.BoolVar(&params.Tool, "tool", false, "")
			flag.Float64Var(&params.TopP, "top_p", 0.95, "")
			flag.BoolVar(&params.Unsafe, "unsafe", false, "")
			flag.BoolVar(&params.Version, "v", false, "")

			progName := filepath.Base(t.Name())
			os.Args = []string{progName}
			os.Args = append(os.Args, tc.args...)
			err := flag.CommandLine.Parse(os.Args[1:])
			if err != nil {
				t.Fatalf("Error parsing params: %v", err)
			}
			params.Args = flag.CommandLine.Args()
			params.Interactive = tc.interactive
			actual := isParamsValid(params)
			if actual != tc.expected {
				t.Errorf("For test case '%s', expected %t, but got %t, Args: %v", tc.name, tc.expected, actual, tc.args)
			}
		})
	}
}
