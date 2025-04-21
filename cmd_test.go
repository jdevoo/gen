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
		name     string
		args     []string
		expected bool
	}{
		{
			name:     "no flag no prompt",
			args:     []string{},
			expected: false,
		},
		{
			name:     "-tool with prompt",
			args:     []string{"-tool", "list available models"},
			expected: true,
		},
		{
			name:     "system flag no prompt",
			args:     []string{"-s"},
			expected: false,
		},
		{
			name:     "chat mode no prompt",
			args:     []string{"-c"},
			expected: false,
		},
		{
			name:     "chat mode with prompt",
			args:     []string{"-c", "ten names for money"},
			expected: true,
		},
		{
			name:     "-c and -s with prompt",
			args:     []string{"-s", "-c", "ten names for money"},
			expected: true,
		},
		{
			name:     "-code with prompt",
			args:     []string{"-code", "Levenshtein distance between Paris and Praha"},
			expected: true,
		},
		{
			name:     "embed and digest path but nothing to embed",
			args:     []string{"-e", "-d", "/path/to/digest"},
			expected: false,
		},
		{
			name:     "path to regular prompt with parameter",
			args:     []string{"-f", "/path/to/some.prompt", "-p", "key=value"},
			expected: true,
		},
	}

	var params *Parameters
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset params for each test case.
			flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)
			params = &Parameters{} // Re-initialize the params struct
			flag.BoolVar(&params.Verbose, "V", false, "")
			flag.BoolVar(&params.ChatMode, "c", false, "")
			flag.BoolVar(&params.Code, "code", false, "")
			params.DigestPaths = ParamArray{}
			flag.Var(&params.DigestPaths, "d", "")
			flag.BoolVar(&params.Embed, "e", false, "")
			params.FilePaths = ParamArray{}
			flag.Var(&params.FilePaths, "f", "")
			flag.BoolVar(&params.Help, "h", false, "")
			flag.BoolVar(&params.JSON, "json", false, "")
			flag.IntVar(&params.K, "k", 3, "")
			flag.Float64Var(&params.Lambda, "l", 0.5, "")
			flag.StringVar(&params.GenModel, "m", "gemini-1.5-flash", "")
			flag.BoolVar(&params.OnlyKvs, "o", false, "")
			keyVals = ParamMap{} // Reset the keyVals map
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
			actual := isParamsValid(params)
			if actual != tc.expected {
				t.Errorf("For test case '%s', expected %t, but got %t, Args: %v", tc.name, tc.expected, actual, tc.args)
			}
		})
	}
}
