package main

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

func TestFlags(t *testing.T) {
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

	var flags *Flags
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset flags for each test case.  Crucial!
			flag.CommandLine = flag.NewFlagSet(os.Args[0], flag.ExitOnError)
			flags = &Flags{} // Re-initialize the flags struct
			flag.BoolVar(&flags.Verbose, "V", false, "")
			flag.BoolVar(&flags.ChatMode, "c", false, "")
			flag.BoolVar(&flags.Code, "code", false, "")
			flags.DigestPaths = ParamArray{}
			flag.Var(&flags.DigestPaths, "d", "")
			flag.BoolVar(&flags.Embed, "e", false, "")
			flags.FilePaths = ParamArray{}
			flag.Var(&flags.FilePaths, "f", "")
			flag.BoolVar(&flags.Help, "h", false, "")
			flag.BoolVar(&flags.JSON, "json", false, "")
			flag.IntVar(&flags.K, "k", 3, "")
			flag.Float64Var(&flags.Lambda, "l", 0.5, "")
			flag.StringVar(&flags.GenModel, "m", "gemini-1.5-flash", "")
			flag.BoolVar(&flags.OnlyKvs, "o", false, "")
			keyVals = ParamMap{} // Reset the keyVals map
			flag.Var(&keyVals, "p", "")
			flag.BoolVar(&flags.SystemInstruction, "s", false, "")
			flag.BoolVar(&flags.TokenCount, "t", false, "")
			flag.Float64Var(&flags.Temp, "temp", 1.0, "")
			flag.BoolVar(&flags.Tool, "tool", false, "")
			flag.Float64Var(&flags.TopP, "top_p", 0.95, "")
			flag.BoolVar(&flags.Unsafe, "unsafe", false, "")
			flag.BoolVar(&flags.Version, "v", false, "")

			progName := filepath.Base(t.Name())
			os.Args = []string{progName}
			os.Args = append(os.Args, tc.args...)
			err := flag.CommandLine.Parse(os.Args[1:])
			if err != nil {
				t.Fatalf("Error parsing flags: %v", err)
			}

			actual := isValidFlagSet(flags)
			if actual != tc.expected {
				t.Errorf("For test case '%s', expected %t, but got %t, Args: %v", tc.name, tc.expected, actual, tc.args)
			}
		})
	}
}
