package main

import (
	"bytes"
	"flag"
	"os"
	"strings"
	"testing"
)

func TestFlags(t *testing.T) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	tests := []struct {
		cmd        string
		stdin      string
		args       []string
		wantExit   int
		wantOutput []string
	}{
		{
			"gen",
			"",
			[]string{"-h"},
			0,
			[]string{"Usage:"},
		},
		{
			"gen",
			"",
			[]string{},
			1,
			[]string{"Usage:"},
		},
		{
			"gen",
			"",
			[]string{"-s", "ten names of flowers"},
			1,
			[]string{"Usage:"},
		},
		{
			"gen",
			"",
			[]string{"-c"},
			1,
			[]string{"Usage:"},
		},
		{
			"gen",
			"",
			[]string{"ten names of flowers"},
			0,
			[]string{"rose", "lily", "tulip"},
		},
		{
			"gen",
			"you understand english but always reply in french",
			[]string{"-s", "ten names of flowers"},
			0,
			[]string{"rose", "tulipe", "marguerite"},
		},
		{
			"gen",
			"you understand english but always reply in french",
			[]string{"ten names of flowers"},
			0,
			[]string{"rose", "tulipe", "marguerite"},
		},
		{
			"gen",
			"",
			[]string{"-tool", "-m", "gemini-1.5-pro", "list known models"},
			0,
			[]string{"gemini-1.5-flash"},
		},
		{
			"gen",
			"",
			[]string{"-tool", "-m", "gemini-1.5-pro", "-p", "DSN=postgres://steampipe:146f_4bc7_9c03@127.0.0.1:9193/steampipe", "retrieve AWS account IDs"},
			0,
			[]string{"["},
		},
	}

	for _, test := range tests {
		flag.CommandLine = flag.NewFlagSet(test.cmd, flag.ExitOnError)
		os.Args = append([]string{test.cmd}, test.args...)
		t.Log(os.Args)
		var out bytes.Buffer
		actualExit := emitGen(strings.NewReader(test.stdin), &out)
		if test.wantExit != actualExit {
			t.Errorf("Wrong exit code for args: %v %v, expected: %v, got: %v",
				test.stdin, test.args, test.wantExit, actualExit)
			continue
		}
		actualOutput := out.String()
		if !anyMatches([]string{actualOutput}, test.wantOutput...) {
			t.Errorf("Wrong output for args: %v %v, expected: %v, got: %v",
				test.stdin, test.args, strings.Join(test.wantOutput, ","), actualOutput)
		}
	}
}
