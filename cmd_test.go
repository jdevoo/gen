package main

import (
	"bytes"
	"flag"
	"os"
	"strings"
	"testing"
)

func TestFlags(T *testing.T) {
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	tests := []struct {
		cmd        string
		stdin      string
		args       []string
		wantExit   int
		wantOutput string
	}{
		{
			"gen",
			"",
			[]string{"-h"},
			1,
			"Usage:",
		},
		{
			"gen",
			"",
			[]string{},
			1,
			"Usage:",
		},
		{
			"gen",
			"",
			[]string{"-s", "ten names for money"},
			1,
			"Usage:",
		},
		{
			"gen",
			"",
			[]string{"-c"},
			1,
			"Usage:",
		},
		{
			"gen",
			"",
			[]string{"ten names for money"},
			0,
			"Cash",
		},
		{
			"gen",
			"you understand english but always reply in french",
			[]string{"-s", "ten names for money"},
			0,
			"illets",
		},
		{
			"gen",
			"you understand english but always reply in french",
			[]string{"ten names for money"},
			0,
			"oici",
		},
	}

	for _, test := range tests {
		flag.CommandLine = flag.NewFlagSet(test.cmd, flag.ExitOnError)
		os.Args = append([]string{test.cmd}, test.args...)
		var out bytes.Buffer
		actualExit := emitGen(strings.NewReader(test.stdin), &out)
		if test.wantExit != actualExit {
			T.Errorf("Wrong exit code for args: %v %v, expected: %v, got: %v",
				test.stdin, test.args, test.wantExit, actualExit)
			continue
		}
		actualOutput := out.String()
		if !strings.Contains(actualOutput, test.wantOutput) {
			T.Errorf("Wrong output for args: %v %v, expected: %v, got: %v",
				test.stdin, test.args, test.wantOutput, actualOutput)
		}
	}
}
