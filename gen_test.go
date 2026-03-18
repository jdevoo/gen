package main

import (
	"context"
	"encoding/json"
	"flag"
	"regexp"
	"strings"
	"testing"
)

type OutputExpectations struct {
	Contains    []string
	NotContains []string
	Matches     []string
	IsJSON      bool
	MinLength   int
}

func prepareTestContext(t *testing.T, interactive bool, args ...string) context.Context {
	t.Helper()

	keyVals := ParamMap{}
	params := &Parameters{}

	fs := flag.NewFlagSet("test", flag.ContinueOnError)

	err := parseFlags(fs, params, &keyVals, args)
	if err != nil {
		t.Fatalf("failed to parse flags in test: %v", err)
	}

	// override from !isRedirected(os.Stdin)
	params.Interactive = interactive

	ctx := context.WithValue(context.Background(), "params", params)
	ctx = context.WithValue(ctx, "keyVals", keyVals)

	return ctx
}

func AssertOutput(t *testing.T, actual string, want OutputExpectations) {
	t.Helper()

	if len(actual) < want.MinLength {
		t.Errorf("output too short: got %d bytes, want at least %d", len(actual), want.MinLength)
	}
	for _, s := range want.Contains {
		if !strings.Contains(actual, s) {
			t.Errorf("expected output to contain %q, got: %s", s, actual)
		}
	}
	for _, s := range want.NotContains {
		if strings.Contains(actual, s) {
			t.Errorf("expected output NOT to contain %q", s)
		}
	}
	for _, pattern := range want.Matches {
		re := regexp.MustCompile(pattern)
		if !re.MatchString(actual) {
			t.Errorf("output did not match regex %q", pattern)
		}
	}
	if want.IsJSON {
		var js json.RawMessage
		if err := json.Unmarshal([]byte(actual), &js); err != nil {
			t.Errorf("output is not valid JSON: %v, got: %s", err, actual)
		}
	}
}

func TestGenContent_Basic(t *testing.T) {
	input := strings.NewReader("")
	var output strings.Builder

	ctx := prepareTestContext(t, true, "-temp", "0", "test")
	err := genContent(ctx, input, &output)
	if err != nil {
		t.Fatalf("genContent failed: %v", err)
	}

	AssertOutput(t, output.String(), OutputExpectations{
		MinLength: 20,
	})
}

func TestGenContent_Redirect(t *testing.T) {
	input := strings.NewReader("test")
	var output strings.Builder

	ctx := prepareTestContext(t, false, "-temp", "0", "-")
	err := genContent(ctx, input, &output)
	if err != nil {
		t.Fatalf("genContent failed: %v", err)
	}

	AssertOutput(t, output.String(), OutputExpectations{
		MinLength: 20,
	})
}
