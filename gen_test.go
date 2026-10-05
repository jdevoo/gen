package main

import (
	"context"
	"encoding/json"
	"flag"
	"regexp"
	"strings"
	"testing"

	"github.com/jdevoo/gen/core"
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

	keyVals := core.ParamMap{}
	params := &core.Parameters{}

	fs := flag.NewFlagSet("test", flag.ContinueOnError)

	err := parseFlags(fs, params, &keyVals, args)
	if err != nil {
		t.Fatalf("failed to parse flags in test: %v", err)
	}

	// override from !isRedirected(os.Stdin)
	params.Interactive = interactive

	ctx := context.WithValue(context.Background(), core.ParamsKey, params)
	ctx = context.WithValue(ctx, core.KeyValsKey, keyVals)

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

func TestFormatPreFlightUsage(t *testing.T) {
	if got := formatPreFlightUsage(15, 0); got != "15 tokens (prompt: 15)" {
		t.Errorf("got %q, want %q", got, "15 tokens (prompt: 15)")
	}
	if got := formatPreFlightUsage(100, 30); got != "100 tokens (prompt: 100, cached: 30)" {
		t.Errorf("got %q, want %q", got, "100 tokens (prompt: 100, cached: 30)")
	}
}

func TestTokenUsage(t *testing.T) {
	u1 := &core.TokenUsage{
		TotalTokens:         100,
		PromptTokens:        40,
		CandidatesTokens:    60,
		ThoughtsTokens:      10,
		CachedContentTokens: 5,
		ToolUsePromptTokens: 20,
	}
	u2 := &core.TokenUsage{
		TotalTokens:         200,
		PromptTokens:        80,
		CandidatesTokens:    120,
		ThoughtsTokens:      30,
		CachedContentTokens: 15,
		ToolUsePromptTokens: 40,
	}

	u1.Add(u2)
	if u1.TotalTokens != 300 || u1.PromptTokens != 120 || u1.CandidatesTokens != 180 ||
		u1.ThoughtsTokens != 40 || u1.CachedContentTokens != 20 || u1.ToolUsePromptTokens != 60 {
		t.Errorf("Add resulted in unexpected values: %+v", u1)
	}

	// test nil add doesn't panic
	u1.Add(nil)
}
