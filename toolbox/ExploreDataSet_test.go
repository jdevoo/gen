package toolbox

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/jdevoo/gen/core"
)

func TestExploreDataSet(t *testing.T) {
	ctx := context.Background()
	tool := Tool{}

	// Missing DSN entirely
	argsMissingDSN := ExploreDataSetArgs{
		SQL: "SELECT 1;",
	}
	_, err := tool.ExploreDataSet(ctx, argsMissingDSN)
	if err == nil {
		t.Fatal("expected ExploreDataSet to return a Go error when DSN is missing from keyVals")
	}
	if !strings.Contains(err.Error(), "keyVals not found in context") {
		t.Errorf("expected error containing 'keyVals not found in context', got: %v", err)
	}

	// keyVals exists but DSN is not defined
	keyValsNoDSN := core.ParamMap{}
	ctxNoDSN := context.WithValue(ctx, core.KeyValsKey, keyValsNoDSN)
	_, err = tool.ExploreDataSet(ctxNoDSN, argsMissingDSN)
	if err == nil {
		t.Fatal("expected ExploreDataSet to return a Go error when DSN is missing from keyVals")
	}
	if !strings.Contains(err.Error(), "missing parameter") {
		t.Errorf("expected error containing 'missing parameter', got: %v", err)
	}

	// Connection failure (invalid port/host) - expects embedded error response for LLM to handle
	argsWithBadDSN := ExploreDataSetArgs{
		SQL: "SELECT * FROM users;",
	}
	keyVals := core.ParamMap{"DSN": "postgres://postgres:password@localhost:54321/nonexistent?sslmode=disable"}
	ctxWithBadDSN := context.WithValue(ctx, core.KeyValsKey, keyVals)
	part, err := tool.ExploreDataSet(ctxWithBadDSN, argsWithBadDSN)
	if err != nil {
		t.Fatalf("ExploreDataSet unexpectedly returned a Go error instead of embedding it in function response: %v", err)
	}
	if part == nil || part.FunctionResponse == nil {
		t.Fatal("expected function response part")
	}

	var resMap map[string]any
	jsonBytes, _ := json.Marshal(part.FunctionResponse.Response)
	json.Unmarshal(jsonBytes, &resMap)

	if resMap["output"] != "ERROR" {
		t.Errorf("expected ERROR output, got %v", resMap["output"])
	}
	if resMap["error"] == nil || resMap["error"] == "" {
		t.Fatal("expected error message to be returned to the LLM")
	}
}
