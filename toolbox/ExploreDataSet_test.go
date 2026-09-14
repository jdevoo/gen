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
	ctxWithBadDSN = context.WithValue(ctxWithBadDSN, core.ParamsKey, &core.Parameters{})
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

func TestParseDSN(t *testing.T) {
	tests := []struct {
		dsn        string
		wantDriver string
		wantConn   string
	}{
		{"sqlite:/path/file.db", "sqlite", "/path/file.db"},
		{"sqlite:///path/file.db", "sqlite", "/path/file.db"},
		{"sqlite:file.db", "sqlite", "file.db"},
		{"/data/app.db", "sqlite", "/data/app.db"},
		{"mydb.sqlite", "sqlite", "mydb.sqlite"},
		{"postgres://user:pass@localhost:5432/mydb", "postgres", "postgres://user:pass@localhost:5432/mydb"},
		{"postgresql://user:pass@localhost:5432/mydb", "postgres", "postgresql://user:pass@localhost:5432/mydb"},
		{"host=localhost user=pqgo dbname=test", "postgres", "host=localhost user=pqgo dbname=test"},
	}

	for _, tt := range tests {
		driver, conn, err := parseDSN(tt.dsn)
		if err != nil {
			t.Errorf("parseDSN(%q) returned unexpected error: %v", tt.dsn, err)
		}
		if driver != tt.wantDriver {
			t.Errorf("parseDSN(%q) driver = %q, want %q", tt.dsn, driver, tt.wantDriver)
		}
		if conn != tt.wantConn {
			t.Errorf("parseDSN(%q) conn = %q, want %q", tt.dsn, conn, tt.wantConn)
		}
	}
}

func TestExploreDataSet_SQLite(t *testing.T) {
	ctx := context.Background()
	tool := Tool{}
	dbPath := t.TempDir() + "/test.db"
	dsn := "sqlite:" + dbPath

	makeCtx := func(wantJSON bool) context.Context {
		c := context.WithValue(ctx, core.KeyValsKey, core.ParamMap{"DSN": dsn})
		return context.WithValue(c, core.ParamsKey, &core.Parameters{JSON: wantJSON})
	}

	// 1. DDL: CREATE TABLE
	createPart, err := tool.ExploreDataSet(makeCtx(false), ExploreDataSetArgs{
		SQL: "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
	})
	if err != nil {
		t.Fatalf("CREATE TABLE failed: %v", err)
	}
	if createPart.FunctionResponse.Response["output"] != "SUCCESS" {
		t.Fatalf("expected SUCCESS, got: %v", createPart.FunctionResponse.Response)
	}

	// 2. DML: INSERT INTO
	insertPart, err := tool.ExploreDataSet(makeCtx(false), ExploreDataSetArgs{
		SQL: "INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob');",
	})
	if err != nil {
		t.Fatalf("INSERT failed: %v", err)
	}
	if insertPart.FunctionResponse.Response["output"] != "SUCCESS" {
		t.Fatalf("expected SUCCESS, got: %v", insertPart.FunctionResponse.Response)
	}
	insertText, _ := insertPart.FunctionResponse.Response["text"].(string)
	if !strings.Contains(insertText, "Rows affected: 2") {
		t.Errorf("expected 'Rows affected: 2', got %q", insertText)
	}

	// 3. SELECT (table formatting)
	selectPart, err := tool.ExploreDataSet(makeCtx(false), ExploreDataSetArgs{
		SQL: "SELECT id, name FROM users ORDER BY id;",
	})
	if err != nil {
		t.Fatalf("SELECT failed: %v", err)
	}
	selectText, _ := selectPart.FunctionResponse.Response["text"].(string)
	expectedLines := "id | name\n1 | Alice\n2 | Bob"
	if strings.TrimSpace(selectText) != expectedLines {
		t.Errorf("got select text:\n%q\nwant:\n%q", selectText, expectedLines)
	}

	// 4. SELECT (JSON formatting)
	jsonPart, err := tool.ExploreDataSet(makeCtx(true), ExploreDataSetArgs{
		SQL: "SELECT id, name FROM users ORDER BY id;",
	})
	if err != nil {
		t.Fatalf("SELECT (JSON) failed: %v", err)
	}
	rows, ok := jsonPart.FunctionResponse.Response["rows"].([]map[string]any)
	if !ok || len(rows) != 2 {
		t.Fatalf("expected 2 JSON rows, got %v", jsonPart.FunctionResponse.Response["rows"])
	}
	if rows[0]["name"] != "Alice" || rows[1]["name"] != "Bob" {
		t.Errorf("unexpected rows content: %v", rows)
	}

	// 5. PRAGMA statement (schema introspection)
	pragmaPart, err := tool.ExploreDataSet(makeCtx(false), ExploreDataSetArgs{
		SQL: "PRAGMA table_info(users);",
	})
	if err != nil {
		t.Fatalf("PRAGMA failed: %v", err)
	}
	pragmaText, _ := pragmaPart.FunctionResponse.Response["text"].(string)
	if !strings.Contains(pragmaText, "name") || !strings.Contains(pragmaText, "type") {
		t.Errorf("expected PRAGMA output to contain table columns, got %q", pragmaText)
	}

	// 6. SQL Syntax Error returned to LLM
	errPart, err := tool.ExploreDataSet(makeCtx(false), ExploreDataSetArgs{
		SQL: "SELECT * FROM nonexistent_table;",
	})
	if err != nil {
		t.Fatalf("expected error inside part, got Go error: %v", err)
	}
	if errPart.FunctionResponse.Response["output"] != "ERROR" {
		t.Errorf("expected output ERROR, got %v", errPart.FunctionResponse.Response["output"])
	}
}
