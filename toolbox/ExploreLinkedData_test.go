package toolbox

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jdevoo/gen/core"
)

func TestExploreLinkedData_SELECT(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
    <variable name="p"/>
  </head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/s</uri></binding>
      <binding name="p"><literal xml:lang="en">hello</literal></binding>
    </result>
  </results>
</sparql>`))
		},
	))
	defer server.Close()

	// Test fallback to SPARQL_ENDPOINT
	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})
	ctx = context.WithValue(ctx, core.ParamsKey, &core.Parameters{})

	tTool := Tool{}
	part, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "SELECT ?s ?p WHERE { ?s ?p ?o }",
	})
	if err != nil {
		t.Fatalf("ExploreLinkedData failed: %v", err)
	}

	if part.FunctionResponse == nil {
		t.Fatal("expected function response")
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}

	text, ok := resp["text"].(string)
	if !ok {
		t.Fatal("expected text to be a string")
	}

	expectedText := "s | p\nhttp://example.org/s | hello"
	if strings.TrimSpace(text) != expectedText {
		t.Errorf("got text %q, want %q", text, expectedText)
	}
}

func TestExploreLinkedData_QueryServicePrecedence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
  </head>
  <results></results>
</sparql>`))
		},
	))
	defer server.Close()

	// Test SPARQL_QUERY_SERVICE takes precedence over SPARQL_ENDPOINT
	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_QUERY_SERVICE": server.URL,
		"SPARQL_ENDPOINT":      "http://invalid-endpoint/sparql", // should be ignored
	})
	ctx = context.WithValue(ctx, core.ParamsKey, &core.Parameters{})

	tTool := Tool{}
	part, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "SELECT ?s WHERE { ?s ?p ?o }",
	})
	if err != nil {
		t.Fatalf("ExploreLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}
}

func TestExploreLinkedData_ASK(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head></head>
  <boolean>true</boolean>
</sparql>`))
		},
	))
	defer server.Close()

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})
	ctx = context.WithValue(ctx, core.ParamsKey, &core.Parameters{})

	tTool := Tool{}
	part, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "ASK { ?s ?p ?o }",
	})
	if err != nil {
		t.Fatalf("ExploreLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}

	text := resp["text"].(string)
	expectedText := "Result: true"
	if text != expectedText {
		t.Errorf("got text %q, want %q", text, expectedText)
	}
}

func TestExploreLinkedData_CONSTRUCT(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`<http://example.org/s> <http://example.org/p> "hello" .`))
		},
	))
	defer server.Close()

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})
	ctx = context.WithValue(ctx, core.ParamsKey, &core.Parameters{})

	tTool := Tool{}
	part, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }",
	})
	if err != nil {
		t.Fatalf("ExploreLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}

	text := resp["text"].(string)
	expectedText := "subject | predicate | object\nhttp://example.org/s | http://example.org/p | hello"
	if strings.TrimSpace(text) != expectedText {
		t.Errorf("got text %q, want %q", text, expectedText)
	}
}

func TestUpdateLinkedData_Success(t *testing.T) {
	var capturedContentType string
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			capturedContentType = r.Header.Get("Content-Type")
			w.WriteHeader(http.StatusNoContent) // 204
		},
	))
	defer server.Close()

	// Test fallback to SPARQL_ENDPOINT
	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})

	tTool := Tool{}
	part, err := tTool.UpdateLinkedData(ctx, UpdateLinkedDataArgs{
		SPARQL: "INSERT DATA { <http://s> <http://p> <http://o> }",
	})
	if err != nil {
		t.Fatalf("UpdateLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}

	text := resp["text"].(string)
	if text != "Update executed successfully." {
		t.Errorf("expected confirmation string, got %q", text)
	}

	if capturedContentType != "application/sparql-update" {
		t.Errorf("expected content type application/sparql-update, got %q", capturedContentType)
	}
}

func TestUpdateLinkedData_UpdateServicePrecedence(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		},
	))
	defer server.Close()

	// Test SPARQL_UPDATE_SERVICE takes precedence over SPARQL_ENDPOINT
	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_UPDATE_SERVICE": server.URL,
		"SPARQL_ENDPOINT":        "http://invalid-endpoint/sparql", // should be ignored
	})

	tTool := Tool{}
	part, err := tTool.UpdateLinkedData(ctx, UpdateLinkedDataArgs{
		SPARQL: "CLEAR ALL",
	})
	if err != nil {
		t.Fatalf("UpdateLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}
}

func TestUpdateLinkedData_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "Malformed update syntax", http.StatusBadRequest)
		},
	))
	defer server.Close()

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})

	tTool := Tool{}
	part, err := tTool.UpdateLinkedData(ctx, UpdateLinkedDataArgs{
		SPARQL: "INSERT DATA { invalid }",
	})
	if err != nil {
		t.Fatalf("UpdateLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "ERROR" {
		t.Errorf("expected output ERROR, got %v", resp["output"])
	}

	errText := resp["error"].(string)
	if !strings.Contains(errText, "Malformed update syntax") {
		t.Errorf("unexpected error text: %q", errText)
	}
}

func TestExploreLinkedData_MissingEndpoint(t *testing.T) {
	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{})

	tTool := Tool{}
	_, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "SELECT * WHERE { ?s ?p ?o }",
	})
	if err == nil {
		t.Fatal("expected error due to missing SPARQL_ENDPOINT")
	}

	paramErr, ok := err.(*core.ParamError)
	if !ok {
		t.Fatalf("expected core.ParamError, got %T", err)
	}

	if !strings.Contains(paramErr.Error(), "missing parameter") {
		t.Errorf("unexpected error message: %v", paramErr.Error())
	}
}

func TestExploreLinkedData_JSON_Mode(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="s"/>
    <variable name="p"/>
  </head>
  <results>
    <result>
      <binding name="s"><uri>http://example.org/s</uri></binding>
      <binding name="p"><literal xml:lang="en">hello</literal></binding>
    </result>
  </results>
</sparql>`))
		},
	))
	defer server.Close()

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})
	ctx = context.WithValue(ctx, core.ParamsKey, &core.Parameters{JSON: true})

	tTool := Tool{}
	part, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "SELECT ?s ?p WHERE { ?s ?p ?o }",
	})
	if err != nil {
		t.Fatalf("ExploreLinkedData failed: %v", err)
	}

	resp := part.FunctionResponse.Response
	if resp["output"] != "SUCCESS" {
		t.Errorf("expected output SUCCESS, got %v", resp["output"])
	}

	bindings, ok := resp["bindings"].([]map[string]string)
	if !ok || len(bindings) != 1 {
		t.Fatalf("expected bindings to be []map[string]string with 1 element, got %v", resp["bindings"])
	}
	if bindings[0]["s"] != "http://example.org/s" || bindings[0]["p"] != "hello" {
		t.Errorf("unexpected bindings content: %v", bindings[0])
	}
}

func TestExploreLinkedData_UserAgent(t *testing.T) {
	var capturedUA string
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			capturedUA = r.Header.Get("User-Agent")
			_, _ = w.Write([]byte(`<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head><variable name="s"/></head>
  <results></results>
</sparql>`))
		},
	))
	defer server.Close()

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT":   server.URL,
		"SPARQL_USER_AGENT": "gen-test-agent/1.0",
	})
	ctx = context.WithValue(ctx, core.ParamsKey, &core.Parameters{})

	tTool := Tool{}
	_, err := tTool.ExploreLinkedData(ctx, ExploreLinkedDataArgs{
		SPARQL: "SELECT ?s WHERE { ?s ?p ?o }",
	})
	if err != nil {
		t.Fatalf("ExploreLinkedData failed: %v", err)
	}

	if capturedUA != "gen-test-agent/1.0" {
		t.Errorf("expected User-Agent %q, got %q", "gen-test-agent/1.0", capturedUA)
	}
}

func TestUpdateLinkedData_UserAgent(t *testing.T) {
	var capturedUA string
	server := httptest.NewServer(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			capturedUA = r.Header.Get("User-Agent")
			w.WriteHeader(http.StatusNoContent)
		},
	))
	defer server.Close()

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT":   server.URL,
		"SPARQL_USER_AGENT": "gen-update-agent/1.0",
	})

	tTool := Tool{}
	_, err := tTool.UpdateLinkedData(ctx, UpdateLinkedDataArgs{
		SPARQL: "CLEAR ALL",
	})
	if err != nil {
		t.Fatalf("UpdateLinkedData failed: %v", err)
	}

	if capturedUA != "gen-update-agent/1.0" {
		t.Errorf("expected User-Agent %q, got %q", "gen-update-agent/1.0", capturedUA)
	}
}
