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

	ctx := context.WithValue(context.Background(), core.KeyValsKey, core.ParamMap{
		"SPARQL_ENDPOINT": server.URL,
	})

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
