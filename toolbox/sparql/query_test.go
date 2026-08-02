package sparql

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"
)

func TestQuery_Local(t *testing.T) {
	t.Run("request error", func(t *testing.T) {
		c, err := New("foo")
		if err != nil {
			t.Error(err)
			return
		}
		if _, err := c.Query(context.Background(), ""); err == nil {
			t.Errorf(`Query() error expected unsupported protocol scheme ""`)
		}
	})

	t.Run("not ok", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "", http.StatusBadRequest)
			},
		))
		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			ResultParser: NewXMLResultParser(),
		}
		if _, err := c.Query(context.Background(), ""); err == nil {
			t.Errorf("Query() error expected from empty quer")
		}
	})

	t.Run("malformed", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				_, _ = fmt.Fprint(w, "malformed")
			},
		))
		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			ResultParser: NewXMLResultParser(),
		}
		if _, err := c.Query(context.Background(), ""); err == nil {
			t.Errorf("Query() error expected from malformed")
			return
		}
	})

	t.Run("overlapping parameters", func(t *testing.T) {
		var capturedQuery string
		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				capturedQuery = r.URL.Query().Get("query")
				_, _ = fmt.Fprint(w, `<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="x"/>
  </head>
  <results></results>
</sparql>`)
			},
		))
		defer server.Close()

		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			ResultParser: NewXMLResultParser(),
		}

		_, err := c.Query(context.Background(), "$10 and $1", Param{
			Ordinal: 1,
			Value:   "one",
		}, Param{
			Ordinal: 10,
			Value:   "ten",
		})
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		// If the replacement key sorting bug is not fixed, "$1" will replace part of "$10"
		// resulting in e.g. "one0 and one" or similar corruption.
		// Expected: """ten""" and """one"""
		expected := `"""ten""" and """one"""`
		if capturedQuery != expected {
			t.Errorf("expected query %q, got %q", expected, capturedQuery)
		}
	})

	t.Run("success", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				_, _ = fmt.Fprint(w, `<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="x"/>
  </head>
  <results>
    <result> 
      <binding name="x"><bnode>r2</bnode></binding>
    </result>
  </results>
</sparql>`)
			},
		))
		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			Prefixes:     map[string]URI{"foo": "bar"},
			ResultParser: NewXMLResultParser(),
		}
		result, err := c.Query(context.Background(), "", Param{
			Ordinal: 0,
			Value:   1,
		})
		if err != nil {
			t.Errorf("Client.Query() error = %v", err)
			return
		}
		if got, want := result.Variables(), []string{"x"}; !reflect.DeepEqual(got, want) {
			t.Errorf("result.Variables() = %v, want %v", got, want)
		}
		bindings, err := result.Next()
		if err != nil {
			t.Errorf("iter.Next() error = %v", err)
			return
		}
		if want := map[string]Value{"x": BNode("r2")}; !reflect.DeepEqual(bindings, want) {
			t.Errorf("Client.Query() = %v, want %v", bindings, want)
		}
		if _, err = result.Next(); err != io.EOF {
			t.Errorf("iter.Next() error = %v", err)
			return
		}
	})

	t.Run("POST success", func(t *testing.T) {
		var capturedMethod string
		var capturedContentType string
		var capturedQuery string
		var capturedFormat string

		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				capturedMethod = r.Method
				capturedContentType = r.Header.Get("Content-Type")
				_ = r.ParseForm()
				capturedQuery = r.Form.Get("query")
				capturedFormat = r.Form.Get("format")

				_, _ = fmt.Fprint(w, `<?xml version="1.0"?>
<sparql xmlns="http://www.w3.org/2005/sparql-results#">
  <head>
    <variable name="x"/>
  </head>
  <results></results>
</sparql>`)
			},
		))
		defer server.Close()

		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			ResultParser: NewXMLResultParser(),
			Method:       "POST",
		}

		_, err := c.Query(context.Background(), "SELECT * WHERE { ?s ?p ?o }")
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}

		if capturedMethod != "POST" {
			t.Errorf("expected Method POST, got %s", capturedMethod)
		}
		if capturedContentType != "application/x-www-form-urlencoded" {
			t.Errorf("expected Content-Type application/x-www-form-urlencoded, got %s", capturedContentType)
		}
		if capturedQuery != "SELECT * WHERE { ?s ?p ?o }" {
			t.Errorf("expected query 'SELECT * WHERE { ?s ?p ?o }', got %s", capturedQuery)
		}
		if capturedFormat != "application/sparql-results+xml" {
			t.Errorf("expected format 'application/sparql-results+xml', got %s", capturedFormat)
		}
	})

	t.Run("CONSTRUCT success", func(t *testing.T) {
		var capturedFormat string

		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				capturedFormat = r.URL.Query().Get("format")
				_, _ = fmt.Fprint(w, `<http://example.org/s> <http://example.org/p> <http://example.org/o> .`)
			},
		))
		defer server.Close()

		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			ResultParser: NewXMLResultParser(),
		}

		res, err := c.Query(context.Background(), "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }")
		if err != nil {
			t.Fatalf("Query failed: %v", err)
		}
		defer res.Close()

		if capturedFormat != "text/plain" {
			t.Errorf("expected format 'text/plain', got %s", capturedFormat)
		}

		variables := res.Variables()
		wantVars := []string{"subject", "predicate", "object"}
		if !reflect.DeepEqual(variables, wantVars) {
			t.Errorf("got variables %v, want %v", variables, wantVars)
		}

		row, err := res.Next()
		if err != nil {
			t.Fatalf("Next() failed: %v", err)
		}
		wantRow := map[string]Value{
			"subject":   URI("http://example.org/s"),
			"predicate": URI("http://example.org/p"),
			"object":    URI("http://example.org/o"),
		}
		if !reflect.DeepEqual(row, wantRow) {
			t.Errorf("got row %v, want %v", row, wantRow)
		}
	})

	t.Run("UPDATE success", func(t *testing.T) {
		var capturedMethod string
		var capturedContentType string
		var capturedBody string

		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				capturedMethod = r.Method
				capturedContentType = r.Header.Get("Content-Type")
				bodyBytes, _ := io.ReadAll(r.Body)
				capturedBody = string(bodyBytes)
				w.WriteHeader(http.StatusNoContent)
			},
		))
		defer server.Close()

		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			Prefixes:     map[string]URI{"ex": "http://example.org/"},
			ResultParser: NewXMLResultParser(),
		}

		err := c.Update(context.Background(), "INSERT DATA { ex:s ex:p $1 }", Param{
			Ordinal: 1,
			Value:   "val",
		})
		if err != nil {
			t.Fatalf("UPDATE failed: %v", err)
		}

		if capturedMethod != "POST" {
			t.Errorf("expected Method POST, got %s", capturedMethod)
		}
		if capturedContentType != "application/sparql-update" {
			t.Errorf("expected Content-Type application/sparql-update, got %s", capturedContentType)
		}
		expectedBody := "PREFIX ex: <http://example.org/>\nINSERT DATA { ex:s ex:p \"\"\"val\"\"\" }"
		if capturedBody != expectedBody {
			t.Errorf("expected body %q, got %q", expectedBody, capturedBody)
		}
	})

	t.Run("UPDATE failure", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(
			func(w http.ResponseWriter, r *http.Request) {
				http.Error(w, "Malformed update operation", http.StatusBadRequest)
			},
		))
		defer server.Close()

		c := &Client{
			HTTPClient:   *server.Client(),
			Endpoint:     server.URL,
			ResultParser: NewXMLResultParser(),
		}

		err := c.Update(context.Background(), "INSERT DATA { invalid syntax }")
		if err == nil {
			t.Fatal("Expected update to fail, but succeeded")
		}
		expectedErrorMsg := "SPARQL UPDATE error. status code: 400 msg: Malformed update operation"
		if err.Error() != expectedErrorMsg {
			t.Errorf("expected error message %q, got %q", expectedErrorMsg, err.Error())
		}
	})
}

func TestQuery_Remote(t *testing.T) {
	tests := []struct {
		name     string
		query    string
		params   []Param
		expected int // use 0 for ASK
	}{
		{
			name:     "select p and o given s",
			query:    "SELECT * WHERE { <http://ja.dbpedia.org/resource/東京都> ?p ?o . } LIMIT 1",
			params:   []Param{},
			expected: 1,
		},
		{
			name:  "select s given p and ordinal parameter o",
			query: "SELECT * WHERE { ?s dbo:wikiPageID $1 . } LIMIT 1",
			params: []Param{
				{
					Ordinal: 1,
					Value:   1529557,
				},
			},
			expected: 1,
		},
		{
			name:  "select o given s with ordinal URI and p",
			query: "SELECT * WHERE { $1 <http://ja.dbpedia.org/property/genre> ?genre . } LIMIT 1",
			params: []Param{
				{
					Ordinal: 1,
					Value:   URI("http://ja.dbpedia.org/resource/ももいろクローバーZ"),
				},
			},
			expected: 1,
		},
		{
			name:  "select s given p and ordinal literal o",
			query: "SELECT * WHERE { ?s <http://ja.dbpedia.org/property/name> $1 . } LIMIT 1",
			params: []Param{
				{
					Ordinal: 1,
					Value: Literal{
						Value:       "ももいろクローバーZ",
						LanguageTag: "ja",
					},
				},
			},
			expected: 1,
		},
		{
			name:  "select s given p and ordinal URI typed literal o",
			query: "SELECT * WHERE { ?s dbo:wikiPageLength $1 . } LIMIT 1",
			params: []Param{
				{
					Ordinal: 1,
					Value: Literal{
						Value:    "10480",
						DataType: URI("http://www.w3.org/2001/XMLSchema#nonNegativeInteger"),
					},
				},
			},
			expected: 1,
		},
		{
			name:  "select s given p and ordinal typed literal (DataType) o",
			query: "SELECT * WHERE { ?s dbo:birthYear $1 . } LIMIT 1",
			params: []Param{
				{
					Ordinal: 1,
					Value: Literal{
						Value:    "1995",
						DataType: &DataType{Prefix: "xsd", Name: "gYear"},
					},
				},
			},
			expected: 1,
		},
		{
			name:  "select o given parameterized s and p",
			query: "SELECT * WHERE { $1 $2 ?o . } LIMIT 1",
			params: []Param{
				{
					Ordinal: 1,
					Value:   PrefixedName("dbpedia-ja:有安杏果"),
				},
				{
					Ordinal: 2,
					Value:   URI("http://ja.dbpedia.org/property/born"),
				},
			},
			expected: 1,
		},
		{
			name:  "ask with parameterized predicate",
			query: `ASK { dbpedia-ja:有安杏果 $1 "1995-03-15"^^xsd:date . }`,
			params: []Param{
				{
					Ordinal: 1,
					Value:   URI("http://ja.dbpedia.org/property/born"),
				},
			},
			expected: 0,
		},
	}
	cli, err := New("http://ja.dbpedia.org/sparql",
		WithHTTPClient(&http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 100,
				IdleConnTimeout:     90 * time.Second,
			},
			Timeout: 30 * time.Second,
		}),
		WithPrefix("dbpedia-ja", "http://ja.dbpedia.org/resource/"),
		WithPrefix("dbo", "http://dbpedia.org/ontology/"),
	)
	if err != nil {
		return
	}
	ctx := context.Background()
	if err := cli.Ping(ctx); err != nil {
		return
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res, err := cli.Query(ctx, tt.query, tt.params...)
			if err != nil {
				t.Fatalf("Query failed for %s: %v", tt.name, err)
			}
			n := 0
			if tt.expected == 0 {
				val, err := res.Boolean()
				if err == nil {
					t.Logf("Boolean() = %t", val)
					if !val {
						n = -1
					}
				}
			} else {
				vars := res.Variables()
				for {
					row, err := res.Next()
					if err == io.EOF {
						break
					}
					n += 1
					if err != nil {
						t.Logf("Error reading result row: %v", err)
						break
					}
					for _, v := range vars {
						val := row[v]
						t.Logf("%s: %v", v, val)
					}
				}
			}
			res.Close()
			if err != nil || n != tt.expected {
				t.Errorf("query `%s` failed", tt.name)
			}
		})
	}
}
