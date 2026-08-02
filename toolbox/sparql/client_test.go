package sparql

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestWithResultParser(t *testing.T) {
	resultParser := NewXMLResultParser()
	client := Client{}
	WithResultParser(resultParser)(&client)
	if got, want := resultParser, resultParser; got != want {
		t.Errorf("TestWithResultParser() = %v, want %v", got, want)
	}
}

func TestWithHTTPClient(t *testing.T) {
	timeout := 30 * time.Second
	httpClient := &http.Client{
		Timeout: timeout,
	}
	client := Client{}
	WithHTTPClient(httpClient)(&client)
	if got, want := client.HTTPClient.Timeout, timeout; got != want {
		t.Errorf("TestWithHTTPClient() = %v, want %v", got, want)
	}
}

func TestWithPrefix(t *testing.T) {
	t.Run("prefix", func(t *testing.T) {
		prefix := "dbpedia-ja"
		uri := URI("http://ja.dbpedia.org/resource/")
		client := Client{
			Prefixes: map[string]URI{},
		}
		WithPrefix(prefix, uri)(&client)
		if got, want := client.Prefixes[prefix], uri; got != want {
			t.Errorf("WithPrefix() = %v, want %v", got, want)
		}
	})

	t.Run("nil Prefixes map", func(t *testing.T) {
		prefix := "dbpedia-ja"
		uri := URI("http://ja.dbpedia.org/resource/")
		client := Client{} // Prefixes is nil
		// This should not panic
		WithPrefix(prefix, uri)(&client)
		if got, want := client.Prefixes[prefix], uri; got != want {
			t.Errorf("WithPrefix() with nil Prefixes map = %v, want %v", got, want)
		}
	})
}

func TestWithIdleConnTimeout(t *testing.T) {
	t.Run("nil Transport", func(t *testing.T) {
		client := Client{} // Transport is nil
		timeout := 15 * time.Second
		// This should not panic
		WithIdleConnTimeout(timeout)(&client)
		tr, ok := client.HTTPClient.Transport.(*http.Transport)
		if !ok {
			t.Fatalf("Expected *http.Transport, got %T", client.HTTPClient.Transport)
		}
		if got, want := tr.IdleConnTimeout, timeout; got != want {
			t.Errorf("WithIdleConnTimeout() = %v, want %v", got, want)
		}
	})
}

func TestClient_New(t *testing.T) {
	t.Run("new", func(t *testing.T) {
		endpoint := "http://ja.dbpedia.org/sparql"
		got, err := New(endpoint)
		if err != nil {
			t.Errorf("New() error = %v", err)
			return
		}
		if got.Endpoint != endpoint {
			t.Errorf("New() = %s, want %s", got.Endpoint, endpoint)
		}
	})
}

func TestClient_Close(t *testing.T) {
	t.Run("close", func(t *testing.T) {
		c := &Client{
			HTTPClient: http.Client{
				Transport: http.DefaultTransport,
			},
		}
		if err := c.Close(); err != nil {
			t.Errorf("Client.Close() error = %v", err)
		}
	})
}

func TestClient_Ping(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		wantErr bool
	}{
		{"request error", nil, true},
		{"not ok", func(w http.ResponseWriter, r *http.Request) { http.Error(w, "", http.StatusBadRequest) }, true},
		{"ok", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "ok") }, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(tt.handler)
			if tt.handler == nil {
				server.Close()
			} else {
				defer server.Close()
			}
			c, err := New(server.URL)
			if err != nil {
				t.Fatal(err)
			}
			err = c.Ping(context.Background())
			if (err != nil) != tt.wantErr {
				t.Errorf("Client.Ping() error = %v, expected error: %t", err, tt.wantErr)
			}
		})
	}
}
