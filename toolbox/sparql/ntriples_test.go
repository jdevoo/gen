package sparql

import (
	"context"
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestParseNTriplesTriple(t *testing.T) {
	tests := []struct {
		name      string
		line      string
		wantS     Value
		wantP     URI
		wantO     Value
		wantErr   bool
	}{
		{
			name:      "comment line",
			line:      "# This is a comment",
			wantS:     nil,
			wantP:     "",
			wantO:     nil,
			wantErr:   false,
		},
		{
			name:      "empty line",
			line:      "   \t  ",
			wantS:     nil,
			wantP:     "",
			wantO:     nil,
			wantErr:   false,
		},
		{
			name:      "simple URI triple",
			line:      "<http://example.org/s> <http://example.org/p> <http://example.org/o> .",
			wantS:     URI("http://example.org/s"),
			wantP:     URI("http://example.org/p"),
			wantO:     URI("http://example.org/o"),
			wantErr:   false,
		},
		{
			name:      "blank node subject",
			line:      "_:b0 <http://example.org/p> <http://example.org/o> .",
			wantS:     BNode("b0"),
			wantP:     URI("http://example.org/p"),
			wantO:     URI("http://example.org/o"),
			wantErr:   false,
		},
		{
			name:      "string literal object",
			line:      "<http://example.org/s> <http://example.org/p> \"hello world\" .",
			wantS:     URI("http://example.org/s"),
			wantP:     URI("http://example.org/p"),
			wantO:     Literal{Value: "hello world"},
			wantErr:   false,
		},
		{
			name:      "string literal with language tag",
			line:      "<http://example.org/s> <http://example.org/p> \"bonjour\"@fr .",
			wantS:     URI("http://example.org/s"),
			wantP:     URI("http://example.org/p"),
			wantO:     Literal{Value: "bonjour", LanguageTag: "fr"},
			wantErr:   false,
		},
		{
			name:      "string literal with datatype",
			line:      "<http://example.org/s> <http://example.org/p> \"10\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
			wantS:     URI("http://example.org/s"),
			wantP:     URI("http://example.org/p"),
			wantO:     Literal{Value: "10", DataType: URI("http://www.w3.org/2001/XMLSchema#integer")},
			wantErr:   false,
		},
		{
			name:      "literal with escaped quotes",
			line:      `<http://example.org/s> <http://example.org/p> "hello \"world\"" .`,
			wantS:     URI("http://example.org/s"),
			wantP:     URI("http://example.org/p"),
			wantO:     Literal{Value: `hello "world"`},
			wantErr:   false,
		},
		{
			name:      "malformed URI",
			line:      "<http://example.org/s <http://example.org/p> <http://example.org/o> .",
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, p, o, err := parseNTriplesTriple(tt.line)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseNTriplesTriple() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr {
				if !reflect.DeepEqual(s, tt.wantS) {
					t.Errorf("gotS = %v, wantS %v", s, tt.wantS)
				}
				if p != tt.wantP {
					t.Errorf("gotP = %v, wantP %v", p, tt.wantP)
				}
				if !reflect.DeepEqual(o, tt.wantO) {
					t.Errorf("gotO = %v, wantO %v", o, tt.wantO)
				}
			}
		})
	}
}

func TestNTriplesQueryResult(t *testing.T) {
	data := `
# A comment line
<http://example.org/s> <http://example.org/p> <http://example.org/o> .
# Another comment
_:b1 <http://example.org/p2> "hello"@en .
`
	ctx := context.Background()
	qr := NewNTriplesQueryResult(ctx, io.NopCloser(strings.NewReader(data)))
	defer qr.Close()

	if got, want := qr.Variables(), []string{"subject", "predicate", "object"}; !reflect.DeepEqual(got, want) {
		t.Errorf("qr.Variables() = %v, want %v", got, want)
	}

	// First row
	row1, err := qr.Next()
	if err != nil {
		t.Fatalf("Next() row 1 error = %v", err)
	}
	want1 := map[string]Value{
		"subject":   URI("http://example.org/s"),
		"predicate": URI("http://example.org/p"),
		"object":    URI("http://example.org/o"),
	}
	if !reflect.DeepEqual(row1, want1) {
		t.Errorf("row1 = %v, want %v", row1, want1)
	}

	// Second row
	row2, err := qr.Next()
	if err != nil {
		t.Fatalf("Next() row 2 error = %v", err)
	}
	want2 := map[string]Value{
		"subject":   BNode("b1"),
		"predicate": URI("http://example.org/p2"),
		"object":    Literal{Value: "hello", LanguageTag: "en"},
	}
	if !reflect.DeepEqual(row2, want2) {
		t.Errorf("row2 = %v, want %v", row2, want2)
	}

	// End
	_, err = qr.Next()
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}

	// Boolean should error
	_, err = qr.Boolean()
	if err == nil {
		t.Errorf("expected error on Boolean() for graph query result")
	}
}
