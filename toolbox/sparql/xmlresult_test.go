package sparql

import (
	"context"
	"encoding/xml"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestDecodeXMLQueryResult(t *testing.T) {
	ctx := context.Background()
	t.Run("empty", func(t *testing.T) {
		reader := io.NopCloser(strings.NewReader(``))
		if _, err := DecodeXMLQueryResult(ctx, reader); err != io.EOF {
			t.Errorf("DecodeXMLQueryResult() error = %v", err)
			return
		}
	})
	t.Run("success", func(t *testing.T) {
		reader := io.NopCloser(strings.NewReader(`<head></head>`))
		got, err := DecodeXMLQueryResult(ctx, reader)
		if err != nil {
			t.Errorf("DecodeXMLQueryResult() error = %v", err)
			return
		}
		if got, want := got.Variables(), []string{}; !reflect.DeepEqual(got, want) {
			t.Errorf("DecodeXMLQueryResult() = %v, want %v", got, want)
		}
	})
}

func TestDecodeVariables(t *testing.T) {
	t.Run("bad XML", func(t *testing.T) {
		decoder := xml.NewDecoder(strings.NewReader(`<head><variable></head>`))
		_, err := decodeVariables(decoder)
		if _, ok := err.(*xml.SyntaxError); !ok {
			t.Errorf("decodeVariables() error = %v", err)
			return
		}
	})
	t.Run("success", func(t *testing.T) {
		decoder := xml.NewDecoder(strings.NewReader(`<head><variable name="foo" /></head>`))
		got, err := decodeVariables(decoder)
		if err != nil {
			t.Errorf("decodeVariables() error = %v", err)
			return
		}
		want := []string{"foo"}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("decodeVariables() = %v, want %v", got, want)
		}
		if _, err := decodeVariables(decoder); err != io.EOF {
			t.Errorf("decodeVariables() error = %v", err)
			return
		}
	})
}

func TestXMLQueryResult_Variables(t *testing.T) {
	x := &XMLQueryResult{
		variables: []string{"foo"},
	}
	want := []string{"foo"}
	if got := x.Variables(); !reflect.DeepEqual(got, want) {
		t.Errorf("XMLQueryResult.Variables() = %v, want %v", got, want)
	}
}

func TestXMLQueryResult_Next(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		x := &XMLQueryResult{
			ctx:       context.Background(),
			decoder:   xml.NewDecoder(strings.NewReader(``)),
			variables: []string{"x"}, // Must set variables so it doesn't fail fast as an ASK result
		}
		if _, err := x.Next(); err != io.EOF {
			t.Errorf("XMLQueryResult.Next() error = %v", err)
			return
		}
	})
	t.Run("success", func(t *testing.T) {
		xmlData := `<sparql xmlns="http://www.w3.org/2005/sparql-results#">                                                     
  <results>                                                                                                                             
    <result>                                                                                                                            
      <binding name="x">                                                                                                                
        <bnode>r2</bnode>                                                                                                               
      </binding>                                                                                                                        
    </result>                                                                                                                           
  </results>                                                                                                                            
</sparql>`
		x := &XMLQueryResult{
			ctx:       context.Background(),
			decoder:   xml.NewDecoder(strings.NewReader(xmlData)),
			variables: []string{"x"},
		}
		got, err := x.Next()
		if err != nil {
			t.Errorf("XMLQueryResult.Next() error = %v", err)
			return
		}
		want := map[string]Value{"x": BNode("r2")}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("XMLQueryResult.Next() = %v, want %v", got, want)
		}
	})
	t.Run("Next on ASK result", func(t *testing.T) {
		x := &XMLQueryResult{
			ctx:       context.Background(),
			decoder:   xml.NewDecoder(strings.NewReader("")),
			variables: []string{}, // ASK results have 0 variables
		}
		_, err := x.Next()
		if err == nil || !strings.Contains(err.Error(), "Next() called on an ASK query result") {
			t.Errorf("Expected Next() on ASK query result to fail, got err = %v", err)
		}
	})
	t.Run("malformed XML", func(t *testing.T) {
		x := &XMLQueryResult{
			ctx:       context.Background(),
			decoder:   xml.NewDecoder(strings.NewReader(`<result><binding name="x"><bnode>r2</bnode>`)),
			variables: []string{"x"},
		}
		_, err := x.Next()
		if err == nil {
			t.Error("Expected an error for malformed XML, but got none")
		}
	})
}

func TestDecodeResult(t *testing.T) {
	tests := []struct {
		name     string
		xmlInput string
		size     int
		want     map[string]Value
		wantErr  error
	}{
		{
			name:     "empty",
			xmlInput: `<result></result>`,
			size:     0,
			want:     make(map[string]Value),
			wantErr:  nil,
		},
		{
			name:     "bad result",
			xmlInput: `<result>`,
			size:     0,
			want:     nil,
			wantErr:  &xml.SyntaxError{},
		},
		{
			name:     "bad binding",
			xmlInput: `<result><binding></result>`,
			size:     0,
			want:     nil,
			wantErr:  &xml.SyntaxError{},
		},
		{
			name: "success",
			xmlInput: `<result>
<binding name="x"><bnode>r2</bnode></binding>
<binding name="hpage"><uri>http://work.example.org/bob/</uri></binding>
<binding name="name"><literal xml:lang="en">Bob</literal></binding>
<binding name="age"><literal datatype="http://www.w3.org/2001/XMLSchema#integer">30</literal></binding>
<binding name="mbox"><uri>mailto:bob@work.example.org</uri></binding>
</result>`,
			size: 5,
			want: map[string]Value{
				"x":     BNode("r2"),
				"hpage": URI("http://work.example.org/bob/"),
				"name": Literal{
					Value:       "Bob",
					LanguageTag: "en",
				},
				"age": Literal{
					Value:    "30",
					DataType: URI("http://www.w3.org/2001/XMLSchema#integer"),
				},
				"mbox": URI("mailto:bob@work.example.org"),
			},
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decoder := xml.NewDecoder(strings.NewReader(tt.xmlInput))
			if _, err := decoder.Token(); err != nil {
				t.Fatalf("Unexpected error during tokenization: %v", err)
			}
			got, err := decodeResult(decoder, tt.size)
			if !errorsMatch(err, tt.wantErr) {
				t.Errorf("decodeResult() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("decodeResult() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDecodeBinding(t *testing.T) {
	tests := []struct {
		name      string
		xmlInput  string
		wantName  string
		wantValue Value
		wantErr   error
	}{
		{
			name:      "empty",
			xmlInput:  `<binding name="x"></binding>`,
			wantName:  "",
			wantValue: nil,
			wantErr:   io.EOF,
		},
		{
			name:      "bnode",
			xmlInput:  `<binding name="x"><bnode>r2</bnode></binding>`,
			wantName:  "x",
			wantValue: BNode("r2"),
			wantErr:   nil,
		},
		{
			name:      "bad bnode",
			xmlInput:  `<binding name="x"><bnode>r2</binding>`,
			wantName:  "",
			wantValue: nil,
			wantErr:   &xml.SyntaxError{},
		},
		{
			name: "URI",
			xmlInput: `<binding name="x">
<uri>http://work.example.org/bob/</uri>
</binding>`,
			wantName:  "x",
			wantValue: URI("http://work.example.org/bob/"),
			wantErr:   nil,
		},
		{
			name: "bad URI",
			xmlInput: `<binding name="x">
<uri>http://work.example.org/bob/
</binding>`,
			wantName:  "",
			wantValue: nil,
			wantErr:   &xml.SyntaxError{},
		},
		{
			name: "literal with language attribute",
			xmlInput: `<binding name="x">
<literal xml:lang="en">Bob</literal>
</binding>`,
			wantName: "x",
			wantValue: Literal{
				Value:       "Bob",
				LanguageTag: "en",
			},
			wantErr: nil,
		},
		{
			name: "literal with data type",
			xmlInput: `<binding name="x">
<literal datatype="http://www.w3.org/2001/XMLSchema#integer">30</literal>
</binding>`,
			wantName: "x",
			wantValue: Literal{
				Value:    "30",
				DataType: URI("http://www.w3.org/2001/XMLSchema#integer"),
			},
			wantErr: nil,
		},
		{
			name: "literal with data type and language attribute",
			xmlInput: `<binding name="x">
<literal xml:lang="en" datatype="http://www.w3.org/2001/XMLSchema#string">foo</literal>
</binding>`,
			wantName: "x",
			wantValue: Literal{
				Value:       "foo",
				DataType:    URI("http://www.w3.org/2001/XMLSchema#string"),
				LanguageTag: "en",
			},
			wantErr: nil,
		},
		{
			name: "bad literal",
			xmlInput: `<binding name="x">
<literal>foo
</binding>`,
			wantName:  "",
			wantValue: nil,
			wantErr:   &xml.SyntaxError{},
		},
		{
			name: "unknown binding",
			xmlInput: `<binding name="x">
<foo>bar</foo>
</binding>`,
			wantName:  "",
			wantValue: nil,
			wantErr:   errors.New("unknown binding foo"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decoder := xml.NewDecoder(strings.NewReader(tt.xmlInput))
			token, err := decoder.Token()
			if err != nil {
				t.Fatalf("Unexpected error during tokenization: %v", err)
			}
			element := token.(xml.StartElement)
			gotName, gotValue, err := decodeBinding(decoder, &element)
			if !errorsMatch(err, tt.wantErr) {
				t.Errorf("decodeResult() error = %v, wantErr %v", err, tt.wantErr)
			}
			if gotName != tt.wantName {
				t.Errorf("decodeBinding() gotName = %v, wantName %v", gotName, tt.wantName)
			}
			if !reflect.DeepEqual(gotValue, tt.wantValue) {
				t.Errorf("decodeBinding() gotValue = %v, wantValue %v", gotValue, tt.wantValue)
			}
		})
	}
}

func TestXMLQueryResult_Boolean(t *testing.T) {
	tests := []struct {
		name     string
		xmlInput string
		want     bool
		wantErr  error
	}{
		{
			name:     "empty",
			xmlInput: ``,
			want:     false,
			wantErr:  io.EOF,
		},
		{
			name:     "bad XML",
			xmlInput: `<boolean>`,
			want:     false,
			wantErr:  &xml.SyntaxError{},
		},
		{
			name:     "success",
			xmlInput: `<boolean>true</boolean>`,
			want:     true,
			wantErr:  nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := &XMLQueryResult{
				ctx:       context.Background(),
				decoder:   xml.NewDecoder(strings.NewReader(tt.xmlInput)),
				variables: []string{}, // ASK results have 0 variables
			}
			got, err := x.Boolean()
			if !errorsMatch(err, tt.wantErr) {
				t.Errorf("XMLQueryResult.Boolean() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("XMLQueryResult.Boolean() = %v, want %v", got, tt.want)
			}
		})
	}

	t.Run("Boolean on SELECT result", func(t *testing.T) {
		x := &XMLQueryResult{
			ctx:       context.Background(),
			decoder:   xml.NewDecoder(strings.NewReader("")),
			variables: []string{"x"}, // SELECT results have variables
		}
		_, err := x.Boolean()
		if err == nil || !strings.Contains(err.Error(), "Boolean() called on a SELECT query result") {
			t.Errorf("Expected Boolean() on SELECT query result to fail, got err = %v", err)
		}
	})
}

func TestXMLQueryResult_Close(t *testing.T) {
	x := &XMLQueryResult{
		ctx: context.Background(),
		r:   io.NopCloser(strings.NewReader("")),
	}
	if err := x.Close(); err != nil {
		t.Errorf("XMLQueryResult.Close() error = %v", err)
	}
}

// Helper function to compare errors, handling nil and type assertions.
func errorsMatch(err1, err2 error) bool {
	if err1 == nil && err2 == nil {
		return true
	}
	if err1 == nil || err2 == nil {
		return false
	}
	if _, ok := err2.(*xml.SyntaxError); ok {
		_, ok2 := err1.(*xml.SyntaxError)
		return ok2
	}
	return reflect.TypeOf(err1) == reflect.TypeOf(err2)
}
