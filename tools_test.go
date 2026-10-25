package main

import (
	"context"
	"reflect"
	"testing"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

type SampleAddress struct {
	Street  string  `json:"street"`
	ZipCode *string `json:"zipCode,omitempty"`
}

type ComplexArgs struct {
	Name      string          `json:"name"`
	Age       *int            `json:"age,omitempty"`
	Tags      []string        `json:"tags"`
	Scores    []float64       `json:"scores,omitempty"`
	Address   SampleAddress   `json:"address"`
	PriorAddr *SampleAddress  `json:"priorAddr,omitempty"`
	History   []SampleAddress `json:"history,omitempty"`
}

func TestBuildFieldSchema_PrimitivesAndComplexTypes(t *testing.T) {
	structType := reflect.TypeOf(ComplexArgs{})

	tests := []struct {
		fieldName    string
		wantType     genai.Type
		wantOptional bool
		wantNullable bool
		checkExtra   func(t *testing.T, s *genai.Schema)
	}{
		{
			fieldName:    "Name",
			wantType:     genai.TypeString,
			wantOptional: false,
			wantNullable: false,
		},
		{
			fieldName:    "Age",
			wantType:     genai.TypeInteger,
			wantOptional: true,
			wantNullable: true,
		},
		{
			fieldName:    "Tags",
			wantType:     genai.TypeArray,
			wantOptional: false,
			wantNullable: false,
			checkExtra: func(t *testing.T, s *genai.Schema) {
				if s.Items == nil || s.Items.Type != genai.TypeString {
					t.Fatalf("expected Items of TypeString, got %v", s.Items)
				}
			},
		},
		{
			fieldName:    "Scores",
			wantType:     genai.TypeArray,
			wantOptional: true,
			wantNullable: false,
			checkExtra: func(t *testing.T, s *genai.Schema) {
				if s.Items == nil || s.Items.Type != genai.TypeNumber {
					t.Fatalf("expected Items of TypeNumber, got %v", s.Items)
				}
			},
		},
		{
			fieldName:    "Address",
			wantType:     genai.TypeObject,
			wantOptional: false,
			wantNullable: false,
			checkExtra: func(t *testing.T, s *genai.Schema) {
				if len(s.Properties) != 2 {
					t.Fatalf("expected 2 properties in Address, got %d", len(s.Properties))
				}
				if s.Properties["street"] == nil || s.Properties["street"].Type != genai.TypeString {
					t.Errorf("expected 'street' string property in Address")
				}
				if len(s.Required) != 1 || s.Required[0] != "street" {
					t.Errorf("expected 'street' to be required, got %v", s.Required)
				}
			},
		},
		{
			fieldName:    "PriorAddr",
			wantType:     genai.TypeObject,
			wantOptional: true,
			wantNullable: true,
		},
		{
			fieldName:    "History",
			wantType:     genai.TypeArray,
			wantOptional: true,
			wantNullable: false,
			checkExtra: func(t *testing.T, s *genai.Schema) {
				if s.Items == nil || s.Items.Type != genai.TypeObject {
					t.Fatalf("expected Items of TypeObject, got %v", s.Items)
				}
				if s.Items.Properties["street"] == nil {
					t.Errorf("expected 'street' in History array item properties")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.fieldName, func(t *testing.T) {
			field, ok := structType.FieldByName(tt.fieldName)
			if !ok {
				t.Fatalf("field %s not found", tt.fieldName)
			}
			schema, _, isOptional, err := buildFieldSchema(field)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if schema.Type != tt.wantType {
				t.Errorf("got Type %s, want %s", schema.Type, tt.wantType)
			}
			if isOptional != tt.wantOptional {
				t.Errorf("got isOptional %v, want %v", isOptional, tt.wantOptional)
			}
			if (schema.Nullable != nil && *schema.Nullable) != tt.wantNullable {
				t.Errorf("got Nullable %v, want %v", schema.Nullable != nil && *schema.Nullable, tt.wantNullable)
			}
			if tt.checkExtra != nil {
				tt.checkExtra(t, schema)
			}
		})
	}
}

func TestBuildFieldSchema_UnsupportedKind(t *testing.T) {
	type InvalidStruct struct {
		Ch chan int `json:"ch"`
	}
	field, _ := reflect.TypeOf(InvalidStruct{}).FieldByName("Ch")
	_, _, _, err := buildFieldSchema(field)
	if err == nil {
		t.Fatalf("expected error for chan kind, got nil")
	}
}

func TestRegisterGenTools(t *testing.T) {
	cfg := &genai.GenerateContentConfig{}
	err := registerGenTools(cfg)
	if err != nil {
		t.Fatalf("registerGenTools failed: %v", err)
	}
	if len(cfg.Tools) == 0 || len(cfg.Tools[0].FunctionDeclarations) == 0 {
		t.Fatalf("expected registered tool declarations, got none")
	}

	var foundCaptureScreen bool
	for _, decl := range cfg.Tools[0].FunctionDeclarations {
		if decl.Name == "CaptureScreen" {
			foundCaptureScreen = true
			if decl.Parameters == nil {
				t.Errorf("CaptureScreen should have parameters")
			}
		}
	}
	if !foundCaptureScreen {
		t.Errorf("CaptureScreen tool declaration not found")
	}
}

func TestInvokeGenTool_MissingAndTypeChecking(t *testing.T) {
	ctx := context.WithValue(context.Background(), core.ParamsKey, &core.Parameters{})

	// Non-existent tool
	part, err := invokeGenTool(ctx, &genai.FunctionCall{Name: "NoSuchTool"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if part.FunctionResponse == nil || part.FunctionResponse.Response["error"] == nil {
		t.Errorf("expected error in response for non-existent tool")
	}

	// Missing required parameter on GetMCPResource
	part, err = invokeGenTool(ctx, &genai.FunctionCall{
		Name: "GetMCPResource",
		Args: map[string]any{},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if part.FunctionResponse == nil || part.FunctionResponse.Response["error"] == nil {
		t.Errorf("expected missing parameter error, got %+v", part.FunctionResponse.Response)
	}
}
