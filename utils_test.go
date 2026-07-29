package main

import (
	"testing"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

// TestParamMapSet tests setting prompt parameter.
func TestParamMapSet(t *testing.T) {
	tests := []struct {
		arg         string
		expected    core.ParamMap
		expectedErr bool
	}{
		{
			arg:         "name=John",
			expected:    core.ParamMap{"name": "John"},
			expectedErr: false,
		},
		{
			arg:         "invalid",
			expected:    core.ParamMap{},
			expectedErr: true,
		},
		{
			arg:         "missing equal",
			expected:    core.ParamMap{},
			expectedErr: true,
		},
		{
			arg:         "name==",
			expected:    core.ParamMap{"name": "="},
			expectedErr: false,
		},
		{
			arg:         "blank=",
			expected:    core.ParamMap{"blank": ""},
			expectedErr: false,
		},
	}

	for _, test := range tests {
		res := core.ParamMap{}
		err := res.Set(test.arg)
		if (err != nil) != test.expectedErr {
			t.Errorf("Set(%q) error = %v, expectedErr %v", test.arg, err, test.expectedErr)
			continue
		}
		if test.expected["name"] != res["name"] {
			t.Errorf("Expected 'name' to be '%s', got '%s'", test.expected["name"], res["name"])
		}
	}
}

// TestSearchReplace tests the searchReplace function.
func TestSearchReplace(t *testing.T) {
	tests := []struct {
		prompt   string
		params   core.ParamMap
		expected string
	}{
		{
			prompt:   "Hello {NAME}, how are you? Long live {name}!",
			params:   core.ParamMap{"name": "World"},
			expected: "Hello World, how are you? Long live World!",
		},
		{
			prompt:   "This is a {adjective} {noun}.",
			params:   core.ParamMap{"adjective": "beautiful", "noun": "day"},
			expected: "This is a beautiful day.",
		},
		{
			prompt:   "This is a test string.",
			params:   core.ParamMap{},
			expected: "This is a test string.",
		},
		{
			prompt:   "This is a {empty} test string.",
			params:   core.ParamMap{"empty": ""},
			expected: "This is a  test string.",
		},
	}

	for _, test := range tests {
		result := searchReplace(test.prompt, test.params)
		if result != test.expected {
			t.Errorf("Expected '%s', got '%s'", test.expected, result)
		}
	}
}

// TestAnyMatches tests list inclusion.
func TestAnyMatches(t *testing.T) {
	tests := []struct {
		inputArray []string
		inputCand  []string
		expected   bool
	}{
		{
			inputArray: []string{},
			inputCand:  []string{".prompt"},
			expected:   false,
		},
		{
			inputArray: []string{"image.png", "my.prompt"},
			inputCand:  []string{".prompt"},
			expected:   true,
		},
		{
			inputArray: []string{"my.sprompt"},
			inputCand:  []string{".prompt"},
			expected:   false,
		},
	}

	for _, test := range tests {
		res := anyMatches(test.inputArray, test.inputCand...)
		if test.expected != res {
			t.Errorf("Expected %t, got %t", test.expected, res)
		}
	}
}

// TestAnyMatch tests list inclusion.
func TestAllMatch(t *testing.T) {
	tests := []struct {
		inputArray []string
		inputCand  string
		expected   bool
	}{
		{
			inputArray: []string{},
			inputCand:  ".prompt",
			expected:   false,
		},
		{
			inputArray: []string{"image.png", "my.prompt"},
			inputCand:  ".prompt",
			expected:   false,
		},
		{
			inputArray: []string{"my.sprompt"},
			inputCand:  ".sprompt",
			expected:   true,
		},
	}

	for _, test := range tests {
		res := allMatch(test.inputArray, test.inputCand)
		if test.expected != res {
			t.Errorf("Expected %t, got %t for %v", test.expected, res, test.inputArray)
		}
	}
}

// TestOneMatches tests list inclusion.
func TestOneMatches(t *testing.T) {
	tests := []struct {
		inputArray []string
		inputCand  string
		expected   bool
	}{
		{
			inputArray: []string{},
			inputCand:  "-",
			expected:   false,
		},
		{
			inputArray: []string{"-", "my.prompt"},
			inputCand:  ".prompt",
			expected:   true,
		},
		{
			inputArray: []string{"my.sprompt"},
			inputCand:  "-",
			expected:   false,
		},
	}

	for _, test := range tests {
		res := oneMatches(test.inputArray, test.inputCand)
		if test.expected != res {
			t.Errorf("Expected %t, got %t for %v", test.expected, res, test.inputArray)
		}
	}
}

// TestPartHasKey tests parts with {digest}.
func TestPartHasKey(t *testing.T) {
	tests := []struct {
		inputParts []*genai.Part
		inputKey   string
		expected   int
	}{
		{
			inputParts: []*genai.Part{
				{Text: "some prompt without key"},
				{Text: "some other prompt without key"},
			},
			inputKey: "{digest}",
			expected: -1,
		},
		{
			inputParts: []*genai.Part{
				{Text: "some prompt with {digest}"},
			},
			inputKey: "{digest}",
			expected: 0,
		},
		{
			inputParts: []*genai.Part{
				{Text: "some prompt without key}"},
				{Text: "some prompt with {digest}"},
			},
			inputKey: "{digest}",
			expected: 1,
		},
	}

	for _, test := range tests {
		res := partWithKey(test.inputParts, test.inputKey)
		if test.expected != res {
			t.Errorf("Expected %d, got %d for %v", test.expected, res, test.inputParts)
		}
	}
}

// TestReplacePart tests {digest} substitution.
func TestReplacePart(t *testing.T) {
	tests := []struct {
		inputParts []*genai.Part
		inputIdx   int
		inputKey   string
		inputVal   []QueryResult
		expected   []*genai.Part
	}{
		{
			inputParts: []*genai.Part{
				{Text: "prompt with key in first position {digest}"},
				{Text: "other prompt without key"},
				{Text: "yet another prompt without key"},
			},
			inputIdx: 0,
			inputKey: "{digest}",
			inputVal: []QueryResult{
				{
					Document{
						nil,
						"bla",
						nil,
					},
					0,
				},
			},
			expected: []*genai.Part{
				{Text: "prompt with key in first position bla"},
				{Text: "other prompt without key"},
				{Text: "yet another prompt without key"},
			},
		},
		{
			inputParts: []*genai.Part{
				{Text: "other prompt without key"},
				{Text: "yet another prompt without key"},
				{Text: "prompt with key in last position {digest}"},
			},
			inputIdx: 2,
			inputKey: "{digest}",
			inputVal: []QueryResult{
				{
					Document{
						nil,
						"bla",
						nil,
					},
					0,
				},
			},
			expected: []*genai.Part{
				{Text: "other prompt without key"},
				{Text: "yet another prompt without key"},
				{Text: "prompt with key in last position bla"},
			},
		},
	}

	for _, test := range tests {
		replacePart(&test.inputParts, test.inputIdx, test.inputKey, test.inputVal)
		for idx := range test.inputParts {
			if test.inputParts[idx].Text != test.expected[idx].Text {
				t.Errorf("Expected '%s', got '%s'", test.expected[idx].Text, test.inputParts[idx].Text)
				break
			}
		}
	}
}

// TestConjTexts tests the conjTexts function.
func TestConjTexts(t *testing.T) {
	tests := []struct {
		name     string
		input    []*genai.Part
		expected []*genai.Part
	}{
		{
			name:     "empty",
			input:    []*genai.Part{},
			expected: []*genai.Part{},
		},
		{
			name: "multiple text parts",
			input: []*genai.Part{
				{Text: "Hello "},
				{Text: "World!"},
			},
			expected: []*genai.Part{
				{Text: "Hello World!"},
			},
		},
		{
			name: "text with some empty parts",
			input: []*genai.Part{
				{Text: "A"},
				{Text: ""},
				{Text: "B"},
			},
			expected: []*genai.Part{
				{Text: "AB"},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := tc.input
			conjTexts(&input)
			if len(input) != len(tc.expected) {
				t.Fatalf("Expected length %d, got %d", len(tc.expected), len(input))
			}
			for i := range input {
				if input[i].Text != tc.expected[i].Text {
					t.Errorf("Expected element %d to be %q, got %q", i, tc.expected[i].Text, input[i].Text)
				}
			}
		})
	}
}

// TestPrependToParts tests prepending documents.
func TestPrependToParts(t *testing.T) {
	parts := []*genai.Part{
		{Text: "Existing"},
	}
	selection := []QueryResult{
		{doc: Document{content: "Doc1"}},
		{doc: Document{content: "Doc2"}},
	}
	prependToParts(&parts, selection)
	if len(parts) != 3 {
		t.Fatalf("Expected length 3, got %d", len(parts))
	}
	if parts[0].Text != "Doc1" || parts[1].Text != "Doc2" || parts[2].Text != "Existing" {
		t.Errorf("Unexpected parts after prepend: %v, %v, %v", parts[0].Text, parts[1].Text, parts[2].Text)
	}
}

// TestAppendToSelection tests extending selections by MMR order.
func TestAppendToSelection(t *testing.T) {
	selection := []QueryResult{
		{doc: Document{content: "Doc1"}, mmr: 0.5},
		{doc: Document{content: "Doc2"}, mmr: 0.8},
	}
	item := QueryResult{doc: Document{content: "Doc3"}, mmr: 0.9}

	// append with limit k = 2
	res := appendToSelection(selection, item, 2)
	if len(res) != 2 {
		t.Fatalf("Expected length 2, got %d", len(res))
	}
	if res[0].mmr != 0.9 || res[1].mmr != 0.8 {
		t.Errorf("Expected sorted descending MMR (0.9, 0.8), got: %v, %v", res[0].mmr, res[1].mmr)
	}
}

// TestZeroOrOneMatches tests zeroOrOneMatches.
func TestZeroOrOneMatches(t *testing.T) {
	tests := []struct {
		strArray []string
		cand     string
		expected bool
	}{
		{[]string{"apple", "banana"}, "cherry", true},
		{[]string{"apple", "banana"}, "apple", true},
		{[]string{"apple", "apple", "banana"}, "apple", false},
	}
	for _, tc := range tests {
		actual := zeroOrOneMatches(tc.strArray, tc.cand)
		if actual != tc.expected {
			t.Errorf("zeroOrOneMatches(%v, %q) = %t, expected %t", tc.strArray, tc.cand, actual, tc.expected)
		}
	}
}

// TestAlignSchema tests schema alignment function.
func TestAlignSchema(t *testing.T) {
	schema := map[string]any{
		"$schema":     "http://json-schema.org/draft-07/schema#",
		"definitions": map[string]any{},
		"$ref":        "#/definitions/foo",
		"type":        []any{"string", "null"},
		"properties": map[string]any{
			"prop1": map[string]any{
				"type": []any{"integer", "null"},
			},
		},
		"items": map[string]any{
			"type": []any{"boolean", "null"},
		},
		"additionalProperties": map[string]any{
			"type": "string",
		},
	}

	alignSchema(schema)

	if _, ok := schema["$schema"]; ok {
		t.Error("$schema was not deleted")
	}
	if _, ok := schema["definitions"]; ok {
		t.Error("definitions was not deleted")
	}
	if _, ok := schema["$ref"]; ok {
		t.Error("$ref was not deleted")
	}
	if val, ok := schema["type"].(string); !ok || val != "string" {
		t.Errorf("expected type to be string, got %v", schema["type"])
	}
	if val, ok := schema["nullable"].(bool); !ok || !val {
		t.Errorf("expected nullable to be true, got %v", schema["nullable"])
	}

	// Check properties recursion
	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("properties missing or incorrect type")
	}
	prop1, ok := props["prop1"].(map[string]any)
	if !ok {
		t.Fatal("prop1 missing or incorrect type")
	}
	if val, ok := prop1["type"].(string); !ok || val != "integer" {
		t.Errorf("expected prop1 type to be integer, got %v", prop1["type"])
	}
	if val, ok := prop1["nullable"].(bool); !ok || !val {
		t.Errorf("expected prop1 nullable to be true, got %v", prop1["nullable"])
	}

	// Check items recursion
	items, ok := schema["items"].(map[string]any)
	if !ok {
		t.Fatal("items missing or incorrect type")
	}
	if val, ok := items["type"].(string); !ok || val != "boolean" {
		t.Errorf("expected items type to be boolean, got %v", items["type"])
	}
	if val, ok := items["nullable"].(bool); !ok || !val {
		t.Errorf("expected items nullable to be true, got %v", items["nullable"])
	}

	// alignSchema with nil
	alignSchema(nil) // should not panic
}

// TestIsValidPart tests part validation checker.
func TestIsValidPart(t *testing.T) {
	tests := []struct {
		name     string
		part     *genai.Part
		expected bool
	}{
		{"nil", nil, false},
		{"empty", &genai.Part{}, false},
		{"text", &genai.Part{Text: "hello"}, true},
		{"inline data empty", &genai.Part{InlineData: &genai.Blob{}}, false},
		{"inline data valid", &genai.Part{InlineData: &genai.Blob{Data: []byte{1, 2, 3}}}, true},
		{"file data empty", &genai.Part{FileData: &genai.FileData{}}, false},
		{"file data valid", &genai.Part{FileData: &genai.FileData{FileURI: "gs://uri"}}, true},
		{"function call empty", &genai.Part{FunctionCall: &genai.FunctionCall{}}, false},
		{"function call valid", &genai.Part{FunctionCall: &genai.FunctionCall{Name: "my_func"}}, true},
		{"function response empty", &genai.Part{FunctionResponse: &genai.FunctionResponse{}}, false},
		{"function response valid", &genai.Part{FunctionResponse: &genai.FunctionResponse{Name: "my_func"}}, true},
		{"code exec result empty", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{}}, false},
		{"code exec result valid output", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Output: "ok"}}, true},
		{"code exec result valid outcome", &genai.Part{CodeExecutionResult: &genai.CodeExecutionResult{Outcome: "ok"}}, true},
		{"executable code empty", &genai.Part{ExecutableCode: &genai.ExecutableCode{}}, false},
		{"executable code valid", &genai.Part{ExecutableCode: &genai.ExecutableCode{Code: "fmt.Println()"}}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := isValidPart(tc.part)
			if actual != tc.expected {
				t.Errorf("isValidPart() = %t, expected %t", actual, tc.expected)
			}
		})
	}
}
