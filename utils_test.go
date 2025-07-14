package main

import (
	"testing"

	"google.golang.org/genai"
)

// TestParamMapSet tests setting prompt parameter.
func TestParamMapSet(t *testing.T) {
	tests := []struct {
		arg         string
		expected    ParamMap
		expectedErr bool
	}{
		{
			arg:         "name=John",
			expected:    ParamMap{"name": "John"},
			expectedErr: false,
		},
		{
			arg:         "invalid",
			expected:    ParamMap{},
			expectedErr: true,
		},
		{
			arg:         "missing equal",
			expected:    ParamMap{},
			expectedErr: true,
		},
		{
			arg:         "name==",
			expected:    ParamMap{"name": "="},
			expectedErr: false,
		},
		{
			arg:         "blank=",
			expected:    ParamMap{"blank": ""},
			expectedErr: false,
		},
	}

	for _, test := range tests {
		res := ParamMap{}
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
		params   ParamMap
		expected string
	}{
		{
			prompt:   "Hello {NAME}, how are you? Long live {name}!",
			params:   ParamMap{"name": "World"},
			expected: "Hello World, how are you? Long live World!",
		},
		{
			prompt:   "This is a {adjective} {noun}.",
			params:   ParamMap{"adjective": "beautiful", "noun": "day"},
			expected: "This is a beautiful day.",
		},
		{
			prompt:   "This is a test string.",
			params:   ParamMap{},
			expected: "This is a test string.",
		},
		{
			prompt:   "This is a {empty} test string.",
			params:   ParamMap{"empty": ""},
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
