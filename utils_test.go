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

func TestParse_Basic(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "plain text without markdown",
			input:    "Hello, world!",
			expected: "\033[97mHello, world!\033[0m",
		},
		{
			name:     "single bold segment",
			input:    "Hello **world**!",
			expected: "\033[97mHello \033[1mworld\033[22m!\033[0m",
		},
		{
			name:     "multiple bold segments",
			input:    "**Hello** and **welcome**",
			expected: "\033[97m\033[1mHello\033[22m and \033[1mwelcome\033[22m\033[0m",
		},
		{
			name:     "unmatched bold opening sequence",
			input:    "This is **bold and doesn't close",
			expected: "\033[97mThis is \033[1mbold and doesn't close\033[0m",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parser := newParser()
			actual := parser.parse(tc.input)
			if actual != tc.expected {
				t.Errorf("\nExpected: %q\nActual:   %q", tc.expected, actual)
			}
		})
	}
}

func TestParse_StatefulAndBuffering(t *testing.T) {
	// 1. Test bold state carrying over to subsequent calls
	t.Run("bold persistence across calls", func(t *testing.T) {
		parser := newParser()

		// First segment starts the bold tag but does not close it
		res1 := parser.parse("This is **bold")
		expected1 := "\033[97mThis is \033[1mbold\033[0m"
		if res1 != expected1 {
			t.Errorf("First Parse failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}

		// Second segment continues the bold state and then closes it
		res2 := parser.parse(" text continuing** and normal")
		expected2 := "\033[97m\033[1m text continuing\033[22m and normal\033[0m"
		if res2 != expected2 {
			t.Errorf("Second Parse failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
	})

	// 2. Test buffering of a single trailing asterisk
	t.Run("single trailing asterisk buffer", func(t *testing.T) {
		parser := newParser()

		// First segment ends with a single '*'. It should be buffered and not rendered yet.
		res1 := parser.parse("This is a *")
		expected1 := "\033[97mThis is a \033[0m"
		if res1 != expected1 {
			t.Errorf("Buffering Parse failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}
		if parser.buffer != "*" {
			t.Errorf("Expected buffer to contain '*' but got %q", parser.buffer)
		}

		// Second segment begins with another '*' completing the bold sequence
		res2 := parser.parse("*bold text**")
		expected2 := "\033[97m\033[1mbold text\033[22m\033[0m"
		if res2 != expected2 {
			t.Errorf("Reconstructed Parse failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
		if parser.buffer != "" {
			t.Errorf("Expected buffer to be cleared but got %q", parser.buffer)
		}
	})
}

func TestParse_Concurrency(t *testing.T) {
	// Verifies that calling Parse concurrently does not cause race conditions (using the parser's internal Mutex)
	parser := newParser()
	done := make(chan bool)

	worker := func() {
		for i := 0; i < 100; i++ {
			_ = parser.parse("test **concurrency** stuff")
		}
		done <- true
	}

	go worker()
	go worker()

	<-done
	<-done
}
