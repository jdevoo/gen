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

// TestMarkdownParser contains all test cases for the Parse method.
func TestMarkdownParser(t *testing.T) {
	type testCase struct {
		name           string
		input          string
		initialIsBold  bool
		expectedOutput string
		expectedIsBold bool
	}

	tests := []testCase{
		{
			name:           "Empty string",
			input:          "",
			initialIsBold:  false,
			expectedOutput: "",
			expectedIsBold: false,
		},
		{
			name:           "No bolding markers",
			input:          "This is plain text without any bolding.",
			initialIsBold:  false,
			expectedOutput: "This is plain text without any bolding.",
			expectedIsBold: false,
		},
		{
			name:           "Simple bolding pair",
			input:          "Hello **world**!",
			initialIsBold:  false,
			expectedOutput: "Hello \033[1mworld\033[22m!",
			expectedIsBold: false,
		},
		{
			name:           "Multiple bolding pairs in one string",
			input:          "**First** part and **second** part.",
			initialIsBold:  false,
			expectedOutput: "\033[1mFirst\033[22m part and \033[1msecond\033[22m part.",
			expectedIsBold: false,
		},
		{
			name:           "Unclosed bold at the end",
			input:          "This is **unclosed bold",
			initialIsBold:  false,
			expectedOutput: "This is \033[1munclosed bold",
			expectedIsBold: true, // Parser state should be bold
		},
		{
			name:           "Unclosed bold at the beginning",
			input:          "**Unclosed bold at start",
			initialIsBold:  false,
			expectedOutput: "\033[1mUnclosed bold at start",
			expectedIsBold: true, // Parser state should be bold
		},
		{
			name:           "String starting and ending with bold markers",
			input:          "**Full string bold**",
			initialIsBold:  false,
			expectedOutput: "\033[1mFull string bold\033[22m",
			expectedIsBold: false,
		},
		{
			name:           "Only bolding markers",
			input:          "**",
			initialIsBold:  false,
			expectedOutput: "\033[1m",
			expectedIsBold: true, // Parser state should be bold
		},
		{
			name:           "Two consecutive bolding markers (empty bold segment)",
			input:          "TextA****TextB",
			initialIsBold:  false,
			expectedOutput: "TextA\033[1m\033[22mTextB",
			expectedIsBold: false,
		},
		{
			name:           "Multiple consecutive bolding markers (four asterisks)",
			input:          "****",
			initialIsBold:  false,
			expectedOutput: "\033[1m\033[22m",
			expectedIsBold: false,
		},
		{
			name:           "Three consecutive bolding markers (odd number of asterisks)",
			input:          "***text", // should treat first two as open, then *text
			initialIsBold:  false,
			expectedOutput: "\033[1m*text", // The third * is not part of a  pair
			expectedIsBold: true,
		},
		{
			name:           "Five consecutive bolding markers",
			input:          "*****text", // should treat first two as open, next two as close, then *text
			initialIsBold:  false,
			expectedOutput: "\033[1m\033[22m*text",
			expectedIsBold: false,
		},
		{
			name:           "Text with single asterisks (not bold)",
			input:          "This *is* not bold *formatting*.",
			initialIsBold:  false,
			expectedOutput: "This *is* not bold *formatting*.",
			expectedIsBold: false,
		},
		{
			name:           "Parser starts in bold state, then closes",
			input:          "This text **should close bold",
			initialIsBold:  true,
			expectedOutput: "This text \033[22mshould close bold",
			expectedIsBold: false,
		},
		{
			name:           "Parser starts in bold state, then opens and closes again",
			input:          "Initial **then new bold** segment",
			initialIsBold:  true,
			expectedOutput: "Initial \033[22mthen new bold\033[1m segment", // Initial `` closes, next `` opens
			expectedIsBold: true,                                           // Remains bold because the last  opens it.
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := NewParser()
			p.isBold = tt.initialIsBold
			got := p.Parse(tt.input)
			if got != tt.expectedOutput {
				t.Errorf("Parse() for input %q\n got = %q\nwant = %q", tt.input, got, tt.expectedOutput)
			}
			if p.isBold != tt.expectedIsBold {
				t.Errorf("Parse() for input %q: final isBold state mismatch; got %v, want %v", tt.input, p.isBold, tt.expectedIsBold)
			}
		})
	}

	t.Run("Consecutive calls maintain state", func(t *testing.T) {
		p := NewParser() // isBold = false initially

		// Call 1: Unclosed bold
		input1 := "Part 1: This is **bolding"
		expected1 := "Part 1: This is \033[1mbolding"
		p.isBold = false // Ensure initial state for this sub-test
		got1 := p.Parse(input1)
		if got1 != expected1 {
			t.Errorf("Call 1 output mismatch:\n got = %q\nwant = %q", got1, expected1)
		}
		if p.isBold != true {
			t.Errorf("Call 1 final isBold state mismatch; got %v, want %v", p.isBold, true)
		}

		// Call 2: Continue the bold, then close it
		input2 := " Part 2: more text, **then normal"
		expected2 := " Part 2: more text, \033[22mthen normal" // isBold was true, so  closes it
		// p.isBold is now true from previous call, no need to set again
		got2 := p.Parse(input2)
		if got2 != expected2 {
			t.Errorf("Call 2 output mismatch:\n got = %q\nwant = %q", got2, expected2)
		}
		if p.isBold != false {
			t.Errorf("Call 2 final isBold state mismatch; got %v, want %v", p.isBold, false)
		}

		// Call 3: Start new bold segment
		input3 := " Part 3: start **new bold** again"
		expected3 := " Part 3: start \033[1mnew bold\033[22m again"
		// p.isBold is now false from previous call
		got3 := p.Parse(input3)
		if got3 != expected3 {
			t.Errorf("Call 3 output mismatch:\n got = %q\nwant = %q", got3, expected3)
		}
		if p.isBold != false {
			t.Errorf("Call 3 final isBold state mismatch; got %v, want %v", p.isBold, false)
		}

		// Call 4: Just plain text, ensure state doesn't change
		input4 := " Part 4: plain text."
		expected4 := " Part 4: plain text."
		got4 := p.Parse(input4)
		if got4 != expected4 {
			t.Errorf("Call 4 output mismatch:\n got = %q\nwant = %q", got4, expected4)
		}
		if p.isBold != false {
			t.Errorf("Call 4 final isBold state mismatch; got %v, want %v", p.isBold, false)
		}
	})
}
