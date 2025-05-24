package main

import (
	"strings"
	"testing"

	"google.golang.org/genai"
)

// TestParamMapSet tests setting prompt parameter.
func TestParamMapSet(t *testing.T) {
	tests := []struct {
		arg     string
		want    ParamMap
		wantErr bool
	}{
		{
			arg:     "name=John",
			want:    ParamMap{"name": "John"},
			wantErr: false,
		},
		{
			arg:     "invalid",
			want:    ParamMap{},
			wantErr: true,
		},
		{
			arg:     "missing equal",
			want:    ParamMap{},
			wantErr: true,
		},
		{
			arg:     "name==",
			want:    ParamMap{"name": "="},
			wantErr: false,
		},
		{
			arg:     "blank=",
			want:    ParamMap{"blank": ""},
			wantErr: false,
		},
	}

	for _, test := range tests {
		res := ParamMap{}
		err := res.Set(test.arg)
		if (err != nil) != test.wantErr {
			t.Errorf("Set(%q) error = %v, wantErr %v", test.arg, err, test.wantErr)
			continue
		}
		if test.want["name"] != res["name"] {
			t.Errorf("Expected 'name' to be '%s', got '%s'", test.want["name"], res["name"])
		}
	}
}

// TestReadLine tests user input in chat mode.
func TestReadLine(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{
			input:   "Hello, world!\n",
			want:    "Hello, world!",
			wantErr: false,
		},
		{
			input:   "\n",
			want:    "",
			wantErr: false,
		},
	}

	for _, test := range tests {
		reader := strings.NewReader(test.input)
		res, err := readLine(reader)
		if (err != nil) != test.wantErr {
			t.Errorf("Did not expect error for '%s'", test.input)
			continue
		}
		if res != test.want {
			t.Errorf("Expected '%s', got '%s' for '%s'", test.want, res, test.input)
		}
	}
}

// TestSearchReplace tests the searchReplace function.
func TestSearchReplace(t *testing.T) {
	tests := []struct {
		prompt string
		params ParamMap
		want   string
	}{
		{
			prompt: "Hello {name}, how are you?",
			params: ParamMap{"name": "World"},
			want:   "Hello World, how are you?",
		},
		{
			prompt: "This is a {adjective} {noun}.",
			params: ParamMap{"adjective": "beautiful", "noun": "day"},
			want:   "This is a beautiful day.",
		},
		{
			prompt: "This is a test string.",
			params: ParamMap{},
			want:   "This is a test string.",
		},
		{
			prompt: "This is a {empty} test string.",
			params: ParamMap{"empty": ""},
			want:   "This is a  test string.",
		},
	}

	for _, test := range tests {
		result := searchReplace(test.prompt, test.params)
		if result != test.want {
			t.Errorf("Expected '%s', got '%s'", test.want, result)
		}
	}
}

// TestAnyMatches tests list inclusion.
func TestAnyMatches(t *testing.T) {
	tests := []struct {
		inputArray []string
		inputCand  []string
		wantRes    bool
	}{
		{
			inputArray: []string{},
			inputCand:  []string{".prompt"},
			wantRes:    false,
		},
		{
			inputArray: []string{"image.png", "my.prompt"},
			inputCand:  []string{".prompt"},
			wantRes:    true,
		},
		{
			inputArray: []string{"my.sprompt"},
			inputCand:  []string{".prompt"},
			wantRes:    false,
		},
	}

	for _, test := range tests {
		res := anyMatches(test.inputArray, test.inputCand...)
		if test.wantRes != res {
			t.Errorf("Expected %t, got %t", test.wantRes, res)
		}
	}
}

// TestAnyMatch tests list inclusion.
func TestAllMatch(t *testing.T) {
	tests := []struct {
		inputArray []string
		inputCand  string
		wantRes    bool
	}{
		{
			inputArray: []string{},
			inputCand:  ".prompt",
			wantRes:    false,
		},
		{
			inputArray: []string{"image.png", "my.prompt"},
			inputCand:  ".prompt",
			wantRes:    false,
		},
		{
			inputArray: []string{"my.sprompt"},
			inputCand:  ".sprompt",
			wantRes:    true,
		},
	}

	for _, test := range tests {
		res := allMatch(test.inputArray, test.inputCand)
		if test.wantRes != res {
			t.Errorf("Expected %t, got %t for %v", test.wantRes, res, test.inputArray)
		}
	}
}

// TestOneMatches tests list inclusion.
func TestOneMatches(t *testing.T) {
	tests := []struct {
		inputArray []string
		inputCand  string
		wantRes    bool
	}{
		{
			inputArray: []string{},
			inputCand:  "-",
			wantRes:    false,
		},
		{
			inputArray: []string{"-", "my.prompt"},
			inputCand:  ".prompt",
			wantRes:    true,
		},
		{
			inputArray: []string{"my.sprompt"},
			inputCand:  "-",
			wantRes:    false,
		},
	}

	for _, test := range tests {
		res := oneMatches(test.inputArray, test.inputCand)
		if test.wantRes != res {
			t.Errorf("Expected %t, got %t for %v", test.wantRes, res, test.inputArray)
		}
	}
}

// TestPartHasKey tests parts with {digest}.
func TestPartHasKey(t *testing.T) {
	tests := []struct {
		inputParts []*genai.Part
		inputKey   string
		wantRes    int
	}{
		{
			inputParts: []*genai.Part{
				{Text: "some prompt without key"},
				{Text: "some other prompt without key"},
			},
			inputKey: "{digest}",
			wantRes:  -1,
		},
		{
			inputParts: []*genai.Part{
				{Text: "some prompt with {digest}"},
			},
			inputKey: "{digest}",
			wantRes:  0,
		},
		{
			inputParts: []*genai.Part{
				{Text: "some prompt without key}"},
				{Text: "some prompt with {digest}"},
			},
			inputKey: "{digest}",
			wantRes:  1,
		},
	}

	for _, test := range tests {
		res := partWithKey(test.inputParts, test.inputKey)
		if test.wantRes != res {
			t.Errorf("Expected %d, got %d for %v", test.wantRes, res, test.inputParts)
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
		wantRes    []*genai.Part
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
			wantRes: []*genai.Part{
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
			wantRes: []*genai.Part{
				{Text: "other prompt without key"},
				{Text: "yet another prompt without key"},
				{Text: "prompt with key in last position bla"},
			},
		},
	}

	for _, test := range tests {
		replacePart(&test.inputParts, test.inputIdx, test.inputKey, test.inputVal)
		for idx := range test.inputParts {
			if test.inputParts[idx].Text != test.wantRes[idx].Text {
				t.Errorf("Expected '%s', got '%s'", test.wantRes[idx].Text, test.inputParts[idx].Text)
				break
			}
		}
	}
}
