package main

import (
	"strings"
	"testing"
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

// TestReadLine tests user input in chat mode
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
			t.Errorf("readLine(%q) error = %v, wantErr %v", test.input, err, test.wantErr)
			continue
		}
		if res != test.want {
			t.Errorf("readLine(%q) = %q, want %q", test.input, res, test.want)
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
