package main

import (
	"testing"
)

// TestParse_Basic tests parsing basic markdown blocks to ANSI sequences.
func TestParse_Basic(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "plain text without markdown",
			input:    "Hello, world!",
			expected: ansiWhite + "Hello, world!" + ansiReset,
		},
		{
			name:     "single bold segment",
			input:    "Hello **world**!",
			expected: ansiWhite + "Hello " + ansiBold + "world" + ansiNoBold + "!" + ansiReset,
		},
		{
			name:     "multiple bold segments",
			input:    "**Hello** and **welcome**",
			expected: ansiWhite + ansiBold + "Hello" + ansiNoBold + " and " + ansiBold + "welcome" + ansiNoBold + ansiReset,
		},
		{
			name:     "unmatched bold opening sequence",
			input:    "This is **bold and doesn't close",
			expected: ansiWhite + "This is " + ansiBold + "bold and doesn't close" + ansiReset,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parser := &MarkdownParser{}
			actual := parser.parse(tc.input)
			if actual != tc.expected {
				t.Errorf("\nExpected: %q\nActual:   %q", tc.expected, actual)
			}
		})
	}
}

// TestParse_StatefulAndBuffering tests multi-segment parsing carrying state and buffer across boundaries.
func TestParse_StatefulAndBuffering(t *testing.T) {
	// 1. Test bold state carrying over to subsequent calls
	t.Run("bold persistence across calls", func(t *testing.T) {
		parser := &MarkdownParser{}

		// First segment starts the bold tag but does not close it
		res1 := parser.parse("This is **bold")
		expected1 := ansiWhite + "This is " + ansiBold + "bold" + ansiReset
		if res1 != expected1 {
			t.Errorf("First Parse failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}

		// Second segment continues the bold state and then closes it
		res2 := parser.parse(" text continuing** and normal")
		expected2 := ansiWhite + ansiBold + " text continuing" + ansiNoBold + " and normal" + ansiReset
		if res2 != expected2 {
			t.Errorf("Second Parse failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
	})

	// 2. Test buffering of a single trailing asterisk
	t.Run("single trailing asterisk buffer", func(t *testing.T) {
		parser := &MarkdownParser{}

		// First segment ends with a single '*'. It should be buffered and not rendered yet.
		res1 := parser.parse("This is a *")
		expected1 := ansiWhite + "This is a " + ansiReset
		if res1 != expected1 {
			t.Errorf("Buffering Parse failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}
		if parser.buffer != "*" {
			t.Errorf("Expected buffer to contain '*' but got %q", parser.buffer)
		}

		// Second segment begins with another '*' completing the bold sequence
		res2 := parser.parse("*bold text**")
		expected2 := ansiWhite + ansiBold + "bold text" + ansiNoBold + ansiReset
		if res2 != expected2 {
			t.Errorf("Reconstructed Parse failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
		if parser.buffer != "" {
			t.Errorf("Expected buffer to be cleared but got %q", parser.buffer)
		}
	})
}

// TestParse_Concurrency tests parser safety under concurrent calls.
func TestParse_Concurrency(t *testing.T) {
	// Verifies that calling Parse concurrently does not cause race conditions (using the parser's internal Mutex)
	parser := &MarkdownParser{}
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

// TestParse_SetThought tests color changes when in thought parsing state.
func TestParse_SetThought(t *testing.T) {
	parser := &MarkdownParser{}
	parser.setThought(true)
	actual := parser.parse("Thinking hard")
	expected := ansiYellow + "Thinking hard" + ansiReset
	if actual != expected {
		t.Errorf("Expected: %q, got: %q", expected, actual)
	}
}

// TestFlush tests text formatting and state resetting during buffer flushes.
func TestFlush(t *testing.T) {
	t.Run("flush empty", func(t *testing.T) {
		parser := &MarkdownParser{}
		if actual := parser.flush(false); actual != "" {
			t.Errorf("Expected empty flush, got %q", actual)
		}
	})

	t.Run("flush non-redirected", func(t *testing.T) {
		parser := &MarkdownParser{}
		_ = parser.parse("unfinished *") // sets buffer to "*"
		actual := parser.flush(false)
		expected := ansiWhite + "*" + ansiReset
		if actual != expected {
			t.Errorf("Expected %q, got %q", expected, actual)
		}
	})

	t.Run("flush redirected", func(t *testing.T) {
		parser := &MarkdownParser{}
		_ = parser.parse("unfinished *") // sets buffer to "*"
		actual := parser.flush(true)
		expected := "*"
		if actual != expected {
			t.Errorf("Expected %q, got %q", expected, actual)
		}
	})
}

// TestColorHelpers tests output decorators for terminal formatting.
func TestColorHelpers(t *testing.T) {
	if actual := infos("info"); actual != colorCyan + "info" + ansiReset {
		t.Errorf("infos() = %q, expected %q", actual, colorCyan + "info" + ansiReset)
	}
	if actual := important("important"); actual != colorRed + "important" + ansiReset {
		t.Errorf("important() = %q, expected %q", actual, colorRed + "important" + ansiReset)
	}
	if actual := roles("role"); actual != colorRole + "role" + ansiReset {
		t.Errorf("roles() = %q, expected %q", actual, colorRole + "role" + ansiReset)
	}
}
