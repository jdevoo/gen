package main

import (
	"strings"
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
		{
			name:     "intra-word asterisk redaction like f**king",
			input:    "78. ‘I’m a f**king US citizen’: Man left injured",
			expected: ansiWhite + "78. ‘I’m a f**king US citizen’: Man left injured" + ansiReset,
		},
		{
			name:     "word-ending asterisk redaction like f**?",
			input:    "What the f**?",
			expected: ansiWhite + "What the f**?" + ansiReset,
		},
		{
			name:     "intra-word redaction inside valid bold segment",
			input:    "**He shouted ‘I’m a f**king citizen!’ and ran.**",
			expected: ansiWhite + ansiBold + "He shouted ‘I’m a f**king citizen!’ and ran." + ansiNoBold + ansiReset,
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
	t.Run("bold persistence across calls", func(t *testing.T) {
		parser := &MarkdownParser{}

		// segment starts the bold tag but does not close it
		res1 := parser.parse("This is **bold")
		expected1 := ansiWhite + "This is " + ansiBold + "bold" + ansiReset
		if res1 != expected1 {
			t.Errorf("First Parse failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}

		// segment continues the bold state and then closes it
		res2 := parser.parse(" text continuing** and normal")
		expected2 := ansiWhite + ansiBold + " text continuing" + ansiNoBold + " and normal" + ansiReset
		if res2 != expected2 {
			t.Errorf("Second Parse failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
	})

	t.Run("single trailing asterisk buffer", func(t *testing.T) {
		parser := &MarkdownParser{}

		// segment ends with a single '*' should be buffered and not rendered yet
		res1 := parser.parse("This is a *")
		expected1 := ansiWhite + "This is a " + ansiReset
		if res1 != expected1 {
			t.Errorf("Buffering Parse failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}
		if parser.buffer != "*" {
			t.Errorf("Expected buffer to contain '*' but got %q", parser.buffer)
		}

		// segment begins with another '*' completing the bold sequence
		res2 := parser.parse("*bold text**")
		expected2 := ansiWhite + ansiBold + "bold text" + ansiNoBold + ansiReset
		if res2 != expected2 {
			t.Errorf("Reconstructed Parse failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
		if parser.buffer != "" {
			t.Errorf("Expected buffer to be cleared but got %q", parser.buffer)
		}
	})

	t.Run("intra-word split across chunks (f* + *king)", func(t *testing.T) {
		parser := &MarkdownParser{}
		res1 := parser.parse("I'm a f*")
		expected1 := ansiWhite + "I'm a f" + ansiReset
		if res1 != expected1 {
			t.Errorf("Chunk 1 failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}
		res2 := parser.parse("*king citizen")
		expected2 := ansiWhite + "**king citizen" + ansiReset
		if res2 != expected2 {
			t.Errorf("Chunk 2 failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
	})

	t.Run("intra-word split across chunks (f** + king)", func(t *testing.T) {
		parser := &MarkdownParser{}
		res1 := parser.parse("I'm a f**")
		expected1 := ansiWhite + "I'm a f**" + ansiReset
		if res1 != expected1 {
			t.Errorf("Chunk 1 failed:\nExpected: %q\nActual:   %q", expected1, res1)
		}
		res2 := parser.parse("king citizen")
		expected2 := ansiWhite + "king citizen" + ansiReset
		if res2 != expected2 {
			t.Errorf("Chunk 2 failed:\nExpected: %q\nActual:   %q", expected2, res2)
		}
	})
}

func TestParse_MarkdownImages(t *testing.T) {
	t.Run("swallow image and signal imageFound", func(t *testing.T) {
		parser := &MarkdownParser{}

		input := "Here is the plot:\n![Comparison](comparison.png)\nDone."
		chunk1 := parser.parse(input)

		if !parser.imageFound {
			t.Errorf("Expected imageFound to be true")
		}
		expected1 := ansiWhite + "Here is the plot:\n" + ansiReset
		if chunk1 != expected1 {
			t.Errorf("\nExpected: %q\nActual:   %q", expected1, chunk1)
		}

		chunk2 := parser.parse("")
		expected2 := ansiWhite + "Done." + ansiReset
		if chunk2 != expected2 {
			t.Errorf("\nExpected: %q\nActual:   %q", expected2, chunk2)
		}
	})

	t.Run("handle split between ! and [ across stream chunks", func(t *testing.T) {
		parser := &MarkdownParser{}

		chunk1 := "!"
		_ = parser.parse(chunk1)

		if parser.buffer != "!" {
			t.Fatalf("Expected '!' to be buffered, got %q", parser.buffer)
		}

		// rest of the image tag
		chunk2 := "[Distribution Comparison](comparison.png)\n"
		_ = parser.parse(chunk2)

		if !parser.imageFound {
			t.Errorf("Expected imageFound to be true across chunk boundary")
		}
		if parser.buffer != "" {
			t.Errorf("Expected buffer to be cleared, got %q", parser.buffer)
		}
	})

	t.Run("preserve exclamation mark in sentences without buffering", func(t *testing.T) {
		parser := &MarkdownParser{}
		actual := parser.parse("Great job!")
		expected := ansiWhite + "Great job!" + ansiReset
		if actual != expected {
			t.Errorf("\nExpected: %q\nActual:   %q", expected, actual)
		}
		if parser.buffer != "" {
			t.Errorf("Expected buffer to be empty, got %q", parser.buffer)
		}
	})

	t.Run("preserve image syntax inside code blocks", func(t *testing.T) {
		parser := &MarkdownParser{}

		input := "```markdown\n![alt](test.png)\n```"
		actual := parser.parse(input)

		if parser.imageFound {
			t.Errorf("imageFound should NOT be true inside code block")
		}
		if !strings.Contains(actual, "![alt](test.png)") {
			t.Errorf("Image syntax inside code block should be preserved, got: %q", actual)
		}
	})
}

// TestParse_Concurrency tests parser safety under concurrent calls.
func TestParse_Concurrency(t *testing.T) {
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
	if actual := infos("info"); actual != colorCyan+"info"+ansiReset {
		t.Errorf("infos() = %q, expected %q", actual, colorCyan+"info"+ansiReset)
	}
	if actual := important("important"); actual != colorRed+"important"+ansiReset {
		t.Errorf("important() = %q, expected %q", actual, colorRed+"important"+ansiReset)
	}
	if actual := roles("role"); actual != colorRole+"role"+ansiReset {
		t.Errorf("roles() = %q, expected %q", actual, colorRole+"role"+ansiReset)
	}
}
