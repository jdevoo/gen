package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestSpinner(t *testing.T) {
	var buf bytes.Buffer

	s := NewSpinner("Test Spinner %s",
		WithFrames("123"),
		WithWriter(&buf),
	)

	s.Set("ABC")
	if s.next() != "A" {
		t.Errorf("Expected first next() to be 'A'")
	}
	if s.next() != "B" {
		t.Errorf("Expected second next() to be 'B'")
	}
	if s.next() != "C" {
		t.Errorf("Expected third next() to be 'C'")
	}
	if s.next() != "A" {
		t.Errorf("Expected wrap next() to be 'A'")
	}

	// set s.tpf to be extremely fast for test speed
	s.tpf = 1 * time.Millisecond

	// start the spinner
	s.Start()

	// double start should be a no-op
	s.Start()

	// let it tick a few times
	time.Sleep(10 * time.Millisecond)

	stopped := s.Stop()
	if !stopped {
		t.Errorf("Expected Stop() to return true")
	}

	// stop it again (should return false since already inactive)
	if s.Stop() {
		t.Errorf("Expected second Stop() to return false")
	}

	output := buf.String()
	if !strings.Contains(output, "Test Spinner") {
		t.Errorf("Expected spinner output to contain 'Test Spinner', got %q", output)
	}
	if !strings.Contains(output, HideCursor) {
		t.Errorf("Expected spinner output to contain HideCursor sequence")
	}
	if !strings.Contains(output, ShowCursor) {
		t.Errorf("Expected spinner output to contain ShowCursor sequence")
	}
}
