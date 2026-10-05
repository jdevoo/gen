package main

import (
	"testing"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

func TestSerializeDeserialize(t *testing.T) {
	testCases := []struct {
		doc      Document
		expected Document
	}{
		{
			doc: Document{
				embedding: []float32{1.0, 2.0, 3.0},
				content:   "This is a test document.",
				metadata:  map[string]string{"key1": "value1"},
			},
			expected: Document{
				embedding: []float32{1.0, 2.0, 3.0},
				content:   "This is a test document.",
				metadata:  map[string]string{"key1": "value1"},
			},
		},
		{
			doc: Document{
				embedding: []float32{1.0, 2.0, 3.0},
				content:   "This is a test document.",
				metadata:  map[string]string{},
			},
			expected: Document{
				embedding: []float32{1.0, 2.0, 3.0},
				content:   "This is a test document.",
				metadata:  map[string]string{},
			},
		},
	}

	for _, tc := range testCases {
		serialized, err := serializeDoc(tc.doc)
		if err != nil {
			t.Fatalf("serializeDoc failed: %v", err)
		}
		deserialized, err := deserializeDoc(serialized)
		if err != nil {
			t.Fatalf("deserializeDoc failed: %v", err)
		}
		if !float32SlicesEqual(deserialized.embedding, tc.expected.embedding) {
			t.Errorf("Embedding mismatch: got %v, want %v", deserialized.embedding, tc.expected.embedding)
		}
		if deserialized.content != tc.expected.content {
			t.Errorf("Content mismatch: got %q, want %q", deserialized.content, tc.expected.content)
		}
		if !mapsEqual(deserialized.metadata, tc.expected.metadata) {
			t.Errorf("Metadata mismatch: got %v, want %v", deserialized.metadata, tc.expected.metadata)
		}
	}
}

func float32SlicesEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	expected := float32(32)
	got := dotProduct(a, b)
	if got != expected {
		t.Errorf("Dot product failed, got %f, expected %f", got, expected)
	}
}

func TestAppendAndQueryDigest(t *testing.T) {
	tmpDir := t.TempDir()

	embeddings := []*genai.ContentEmbedding{
		{Values: []float32{0.9, 0.1, 0.0}},
		{Values: []float32{0.0, 0.8, 0.2}},
		{Values: []float32{0.1, 0.2, 0.9}},
	}
	chunks := []string{
		"Introduction to neural networks.",
		"Relational database query optimization.",
		"Quantum computing and entanglement.",
	}
	keyVals := core.ParamMap{"source": "test-doc"}

	err := appendToDigest(tmpDir, embeddings, chunks, keyVals, false, false)
	if err != nil {
		t.Fatalf("appendToDigest failed: %v", err)
	}

	queryEmb := &genai.ContentEmbedding{
		Values: []float32{0.95, 0.05, 0.0}, // high similarity to first doc
	}

	results, err := queryDigest(tmpDir, queryEmb, nil, 2, 0.5, false)
	if err != nil {
		t.Fatalf("queryDigest failed: %v", err)
	}

	if len(results) != 2 {
		t.Fatalf("Expected 2 results, got %d", len(results))
	}

	if results[0].doc.content != chunks[0] {
		t.Errorf("Expected content %q, got %q", chunks[0], results[0].doc.content)
	}

	if results[0].doc.metadata["source"] != "test-doc" {
		t.Errorf("Expected metadata source='test-doc', got %q", results[0].doc.metadata["source"])
	}
}
