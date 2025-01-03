package main

import (
	"testing"
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
