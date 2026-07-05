package main

import (
	"bytes"
	"context"
	"flag"
	"testing"
)

// TestValidateEnv tests the credentials environment variables validation.
func TestValidateEnv(t *testing.T) {
	clearEnv := func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_PROJECT", "")
		t.Setenv("GOOGLE_API_KEY", "")
		t.Setenv("GOOGLE_CLOUD_LOCATION", "")
		t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "")
	}

	tests := []struct {
		name    string
		setup   func(t *testing.T)
		wantErr bool
	}{
		{
			name: "Neither set",
			setup: func(t *testing.T) {
				clearEnv(t)
			},
			wantErr: true,
		},
		{
			name: "Only API key set",
			setup: func(t *testing.T) {
				clearEnv(t)
				t.Setenv("GOOGLE_API_KEY", "secret-key")
			},
			wantErr: false,
		},
		{
			name: "Cloud Project set but Location missing",
			setup: func(t *testing.T) {
				clearEnv(t)
				t.Setenv("GOOGLE_CLOUD_PROJECT", "my-project")
			},
			wantErr: true,
		},
		{
			name: "Cloud Project and Location set",
			setup: func(t *testing.T) {
				clearEnv(t)
				t.Setenv("GOOGLE_CLOUD_PROJECT", "my-project")
				t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
			},
			wantErr: false,
		},
		{
			name: "Both key and project set, but vertexai missing",
			setup: func(t *testing.T) {
				clearEnv(t)
				t.Setenv("GOOGLE_API_KEY", "secret-key")
				t.Setenv("GOOGLE_CLOUD_PROJECT", "my-project")
				t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
			},
			wantErr: true,
		},
		{
			name: "Both key and project set, vertexai true",
			setup: func(t *testing.T) {
				clearEnv(t)
				t.Setenv("GOOGLE_API_KEY", "secret-key")
				t.Setenv("GOOGLE_CLOUD_PROJECT", "my-project")
				t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
				t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
			},
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tc.setup(t)
			err := validateEnv()
			if (err != nil) != tc.wantErr {
				t.Errorf("validateEnv() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

// TestEmitUsage tests the customized CLI usage instructions output.
func TestEmitUsage(t *testing.T) {
	var buf bytes.Buffer
	ctx := context.WithValue(context.Background(), paramsKey, &Parameters{})

	oldOutput := flag.CommandLine.Output()
	flag.CommandLine.SetOutput(&buf)
	defer flag.CommandLine.SetOutput(oldOutput)

	emitUsage(ctx, &buf, false)
	output := buf.String()

	if !bytes.Contains(buf.Bytes(), []byte("Usage: gen [options] <prompt>")) {
		t.Errorf("emitUsage output did not contain Usage text, got %q", output)
	}
}
