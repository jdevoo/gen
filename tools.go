package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"google.golang.org/genai"
)

type Tool struct{}

// ListKnownGeminiModels retrieves the list of available Gemini models
func (t Tool) ListKnownGeminiModels() (string, error) {
	var res string
	ctx := context.Background()
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		genLogFatal(err)
	}
	for m, err := range client.Models.All(ctx) {
		if err != nil {
			return "", err
		}
		res += fmt.Sprintf("%s %s\n", m.Name, m.Description)
	}
	return res, nil
}

// ListAWSServices returns a list of services via Steampipe
func (t Tool) ListAWSServices() (string, error) {
	return queryPostgres("SELECT DISTINCT foreign_table_name FROM information_schema.foreign_tables WHERE foreign_table_schema='aws'")
}

// CountWords returns the count of words in `s`
func (t Tool) CountWords(s string) (string, error) {
	words := strings.Fields(s)
	count := len(words)
	res := strconv.Itoa(count)
	return res, nil
}

// GetWorkingDirectory returns the current directory
func (t Tool) GetWorkingDirectory() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	return wd, nil
}
