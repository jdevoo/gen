package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Tool struct{}

// ListKnownGeminiModels retrieves the list of available generative models
func (t Tool) ListKnownGeminiModels() (string, error) {
	var res string
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		return "", err
	}
	defer client.Close()
	iter := client.ListModels(ctx)
	for {
		m, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return "", err
		}
		res += fmt.Sprintf("%s %s\n", m.Name, m.Description)
	}
	return res, nil
}

// ListAWSServices returns a list of services via steampipe
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
