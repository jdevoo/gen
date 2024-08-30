package main

import (
	"context"
	"fmt"
	"os"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type Tool struct{}

// KnownModels retrieves the list of available generative models
func (t Tool) KnownModels() (string, error) {
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

// RetrieveAWSAccountIDs obtains data from steampipe service
func (t Tool) RetrieveAWSAccountIDs() (string, error) {
	return QueryPostgres("SELECT account_id FROM aws_account")
}
