package main

import (
	"context"
	"fmt"
	"os"

	"github.com/google/generative-ai-go/genai"
	_ "github.com/lib/pq"
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

// DSN e.g. postgres://steampipe:secret@localhost/steampipe
func (t Tool) QueryPostgres(query string) (string, error) {
	return fmt.Sprintf(">>> %s\n", query), nil
	/*
		if val, ok := os.LookupEnv("GENDSN"); !ok || len(val) == 0 {
			return "", fmt.Errorf("QueryPostgres() GENDSN environment variable not set")
		}
		var res []string
		db, err := sql.Open("postgres", os.Getenv("GENDSN"))
		if err != nil {
			return "", err
		}
		defer db.Close()
		rows, err := db.Query(query)
		if err != nil {
			return "", err
		}
		defer rows.Close()
		cols, _ := rows.Columns()
		row := make([]interface{}, len(cols))
		rowPtr := make([]interface{}, len(cols))
		for i := range row {
			rowPtr[i] = &row[i]
		}
		for rows.Next() {
			err := rows.Scan(rowPtr...)
			if err != nil {
				return "", err
			}
			res = append(res, fmt.Sprintf("%v", row))
		}
		if err := rows.Err(); err != nil {
			return "", err
		}
		return strings.Join(res, "\n"), nil
	*/
}
