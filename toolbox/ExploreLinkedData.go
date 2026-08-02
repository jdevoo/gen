package toolbox

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/jdevoo/gen/core"
	"github.com/jdevoo/gen/toolbox/sparql"
	"google.golang.org/genai"
)

type ExploreLinkedDataArgs struct {
	SPARQL string `json:"sparql"`
}

// ExploreLinkedData executes any SPARQL query against a SPARQL endpoint.
func (t Tool) ExploreLinkedData(ctx context.Context, args ExploreLinkedDataArgs) (*genai.Part, error) {
	client, err := getSPARQLClient(ctx, "ExploreLinkedData")
	if err != nil {
		return nil, err
	}

	qr, err := client.Query(ctx, args.SPARQL)
	if err != nil {
		return genai.NewPartFromFunctionResponse(
			"ExploreLinkedData",
			map[string]any{
				"output": "ERROR",
				"error":  err.Error(),
			},
		), nil
	}
	defer qr.Close()

	vars := qr.Variables()

	// handle ASK Query
	if len(vars) == 0 {
		isTrue, err := qr.Boolean()
		if err != nil {
			return genai.NewPartFromFunctionResponse(
				"ExploreLinkedData",
				map[string]any{
					"output": "ERROR",
					"error":  err.Error(),
				},
			), nil
		}
		return genai.NewPartFromFunctionResponse(
			"ExploreLinkedData",
			map[string]any{
				"output": "SUCCESS",
				"text":   fmt.Sprintf("Result: %t", isTrue),
			},
		), nil
	}

	// handle SELECT or CONSTRUCT/DESCRIBE query (Table format)
	var lines []string
	lines = append(lines, strings.Join(vars, " | ")) // header

	for {
		row, err := qr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return genai.NewPartFromFunctionResponse(
				"ExploreLinkedData",
				map[string]any{
					"output": "ERROR",
					"error":  err.Error(),
				},
			), nil
		}

		var rowVals []string
		for _, v := range vars {
			if val, ok := row[v]; ok {
				rowVals = append(rowVals, val.String())
			} else {
				rowVals = append(rowVals, "NULL")
			}
		}
		lines = append(lines, strings.Join(rowVals, " | "))
	}

	return genai.NewPartFromFunctionResponse(
		"ExploreLinkedData",
		map[string]any{
			"output": "SUCCESS",
			"text":   strings.Join(lines, "\n"),
		},
	), nil
}

// Helper to instantiate the SPARQL client using configurations from the context
func getSPARQLClient(ctx context.Context, toolName string) (*sparql.Client, error) {
	keyVals, ok := ctx.Value(core.KeyValsKey).(core.ParamMap)
	if !ok {
		return nil, fmt.Errorf("%s: keyVals not found in context", toolName)
	}

	endpoint, ok := keyVals["SPARQL_ENDPOINT"]
	if !ok {
		return nil, &core.ParamError{
			Message: fmt.Sprintf("%s: missing parameter\n  -p SPARQL_ENDPOINT=http://localhost:8890/sparql", toolName),
		}
	}

	return sparql.New(endpoint)
}
