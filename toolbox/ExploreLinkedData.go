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

// getSPARQLClient instantiates a SPARQL client using configurations from the context.
func getSPARQLClient(ctx context.Context, toolName string, endpointKeys []string) (*sparql.Client, error) {
	keyVals, ok := ctx.Value(core.KeyValsKey).(core.ParamMap)
	if !ok {
		return nil, fmt.Errorf("%s: keyVals not found in context", toolName)
	}

	var endpoint string
	var found bool
	for _, key := range endpointKeys {
		if endpoint, found = keyVals[key]; found {
			break
		}
	}

	if !found {
		return nil, &core.ParamError{
			Message: fmt.Sprintf("%s: missing parameter\n  -p %s=http://localhost:8890/sparql", toolName, endpointKeys[0]),
		}
	}

	return sparql.New(endpoint)
}

type ExploreLinkedDataArgs struct {
	SPARQL string `json:"sparql"`
}

// ExploreLinkedData executes any SPARQL query against a SPARQL endpoint.
func (t Tool) ExploreLinkedData(ctx context.Context, args ExploreLinkedDataArgs) (*genai.Part, error) {
	client, err := getSPARQLClient(ctx, "ExploreLinkedData", []string{"SPARQL_QUERY_ENDPOINT", "SPARQL_QUERY_SERVICE", "SPARQL_ENDPOINT"})
	if err != nil {
		return nil, err
	}

	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("ExploreLinkedData: params not found in context")
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
		if params.JSON {
			return genai.NewPartFromFunctionResponse(
				"ExploreLinkedData",
				map[string]any{
					"output": "SUCCESS",
					"result": isTrue,
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

	if params.JSON {
		var bindings []map[string]string
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
			rowMap := make(map[string]string)
			for _, v := range vars {
				if val, ok := row[v]; ok {
					rowMap[v] = val.String()
				}
			}
			bindings = append(bindings, rowMap)
		}
		return genai.NewPartFromFunctionResponse(
			"ExploreLinkedData",
			map[string]any{
				"output":   "SUCCESS",
				"bindings": bindings,
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

type UpdateLinkedDataArgs struct {
	SPARQL string `json:"sparql"`
}

// UpdateLinkedData executes any SPARQL 1.1 Update operation against a SPARQL endpoint.
func (t Tool) UpdateLinkedData(ctx context.Context, args UpdateLinkedDataArgs) (*genai.Part, error) {
	client, err := getSPARQLClient(ctx, "UpdateLinkedData", []string{"SPARQL_UPDATE_ENDPOINT", "SPARQL_UPDATE_SERVICE", "SPARQL_ENDPOINT"})
	if err != nil {
		return nil, err
	}

	err = client.Update(ctx, args.SPARQL)
	if err != nil {
		return genai.NewPartFromFunctionResponse(
			"UpdateLinkedData",
			map[string]any{
				"output": "ERROR",
				"error":  err.Error(),
			},
		), nil
	}

	return genai.NewPartFromFunctionResponse(
		"UpdateLinkedData",
		map[string]any{
			"output": "SUCCESS",
			"text":   "Update executed successfully.",
		},
	), nil
}
