package toolbox

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

type ExploreDataSetArgs struct {
	SQL string `json:"sql"`
}

// ExecuteSQL executes any SQL query against a PostgreSQL database.
func (t Tool) ExploreDataSet(ctx context.Context, args ExploreDataSetArgs) (*genai.Part, error) {
	keyVals, ok := ctx.Value(core.KeyValsKey).(core.ParamMap)
	if !ok {
		return nil, fmt.Errorf("ExploreDataSet: keyVals not found in context")
	}

	dsn, ok := keyVals["DSN"]
	if !ok {
		return nil, &core.ParamError{
			Message: "ExploreDataSet: missing parameter\n  -p DSN=postgres://pqgo:password@localhost",
		}
	}

	res, err := executePostgresQuery(ctx, dsn, args.SQL)
	if err != nil {
		return genai.NewPartFromFunctionResponse(
			"ExecuteSQL",
			map[string]any{
				"output": "ERROR",
				"error":  err.Error(),
			},
		), nil
	}

	return genai.NewPartFromFunctionResponse(
		"ExecuteSQL",
		map[string]any{
			"output": "SUCCESS",
			"text":   res,
		},
	), nil
}

func executePostgresQuery(ctx context.Context, dsn string, query string) (string, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return "", fmt.Errorf("opening database connection: %v", err)
	}
	defer db.Close()

	// check if valid statement
	trimmed := strings.ToLower(strings.TrimSpace(query))
	isSelect := strings.HasPrefix(trimmed, "select") ||
		strings.HasPrefix(trimmed, "with") ||
		strings.HasPrefix(trimmed, "show") ||
		strings.HasPrefix(trimmed, "explain")

	if isSelect {
		rows, err := db.QueryContext(ctx, query)
		if err != nil {
			return "", err
		}
		defer rows.Close()
		cols, err := rows.Columns()
		if err != nil {
			return "", err
		}
		var res []string
		res = append(res, strings.Join(cols, " | ")) // header row

		row := make([]any, len(cols))
		rowPtr := make([]any, len(cols))
		for i := range row {
			rowPtr[i] = &row[i]
		}
		for rows.Next() {
			err := rows.Scan(rowPtr...)
			if err != nil {
				return "", err
			}
			var rowStr []string
			for _, val := range row {
				if val == nil {
					rowStr = append(rowStr, "NULL")
				} else {
					switch v := val.(type) {
					case []byte:
						rowStr = append(rowStr, string(v))
					default:
						rowStr = append(rowStr, fmt.Sprintf("%v", v))
					}
				}
			}
			res = append(res, strings.Join(rowStr, " | "))
		}

		if err := rows.Err(); err != nil {
			return "", err
		}

		return strings.Join(res, "\n"), nil
	} else {
		result, err := db.ExecContext(ctx, query)
		if err != nil {
			return "", err
		}
		rowsAffected, err := result.RowsAffected()
		if err != nil {
			// commands like CREATE TABLE do not support RowsAffected
			return "Command executed successfully.", nil
		}
		return fmt.Sprintf("Command executed successfully. Rows affected: %d", rowsAffected), nil
	}
}
