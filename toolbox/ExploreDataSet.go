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

// ExploreDataSet executes any SQL query against a PostgreSQL database.
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

	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("ExploreDataSet: params not found in context")
	}

	resMap, err := executePostgresQuery(ctx, dsn, args.SQL, params.JSON)
	if err != nil {
		return genai.NewPartFromFunctionResponse(
			"ExploreDataSet",
			map[string]any{
				"output": "ERROR",
				"error":  err.Error(),
			},
		), nil
	}

	resMap["output"] = "SUCCESS"
	return genai.NewPartFromFunctionResponse("ExploreDataSet", resMap), nil
}

func executePostgresQuery(ctx context.Context, dsn string, query string, wantJSON bool) (map[string]any, error) {
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening database connection: %v", err)
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
			return nil, err
		}
		defer rows.Close()
		cols, err := rows.Columns()
		if err != nil {
			return nil, err
		}

		if wantJSON {
			var jsonRows []map[string]any
			for rows.Next() {
				row := make([]any, len(cols))
				rowPtr := make([]any, len(cols))
				for i := range row {
					rowPtr[i] = &row[i]
				}
				if err := rows.Scan(rowPtr...); err != nil {
					return nil, err
				}
				rowMap := make(map[string]any)
				for i, col := range cols {
					val := row[i]
					if b, ok := val.([]byte); ok {
						rowMap[col] = string(b)
					} else {
						rowMap[col] = val
					}
				}
				jsonRows = append(jsonRows, rowMap)
			}
			if err := rows.Err(); err != nil {
				return nil, err
			}
			return map[string]any{"rows": jsonRows}, nil
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
				return nil, err
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
			return nil, err
		}

		return map[string]any{"text": strings.Join(res, "\n")}, nil
	}

	result, err := db.ExecContext(ctx, query)
	if err != nil {
		return nil, err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		// commands like CREATE TABLE do not support RowsAffected
		if wantJSON {
			return map[string]any{"text": "Command executed successfully."}, nil
		}
		return map[string]any{"text": "Command executed successfully."}, nil
	}

	if wantJSON {
		return map[string]any{"rowsAffected": rowsAffected}, nil
	}
	return map[string]any{"text": fmt.Sprintf("Command executed successfully. Rows affected: %d", rowsAffected)}, nil
}
