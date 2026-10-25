package toolbox

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/jdevoo/gen/core"
	_ "github.com/lib/pq"
	"google.golang.org/genai"
	_ "modernc.org/sqlite"
)

type ExploreRelationalDataArgs struct {
	SQL string `json:"sql"`
}

// ExploreRelationalData executes any SQL query against a PostgreSQL or SQLite database.
func (t Tool) ExploreRelationalData(ctx context.Context, args ExploreRelationalDataArgs) (*genai.Part, error) {
	keyVals, ok := ctx.Value(core.KeyValsKey).(core.ParamMap)
	if !ok {
		return nil, fmt.Errorf("ExploreRelationalData: keyVals not found in context")
	}

	dsn, ok := keyVals["DSN"]
	if !ok {
		return nil, &core.ParamError{
			Message: "ExploreRelationalData: missing parameter\n  -p DSN=postgres://pqgo:password@localhost\n  or -p DSN=sqlite:/path/file.db",
		}
	}

	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("ExploreRelationalData: params not found in context")
	}

	resMap, err := executeSQLQuery(ctx, dsn, args.SQL, params.JSON)
	if err != nil {
		return genai.NewPartFromFunctionResponse(
			"ExploreRelationalData",
			map[string]any{
				"output": "ERROR",
				"error":  err.Error(),
			},
		), nil
	}

	resMap["output"] = "SUCCESS"
	return genai.NewPartFromFunctionResponse("ExploreRelationalData", resMap), nil
}

// parseDSN helps determine the right driver and connection string.
func parseDSN(dsn string) (driver string, connStr string, err error) {
	trimmed := strings.TrimSpace(dsn)
	switch {
	case strings.HasPrefix(trimmed, "sqlite:"):
		rawPath := strings.TrimPrefix(trimmed, "sqlite:")
		rawPath = strings.TrimPrefix(rawPath, "//")
		return "sqlite", rawPath, nil
	case strings.HasPrefix(trimmed, "postgres://"),
		strings.HasPrefix(trimmed, "postgresql://"):
		return "postgres", trimmed, nil
	default:
		if strings.HasSuffix(trimmed, ".db") || strings.HasSuffix(trimmed, ".sqlite") {
			return "sqlite", trimmed, nil
		}
		return "postgres", trimmed, nil
	}
}

func executeSQLQuery(ctx context.Context, dsn string, query string, wantJSON bool) (map[string]any, error) {
	driver, connStr, err := parseDSN(dsn)
	if err != nil {
		return nil, fmt.Errorf("parsing DSN: %v", err)
	}

	db, err := sql.Open(driver, connStr)
	if err != nil {
		return nil, fmt.Errorf("opening database connection: %v", err)
	}
	defer db.Close()

	// check if valid statement
	trimmed := strings.ToLower(strings.TrimSpace(query))
	isSelect := strings.HasPrefix(trimmed, "select") ||
		strings.HasPrefix(trimmed, "with") ||
		strings.HasPrefix(trimmed, "show") ||
		strings.HasPrefix(trimmed, "explain") ||
		strings.HasPrefix(trimmed, "pragma")

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
