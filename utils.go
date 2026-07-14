package main

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"sort"
	"strings"

	_ "github.com/lib/pq"
	"google.golang.org/genai"
)

type ParamError struct {
	Message string
}

func (e *ParamError) Error() string {
	return e.Message
}

// ParamMap holds key-value pairs for string replacement.
type ParamMap map[string]string

// String implements the flag.Value interface for ParamMap.
func (*ParamMap) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
func (m *ParamMap) Set(kv string) error {
	parts := strings.SplitN(kv, "=", 2) // limit splits to 2
	if len(parts) != 2 {
		return fmt.Errorf("invalid parameter %s", kv)
	}
	(*m)[parts[0]] = parts[1]
	return nil
}

// ParamArray holds a list of strings e.g. file paths.
type ParamArray []string

// String implements the flag.Value interface for ParamArray.
func (*ParamArray) String() string { return "" }

// Set implements the flag.Value interface for ParamArray.
func (a *ParamArray) Set(val string) error {
	*a = append(*a, val)
	return nil
}

// conjoin returns a single text resulting from concatenation of all original parts.
// TODO handle other part types
func conjTexts(parts *[]*genai.Part) {
	if len(*parts) == 0 {
		return
	}
	var buf bytes.Buffer
	for _, p := range *parts {
		if p.Text != "" {
			buf.WriteString(string(p.Text))
		}
	}
	*parts = []*genai.Part{{Text: buf.String()}}
}

// searchReplace performs string replacement based on key-value pairs.
func searchReplace(prompt string, pm ParamMap) string {
	res := prompt
	for k, v := range pm {
		searchRegex := regexp.MustCompile("(?i){" + regexp.QuoteMeta(k) + "}")
		res = searchRegex.ReplaceAllString(res, v)
	}
	return res
}

// partWithKey searches prompt parts for occurrence of key and returns index.
func partWithKey(parts []*genai.Part, key string) int {
	for idx, part := range parts {
		if strings.Contains(string(part.Text), key) {
			return idx
		}
	}
	return -1
}

// replacePart returns new array with updated entry at idx.
func replacePart(parts *[]*genai.Part, idx int, key string, selection []QueryResult) {
	var keyVal string
	for _, s := range selection {
		keyVal += s.doc.content
	}
	text := (*parts)[idx].Text
	(*parts)[idx] = &genai.Part{Text: strings.Replace(string(text), key, keyVal, 1)}
}

// prependToParts extends prompts with digest selection.
func prependToParts(parts *[]*genai.Part, selection []QueryResult) {
	var res []*genai.Part
	for _, s := range selection {
		res = append(res, &genai.Part{Text: s.doc.content})
	}
	*parts = append(res, (*parts)...)
}

// appendToSelection extends selection with a query result in decreasing order of MMR up to k chunks.
func appendToSelection(selection []QueryResult, item QueryResult, k int) []QueryResult {
	result := selection
	result = append(result, item)
	sort.Slice(result[:], func(i, j int) bool {
		return result[i].mmr > result[j].mmr
	})
	if len(result) > k {
		return result[0:k]
	}
	return result
}

// processFunctionCalls attempts function calls, first across MCP sessions then gen tools.
func processFunctionCalls(ctx context.Context, fcMap map[string]*genai.FunctionCall) (*genai.Candidate, error) {
	var res []*genai.Part
	for _, fc := range fcMap {
		mcpRes, err := invokeMCPTool(ctx, fc)
		if err != nil {
			return nil, err
		}
		if mcpRes != nil {
			res = append(res, mcpRes)
			continue
		}
		// fc not an MCP tool, must be a native gen tool
		genRes, err := invokeGenTool(ctx, fc)
		if err != nil {
			return nil, err
		}
		res = append(res, genRes)
	}
	return &genai.Candidate{
		Content: &genai.Content{
			Parts: res,
		},
	}, nil
}

// countMatches is a helper that returns how many strings in strArray contain cand (case-insensitive).
func countMatches(strArray []string, cand string) int {
	count := 0
	for _, s := range strArray {
		if strings.Contains(strings.ToLower(s), strings.ToLower(cand)) {
			count++
		}
	}
	return count
}

// anyMatches returns true if any of the candidates match in array.
func anyMatches(strArray []string, candidates ...string) bool {
	for _, c := range candidates {
		if countMatches(strArray, c) > 0 {
			return true
		}
	}
	return false
}

// allMatch returns true if all list elements match.
func allMatch(strArray []string, cand string) bool {
	l := len(strArray)
	return l > 0 && countMatches(strArray, cand) == l
}

// oneMatches returns true if one and only one matches.
func oneMatches(strArray []string, cand string) bool {
	return countMatches(strArray, cand) == 1
}

// oneMatches returns true if one and only one matches.
func zeroOrOneMatches(strArray []string, cand string) bool {
	return countMatches(strArray, cand) <= 1
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

// alignSchema between JSON type fields in MCP vs genai SDK.
func alignSchema(m map[string]any) {
	if m == nil {
		return
	}
	// Handle nullable types
	if t, ok := m["type"].([]any); ok {
		isNullable := false
		var primaryType string
		for _, v := range t {
			if s, ok := v.(string); ok {
				if s == "null" {
					isNullable = true
				} else {
					primaryType = s
				}
			}
		}
		if primaryType != "" {
			m["type"] = primaryType
		}
		if isNullable {
			m["nullable"] = true
		}
	}
	// Recursively clean properties
	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			if sub, ok := v.(map[string]any); ok {
				alignSchema(sub)
			}
		}
	}
	// Handle array items
	if items, ok := m["items"].(map[string]any); ok {
		alignSchema(items)
	}
	// Handle additionalProperties
	if addProps, ok := m["additionalProperties"].(map[string]any); ok {
		alignSchema(addProps)
	}
	// Remove unsupported keywords
	delete(m, "$schema")
	delete(m, "definitions")
	delete(m, "$ref") // GenAI often struggles with local refs
}

// isValidPart evaluates whether a part contains non-empty generative payload.
func isValidPart(p *genai.Part) bool {
	if p == nil {
		return false
	}
	if p.Text != "" {
		return true
	}
	if p.InlineData != nil && len(p.InlineData.Data) > 0 {
		return true
	}
	if p.FileData != nil && p.FileData.FileURI != "" {
		return true
	}
	if p.FunctionCall != nil && p.FunctionCall.Name != "" {
		return true
	}
	if p.FunctionResponse != nil && p.FunctionResponse.Name != "" {
		return true
	}
	if p.CodeExecutionResult != nil && (p.CodeExecutionResult.Outcome != "" || p.CodeExecutionResult.Output != "") {
		return true
	}
	if p.ExecutableCode != nil && p.ExecutableCode.Code != "" {
		return true
	}
	return false
}

// isYouTubeURL validates the link passed via -f to glob.
func isYouTubeURL(path string) bool {
	youtubeRegex := regexp.MustCompile(`(?i)^((?:https?:)?//)?((?:www|m)\.)?((?:youtube(?:-nocookie)?\.com|youtu.be))(/(?:[\w\-]+\?v=|embed/|v/|shorts/|live/)?)([\w\-]+)(\S+)?$`)
	return youtubeRegex.MatchString(path)
}
