package main

import (
	"bytes"
	"database/sql"
	"errors"
	"flag"
	"fmt"
	"log"
	"reflect"
	"regexp"
	"sort"
	"strings"

	_ "github.com/lib/pq"

	"google.golang.org/api/googleapi"
	"google.golang.org/genai"
)

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

// ParamArray holds a list of strings.
type ParamArray []string

// String implements the flag.Value interface for ParamMap.
func (*ParamArray) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
func (a *ParamArray) Set(val string) error {
	*a = append(*a, val)
	return nil
}

// derefParts is a kludge for SendMessageStream
func derefParts(parts []*genai.Part) []genai.Part {
	res := []genai.Part{}
	for _, p := range parts {
		res = append(res, *p)
	}
	return res
}

// conjoin returns a single text resulting from concatenation of all original parts.
// TODO handle other part types
func conjTexts(parts *[]*genai.Part) {
	if len(*parts) == 0 {
		return
	}
	var buf bytes.Buffer
	for i, p := range *parts {
		if p.Text != "" {
			if i > 0 && buf.Len() > 0 {
				buf.WriteString(" ")
			}
			buf.WriteString(string(p.Text))
		}
	}
	*parts = []*genai.Part{{Text: buf.String()}}
}

// searchReplace performs string replacement based on key-value pairs.
func searchReplace(prompt string, pm ParamMap) string {
	res := prompt
	for k, v := range pm {
		searchRegex := regexp.MustCompile("(?i){" + k + "}")
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

// knownTools returns string of comma-separated function names.
func knownTools() string {
	var res []string
	genTool := reflect.TypeOf(Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		res = append(res, fmt.Sprintf("  * %s", genTool.Method(i).Name))
	}
	return strings.Join(res, "\n")
}

// registerTools declares functions of Tool in genai.FunctionDeclaration format.
// TODO support other argument types than string
func registerTools(config *genai.GenerateContentConfig) {
	genTool := reflect.TypeOf(Tool{})
	n := genTool.NumMethod()
	funDecl := make([]*genai.FunctionDeclaration, n)
	for i := 0; i < n; i++ {
		m := genTool.Method(i)
		f := reflect.ValueOf(Tool{}).MethodByName(m.Name)
		t := f.Type()
		argMap := make(map[string]*genai.Schema)
		if t.NumIn() > 0 {
			for j := 0; j < t.NumIn(); j++ {
				switch t.In(j).Kind() {
				case reflect.String:
					argMap[fmt.Sprintf("arg%d", j)] = &genai.Schema{Type: genai.TypeString}
				}
			}
			funDecl[i] = &genai.FunctionDeclaration{
				Name: m.Name,
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: argMap,
				},
			}
		} else {
			funDecl[i] = &genai.FunctionDeclaration{
				Name: m.Name,
			}
		}
	}
	config.Tools = make([]*genai.Tool, 1)
	config.Tools[0] = &genai.Tool{
		FunctionDeclarations: funDecl,
	}
}

// invokeTool calls tool identified by genai.FunctionCall using anonymous argument names.
func invokeTool(fc genai.FunctionCall) string {
	f := reflect.ValueOf(Tool{}).MethodByName(fc.Name)
	var args []reflect.Value
	for i := 0; i < len(fc.Args); i++ {
		t := f.Type().In(i)
		v := reflect.New(t).Elem()
		arg := fc.Args[fmt.Sprintf("arg%d", i)]
		switch t.Kind() {
		case reflect.String:
			v.SetString(arg.(string))
		}
		args = append(args, v)
	}
	vals := f.Call(args)
	if err := vals[1].Interface(); err != nil {
		return fmt.Sprintf("%s error: %s", fc.Name, err)
	}
	return vals[0].String()
}

// hasInvokedTool checks for a function call request, invokes tool and wraps response for model.
// TODO tool error handling
func hasInvokedTool(resp *genai.GenerateContentResponse) (bool, *genai.FunctionResponse) {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return false, &genai.FunctionResponse{}
	}
	for _, fc := range resp.FunctionCalls() {
		res := invokeTool(*fc)
		return true, &genai.FunctionResponse{
			Name: fc.Name,
			Response: map[string]any{
				"Response": res,
			},
		}
	}
	return false, &genai.FunctionResponse{}
}

// genLogFatal refines the error if available and exits with 1
func genLogFatal(err error) {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		log.Fatal(gerr)
	} else {
		log.Fatal(err)
	}
}

// anyMatches returns true if any of the match candidates are found in
func anyMatches(strArray []string, candidates ...string) bool {
	for _, s := range strArray {
		for _, c := range candidates {
			if strings.Contains(strings.ToLower(s), strings.ToLower(c)) {
				return true
			}
		}
	}
	return false
}

// allMatch returns true if all list elements match.
func allMatch(strArray []string, cand string) bool {
	if len(strArray) == 0 {
		return false
	}
	for _, s := range strArray {
		if !strings.Contains(strings.ToLower(s), strings.ToLower(cand)) {
			return false
		}
	}
	return true
}

// oneMatches returns true if one and only one matches.
func oneMatches(strArray []string, cand string) bool {
	var res int
	for _, s := range strArray {
		if strings.Contains(strings.ToLower(s), strings.ToLower(cand)) {
			res += 1
		}
	}
	return res == 1
}

// QueryPostgres submits query to database set by DSN parameter.
func queryPostgres(query string) (string, error) {
	var res []string
	dsn, ok := keyVals["DSN"]
	if !ok || len(dsn) == 0 {
		return "", fmt.Errorf("DSN parameter missing")
	}
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return "", fmt.Errorf("opening DSN '%s': %w", dsn, err)
	}
	defer db.Close()
	rows, err := db.Query(query)
	if err != nil {
		return "", fmt.Errorf("for query '%s': %w", query, err)
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
}

// isFlagSet visits the flags passed to the command at runtime.
func isFlagSet(name string) bool {
	res := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			res = true
		}
	})
	return res
}
