package main

import (
	"bytes"
	"context"
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

// ParamArray holds a list of strings e.g. file paths.
type ParamArray []string

// String implements the flag.Value interface for ParamMap.
func (*ParamArray) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
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

// knownTools returns string of comma-separated function names.
func knownTools(ctx context.Context) (string, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return "", fmt.Errorf("knownTools: params not found in context")
	}

	var res []string

	// gen tools
	genTool := reflect.TypeOf(Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		res = append(res, fmt.Sprintf("  * %s", genTool.Method(i).Name))
	}

	// MCP tools
	for _, sess := range params.MCPSessions {
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			res = append(res, fmt.Sprintf("  * %v", err))
			continue
		}
		for _, tool := range ltr.Tools {
			res = append(res, fmt.Sprintf("  * %v", tool.Name))
		}
	}

	return strings.Join(res, "\n"), nil
}

// registerTools declares functions of type Tool in genai.FunctionDeclaration format.
// TODO add support for arrays and objects
func registerGenTools(config *genai.GenerateContentConfig) error {
	genTool := reflect.TypeOf(Tool{})
	n := genTool.NumMethod()
	genDecls := make([]*genai.FunctionDeclaration, n)
	for i := 0; i < n; i++ {
		m := genTool.Method(i)
		f := reflect.ValueOf(Tool{}).MethodByName(m.Name)
		t := f.Type()
		argMap := map[string]*genai.Schema{}
		if t.NumIn() > 1 { // first tool arg must be context.Context
			for j := 1; j < t.NumIn(); j++ {
				switch t.In(j).Kind() {
				case reflect.String:
					argMap[t.In(j).Name()] = &genai.Schema{Type: genai.TypeString}
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
					argMap[t.In(j).Name()] = &genai.Schema{Type: genai.TypeInteger}
				case reflect.Float32, reflect.Float64:
					argMap[t.In(j).Name()] = &genai.Schema{Type: genai.TypeNumber}
				case reflect.Bool:
					argMap[t.In(j).Name()] = &genai.Schema{Type: genai.TypeBoolean}
				default:
					return fmt.Errorf("unsupported type for tool '%s'", m.Name)
				}
			}
			genDecls[i] = &genai.FunctionDeclaration{
				Name: m.Name,
				Parameters: &genai.Schema{
					Type:       genai.TypeObject,
					Properties: argMap,
				},
			}
		} else {
			genDecls[i] = &genai.FunctionDeclaration{
				Name: m.Name,
			}
		}
	}
	if len(genDecls) > 0 {
		config.Tools = append(config.Tools, &genai.Tool{
			FunctionDeclarations: genDecls,
		})
	}
	return nil
}

// invokeGenTool looks for exported symbols under Tool matching the provided FunctionCall signature.
func invokeGenTool(ctx context.Context, fc *genai.FunctionCall) (string, string) {
	f := reflect.ValueOf(Tool{}).MethodByName(fc.Name)
	if !f.IsValid() {
		return "", fmt.Sprintf("invokeTool: %s invocation error", fc.Name)
	}
	args := []reflect.Value{reflect.ValueOf(ctx)} // first tool arg is context.Context
	for i := 1; i < len(fc.Args)+1; i++ {
		t := f.Type().In(i)
		v := reflect.New(t).Elem()
		argName := f.Type().In(i).Name()
		argVal, ok := fc.Args[argName]
		if !ok {
			args = append(args, v) // arg missing, use zero value
			continue
		}
		switch t.Kind() {
		case reflect.String:
			if s, ok := argVal.(string); ok {
				v.SetString(s)
			} else {
				return "", fmt.Sprintf("%s type mismatch: '%s' expected string, got %T", fc.Name, argName, argVal)
			}
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			if fv, ok := argVal.(float64); ok {
				v.SetInt(int64(fv))
			} else if iv, ok := argVal.(int64); ok {
				v.SetInt(iv)
			} else {
				return "", fmt.Sprintf("%s type mismatch: '%s' expected integer, got %T", fc.Name, argName, argVal)
			}
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			var uintVal uint64
			if fv, ok := argVal.(float64); ok {
				if fv < 0 {
					return "", fmt.Sprintf("%s error: negative value for unsigned integer '%s'", fc.Name, argName)
				}
				uintVal = uint64(fv)
			} else if iv, ok := argVal.(int64); ok {
				if iv < 0 {
					return "", fmt.Sprintf("%s error: negative value for unsigned integer '%s'", fc.Name, argName)
				}
				uintVal = uint64(iv)
			} else {
				return "", fmt.Sprintf("%s type mismatch: '%s' expected unsigned integer, got %T", fc.Name, argName, argVal)
			}
			v.SetUint(uintVal)
		case reflect.Float32, reflect.Float64:
			if fv, ok := argVal.(float64); ok {
				v.SetFloat(fv)
			} else if iv, ok := argVal.(int64); ok {
				v.SetFloat(float64(iv))
			} else {
				return "", fmt.Sprintf("%s type mismatch: '%s' expected float, got %T", fc.Name, argName, argVal)
			}
		case reflect.Bool:
			if b, ok := argVal.(bool); ok {
				v.SetBool(b)
			} else {
				return "", fmt.Sprintf("%s type mismatch: '%s' expected boolean, got %T", fc.Name, argName, argVal)
			}
		}
		args = append(args, v)
	}
	vals := f.Call(args)
	if err := vals[1].Interface(); err != nil {
		return "", fmt.Sprintf("%s error: %v", fc.Name, err)
	}
	return vals[0].String(), ""
}

// processFunCalls looks for suggested function calls across MCP sessions and gen tools.
func processFunCalls(ctx context.Context, resp *genai.GenerateContentResponse) []*genai.Part {
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return []*genai.Part{}
	}
	for _, fc := range resp.FunctionCalls() {
		//parts = append(parts, &genai.Part{
		//	FunctionCall: fc,
		//})
		if res := invokeMCPTool(ctx, fc); len(res) > 0 {
			return res
		}
		res, err := invokeGenTool(ctx, fc)
		if res != "" || err != "" {
			return []*genai.Part{
				genai.NewPartFromFunctionResponse(fc.Name, map[string]any{"output": res, "error": err}),
			}
		}
	}
	return []*genai.Part{}
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
func queryPostgres(ctx context.Context, query string) (string, error) {
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		return "", fmt.Errorf("queryPostgres: keyVals not found in context")
	}
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
	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return "", fmt.Errorf("for query '%s': %w", query, err)
	}
	defer rows.Close()
	cols, _ := rows.Columns()
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

// validPrompts checks prompts against regular interactive vs no redirect or piped content session.
func validPrompts(params *Parameters) bool {
	if (params.Interactive &&
		// no regular prompt privided
		((len(params.Args) == 0 && !anyMatches(params.FilePaths, PExt)) ||
			// system instruction
			(params.SystemInstruction &&
				// not provided as file
				((len(params.Args) == 0 && !anyMatches(params.FilePaths, SPExt)) ||
					// provided as argument but no prompt as file and no chat mode
					(len(params.Args) > 0 && !anyMatches(params.FilePaths, PExt) && !params.ChatMode))))) ||
		(!params.Interactive &&
			// not set as file xor argument
			((!oneMatches(params.FilePaths, "-") && !(len(params.Args) == 1 && params.Args[0] == "-")) ||
				// system instruction
				(params.SystemInstruction &&
					// stdin as file, but no prompt as file or argument
					((!oneMatches(params.FilePaths, "-") && len(params.Args) == 0 && !anyMatches(params.FilePaths, PExt)) ||
						// stdin as argument, no prompt as file
						(len(params.Args) == 1 && params.Args[0] == "-" && !anyMatches(params.FilePaths, PExt) && !params.ChatMode))))) {
		return false
	}
	return true
}

func validRanges(params *Parameters) bool {
	if
	// invalid k values
	(params.K < 0 || params.K > 10) ||
		// invalid lambda values
		(params.Lambda < 0 || params.Lambda > 1) ||
		// invalid temperature values
		(params.Temp < 0 || params.Temp > 2) ||
		// invalid topP values
		(params.TopP < 0 || params.TopP > 1) {
		return false
	}
	return true
}

func validCombos(params *Parameters) bool {
	if
	// code execution with incompatible flags
	(params.CodeGen &&
		(params.JSON || params.Tool || params.GoogleSearch || params.Embed)) ||
		// tool registration with incompatible flags
		(params.Tool &&
			(params.JSON || params.CodeGen || params.GoogleSearch || params.SystemInstruction || params.Embed)) ||
		// search with incompatible flags
		(params.GoogleSearch &&
			(params.JSON || params.Tool || params.CodeGen || params.Embed)) ||
		// image modality with incompatible flags
		(params.ImgModality &&
			(params.GoogleSearch || params.CodeGen || params.Tool || params.JSON || params.ChatMode || params.Embed)) ||
		// walk without file attached that is not some prompt
		(params.Walk &&
			(len(params.FilePaths) == 0 || allMatch(params.FilePaths, PExt) || allMatch(params.FilePaths, SPExt))) ||
		// chat mode
		(params.ChatMode &&
			// with incompatible flags
			(params.JSON || params.GoogleSearch || params.CodeGen || params.Embed)) {
		return false
	}
	return true
}

func validEmbeddings(params *Parameters, keyVals ParamMap) bool {
	if
	// embeddings
	params.Embed &&
		// incompatible flags
		(params.Unsafe || params.JSON ||
			isFlagSet("temp") || isFlagSet("top_p") || isFlagSet("k") || isFlagSet("l") ||
			// no digest set
			len(params.DigestPaths) != 1 ||
			// metadata missing
			(params.OnlyKvs && len(keyVals) == 0) ||
			// prompts set
			anyMatches(params.FilePaths, PExt) || anyMatches(params.FilePaths, SPExt) ||
			// no arguments or files to digest
			(!params.Interactive &&
				!((len(params.Args) == 1 && params.Args[0] == "-") || oneMatches(params.FilePaths, "-")))) {

		return false
	}
	return true
}

// isValidParams performs a complete argument validation.
func isParamsInvalid(params *Parameters, keyVals ParamMap) bool {
	if validPrompts(params) &&
		validRanges(params) &&
		validCombos(params) &&
		validEmbeddings(params, keyVals) {
		return false
	}
	return true
}
