package main

import (
	"bufio"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"

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

// hasInputFromPipe checks if input is being piped to the program.
func hasInputFromStdin(in io.Reader) bool {
	if f, ok := in.(*os.File); ok {
		fileInfo, _ := f.Stat()
		return fileInfo.Mode()&os.ModeCharDevice == 0
	} else {
		// avoid destructive io function call
		r := reflect.ValueOf(in)
		v := reflect.Indirect(r).FieldByName("s")
		return len(v.String()) > 0
	}
}

// hasOutputRedirected checks if ok to emit non-printable characters.
func hasOutputRedirected(out io.Writer) bool {
	if f, ok := out.(*os.File); ok {
		fileInfo, _ := f.Stat()
		return fileInfo.Mode()&os.ModeCharDevice == 0
	}
	return true
}

// readLine reads a line from standard input.
func readLine(r io.Reader) (string, error) {
	scanner := bufio.NewScanner(r)
	if scanner.Scan() {
		return scanner.Text(), nil
	}
	return "", scanner.Err()
}

// searchReplace performs string replacement based on key-value pairs.
func searchReplace(prompt string, pm ParamMap) string {
	res := prompt
	for k, v := range pm {
		res = strings.ReplaceAll(res, "{"+k+"}", v) // all occurrences
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

// hasInvokedTool checks for a function call request, invokes tool and wraps response for model
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

// emitCandidates prints LLM response candidates from GenerateContent
func emitCandidates(out io.Writer, resp []*genai.Candidate) {
	var res string
	for _, cand := range resp {
		if cand != nil && cand.Content != nil {
			for _, p := range cand.Content.Parts {
				if p.Text != "" {
					if !hasOutputRedirected(out) {
						fmt.Fprintf(out, "\033[97m%s\033[0m", p.Text)
					} else {
						fmt.Fprintf(out, "%s", p.Text)
					}
					res += p.Text
				}
				if p.FunctionResponse != nil {
					res += fmt.Sprintf("%+v", p.FunctionResponse)
				}
				if p.InlineData != nil {
					fmt.Fprint(out, p.InlineData.Data)
				}
			}
		}
	}
}

// emitHistory prints the chat history
// TODO improve layout
func emitHistory(out io.Writer, hist []*genai.Content) {
	var res string
	for _, c := range hist {
		for _, p := range c.Parts {
			res += p.Text
		}
	}
	fmt.Fprintf(out, "%s\n", res)
}

// genLogFatal refines the error if available.
func genLogFatal(err error) {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		log.Fatal(gerr)
	} else {
		log.Fatal(err)
	}
}

// uploadFile tracks state until FileStateActive reached.
func uploadFile(ctx context.Context, client *genai.Client, path string) (*genai.File, error) {
	file, err := client.Files.UploadFromPath(ctx, path, nil)
	if err != nil {
		return nil, fmt.Errorf("uploading file '%s': %w", path, err)
	}

	for file.State == genai.FileStateProcessing {
		time.Sleep(1 * time.Second)
		file, err = client.Files.Get(ctx, file.Name, nil)
		if err != nil {
			return nil, fmt.Errorf("processing state for '%s': %w", file.Name, err)
		}
	}
	if file.State != genai.FileStateActive {
		return nil, fmt.Errorf("uploaded file has state '%s': %w", file.State, err)
	}
	return file, nil
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

// filePathHandler processes a single file path.
// parts and sysParts are extended with file content.
func filePathHandler(ctx context.Context, client *genai.Client, filePathVal string, parts *[]*genai.Part, sysParts *[]*genai.Part, keyVals ParamMap) error {
	f, err := os.Open(filePathVal)
	if err != nil {
		return fmt.Errorf("opening file '%s': %w", filePathVal, err)
	}
	defer f.Close()

	switch path.Ext(filePathVal) {
	case pExt, siExt:
		data, err := io.ReadAll(f)
		if err != nil {
			return fmt.Errorf("reading file '%s': %w", filePathVal, err)
		}
		if path.Ext(filePathVal) == siExt {
			*sysParts = append(*sysParts, &genai.Part{Text: searchReplace(string(data), keyVals)})
		} else {
			*parts = append(*parts, &genai.Part{Text: searchReplace(string(data), keyVals)})
		}
	case ".jpg", ".jpeg", ".png", ".gif", ".webp",
		".mp3", ".wav", ".aiff", ".aac", ".ogg", ".flac", ".pdf":
		file, err := uploadFile(ctx, client, filePathVal)
		if err != nil {
			return fmt.Errorf("uploading file '%s': %w", filePathVal, err)
		}
		*parts = append(*parts, &genai.Part{FileData: &genai.FileData{
			FileURI:  file.URI,
			MIMEType: strings.Split(file.MIMEType, ";")[0],
		}})
	default:
		data, err := io.ReadAll(f)
		if err != nil {
			return fmt.Errorf("reading file %s: %w", filePathVal, err)
		}
		*parts = append(*parts, &genai.Part{Text: searchReplace(string(data), keyVals)})
	}

	return nil
}

func glob(ctx context.Context, client *genai.Client, filePathVal string, parts *[]*genai.Part, sysParts *[]*genai.Part, keyVals ParamMap) error {
	fileInfo, err := os.Stat(filePathVal)
	if err == nil && fileInfo.IsDir() {
		filePathVal = filepath.Join(filePathVal, "**/*")
	}
	matches, err := filepath.Glob(filePathVal)
	if err != nil {
		return fmt.Errorf("globbing '%s': %w", filePathVal, err)
	}
	if len(matches) == 0 {
		if _, err := os.Stat(filePathVal); os.IsNotExist(err) {
			return fmt.Errorf("directory or file not found: '%s'", filePathVal)
		}
	}
	// Iterate over the matches and process each file
	for _, match := range matches {
		err := filePathHandler(ctx, client, match, parts, sysParts, keyVals)
		if err != nil {
			return err
		}
	}
	return nil
}

func lastWord(buf *bytes.Buffer) string {
	str := buf.String()
	str = strings.TrimSpace(str)
	if str == "" {
		return ""
	}

	lines := strings.Split(str, "\n")
	if len(lines) < 2 {
		return ""
	}

	lastLine := lines[len(lines)-1]
	prevLine := lines[len(lines)-2]
	if prevLine != "" || len(strings.Fields(lastLine)) != 1 {
		return ""
	}

	words := strings.Fields(lastLine)
	if len(words) == 0 {
		return ""
	}

	last := words[0]
	rest := strings.Join(lines[:len(lines)-2], "\n")

	buf.Reset()
	buf.WriteString(rest)
	return last
}

func retrieveHistory(hist *[]*genai.Content) error {
	*hist = nil
	return nil
}

func persistChat(chat *genai.Chat) error {
	file, _ := os.OpenFile(".gen", os.O_CREATE, os.ModePerm)
	defer file.Close()
	encoder := json.NewEncoder(file)
	hist := chat.History(false)
	if err := encoder.Encode(hist); err != nil {
		return err
	}
	return nil
}
