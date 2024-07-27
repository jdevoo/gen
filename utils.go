package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"reflect"
	"strings"

	"github.com/google/generative-ai-go/genai"
)

// ParamMap holds key-value pairs for string replacement.
type ParamMap map[string]string

// String implements the flag.Value interface for ParamMap.
func (*ParamMap) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
func (m *ParamMap) Set(kv string) error {
	param := strings.SplitN(kv, "=", 2) // limit splits to 2
	if len(param) != 2 {
		return fmt.Errorf("invalid parameter %s", kv)
	}
	(*m)[param[0]] = param[1]
	return nil
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

func knownTools() string {
	var res []string
	genTool := reflect.TypeOf(Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		res = append(res, genTool.Method(i).Name)
	}
	return strings.Join(res, ",")
}

func registerTools(model *genai.GenerativeModel, mode genai.FunctionCallingMode) {
	genTool := reflect.TypeOf(Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		m := genTool.Method(i)
		f := reflect.ValueOf(Tool{}).MethodByName(m.Name)
		t := f.Type()
		schema := make(map[string]*genai.Schema)
		if t.NumIn() > 0 {
			for j := 0; j < t.NumIn(); j++ {
				switch t.In(j).Kind() {
				case reflect.String:
					schema[fmt.Sprintf("arg%d", j)] = &genai.Schema{Type: genai.TypeString}
				}
			}
			model.Tools = append(model.Tools, &genai.Tool{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name: m.Name,
					Parameters: &genai.Schema{
						Type:       genai.TypeObject,
						Properties: schema,
					},
				}},
			})
		} else {
			model.Tools = append(model.Tools, &genai.Tool{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name: m.Name,
				}},
			})
		}
		model.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: mode,
			},
		}
	}
}

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
		log.Fatal(err)
	}
	return vals[0].String()
}

// printGeneratedResponse emits LLM content, invokes tool if FunctionCall found, returns token count
func emitGeneratedResponse(resp *genai.GenerateContentResponse, out io.Writer) int32 {
	var res string
	var tokenCount int32
	for _, cand := range resp.Candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if fc, ok := part.(genai.FunctionCall); ok {
					res += invokeTool(fc)
				} else {
					res += fmt.Sprint(part)
				}
			}
			tokenCount += cand.TokenCount
		}
	}
	fmt.Fprintf(out, "%s", res)
	return tokenCount
}