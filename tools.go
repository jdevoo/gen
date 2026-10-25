package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/jdevoo/gen/core"
	"github.com/jdevoo/gen/toolbox"
	"google.golang.org/genai"
)

// typeToGenAISchema recursively converts a Go type into a genai Schema.
func typeToGenAISchema(t reflect.Type) (*genai.Schema, error) {
	isNullable := false
	if t.Kind() == reflect.Ptr {
		isNullable = true
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.String:
		return &genai.Schema{Type: genai.TypeString, Nullable: &isNullable}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &genai.Schema{Type: genai.TypeInteger, Nullable: &isNullable}, nil
	case reflect.Float32, reflect.Float64:
		return &genai.Schema{Type: genai.TypeNumber, Nullable: &isNullable}, nil
	case reflect.Bool:
		return &genai.Schema{Type: genai.TypeBoolean, Nullable: &isNullable}, nil
	case reflect.Slice:
		itemSchema, err := typeToGenAISchema(t.Elem())
		if err != nil {
			return nil, err
		}
		return &genai.Schema{
			Type:     genai.TypeArray,
			Items:    itemSchema,
			Nullable: &isNullable,
		}, nil
	case reflect.Struct:
		propMap := make(map[string]*genai.Schema)
		var req []string
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			if !field.IsExported() {
				continue
			}
			fSchema, fName, fOptional, err := buildFieldSchema(field)
			if err != nil {
				return nil, err
			}
			propMap[fName] = fSchema
			if !fOptional {
				req = append(req, fName)
			}
		}
		return &genai.Schema{
			Type:       genai.TypeObject,
			Properties: propMap,
			Required:   req,
			Nullable:   &isNullable,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported type: %s", t)
	}
}

// buildFieldSchema parses a struct field tag and type to construct a genai Schema.
func buildFieldSchema(field reflect.StructField) (*genai.Schema, string, bool, error) {
	fieldType := field.Type
	isOptional := fieldType.Kind() == reflect.Ptr

	name := field.Tag.Get("json")
	parts := strings.Split(name, ",")
	name = parts[0]
	if name == "" {
		name = field.Name
	}

	for _, p := range parts[1:] {
		if p == "omitempty" {
			isOptional = true
		}
	}

	schema, err := typeToGenAISchema(fieldType)
	if err != nil {
		return nil, "", false, fmt.Errorf("field '%s': %v", field.Name, err)
	}

	return schema, name, isOptional, nil
}

// knownTools returns string of comma-separated function names.
func knownTools(ctx context.Context) (string, error) {
	params, _ := ctx.Value(core.ParamsKey).(*core.Parameters)

	var res []string

	// gen tools
	genTool := reflect.TypeOf(toolbox.Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		res = append(res, sigGenTool(genTool.Method(i)))
	}

	// MCP tools
	for _, sess := range params.MCPSessions {
		if sess == nil {
			return "", fmt.Errorf("knownTools: nil session pointer found")
		}
		for tool, err := range sess.Tools(ctx, nil) {
			if err != nil {
				break
			}
			res = append(res, sigMCPTool(tool))
		}
	}

	return strings.Join(res, "\n"), nil
}

// sigGenTool inspects a native Gen Tool and returns its signature.
func sigGenTool(m reflect.Method) string {
	f := reflect.ValueOf(toolbox.Tool{}).MethodByName(m.Name)
	t := f.Type()
	var params []string
	// First arg is always context.Context, start inspecting from index 1
	if t.NumIn() > 1 {
		argType := t.In(1)
		if argType.Kind() == reflect.Struct {
			for j := 0; j < argType.NumField(); j++ {
				field := argType.Field(j)
				fieldType := field.Type
				isOptional := false
				if fieldType.Kind() == reflect.Ptr {
					isOptional = true
					fieldType = fieldType.Elem()
				}
				tag := field.Tag.Get("json")
				tagParts := strings.Split(tag, ",")
				name := tagParts[0]
				if name == "" {
					name = field.Name
				}
				if len(tagParts) > 1 {
					for _, p := range tagParts[1:] {
						if p == "omitempty" {
							isOptional = true
						}
					}
				}
				if !isOptional {
					name += "*"
				}
				var paramStr string
				if fieldType.Kind() == reflect.String {
					paramStr = name
				} else if fieldType.Kind() == reflect.Slice {
					paramStr = fmt.Sprintf("%s ([]%s)", name, fieldType.Elem().Kind().String())
				} else if fieldType.Kind() == reflect.Struct {
					paramStr = fmt.Sprintf("%s (object)", name)
				} else {
					paramStr = fmt.Sprintf("%s (%s)", name, fieldType.Kind().String())
				}
				params = append(params, paramStr)
			}
		} else {
			params = append(params, argType.Kind().String()+"*")
		}
	}
	if len(params) == 0 {
		return fmt.Sprintf("  • %s", m.Name)
	}
	return fmt.Sprintf("  • %s %s", m.Name, strings.Join(params, ", "))
}

// registerTools declares functions of type Tool in genai.FunctionDeclaration format.
func registerGenTools(config *genai.GenerateContentConfig) error {
	genTool := reflect.TypeOf(toolbox.Tool{})
	n := genTool.NumMethod()
	genDecls := make([]*genai.FunctionDeclaration, n)

	for i := 0; i < n; i++ {
		m := genTool.Method(i)
		f := reflect.ValueOf(toolbox.Tool{}).MethodByName(m.Name)
		t := f.Type()
		argMap := map[string]*genai.Schema{}
		var req []string

		// first tool arg must be context.Context, check if second args
		if t.NumIn() > 1 {
			argType := t.In(1)
			for j := 0; j < argType.NumField(); j++ {
				schema, name, isOptional, err := buildFieldSchema(argType.Field(j))
				if err != nil {
					return fmt.Errorf("tool '%s': %v", m.Name, err)
				}
				argMap[name] = schema
				if !isOptional {
					req = append(req, name)
				}
			}
		}

		decl := &genai.FunctionDeclaration{Name: m.Name}
		if len(argMap) > 0 {
			decl.Parameters = &genai.Schema{
				Type:       genai.TypeObject,
				Properties: argMap,
				Required:   req,
			}
		}
		genDecls[i] = decl
	}

	if len(genDecls) > 0 {
		config.Tools = append(config.Tools, &genai.Tool{
			FunctionDeclarations: genDecls,
		})
	}
	return nil
}

// invokeGenTool looks for exported symbols under Tool matching the provided FunctionCall signature.
func invokeGenTool(ctx context.Context, fc *genai.FunctionCall) (*genai.Part, error) {
	f := reflect.ValueOf(toolbox.Tool{}).MethodByName(fc.Name)
	if !f.IsValid() {
		return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"error": fmt.Sprintf("invokeGenTool: %s invocation error", fc.Name),
		}), nil
	}

	args := []reflect.Value{reflect.ValueOf(ctx)} // first tool arg is context.Context
	if f.Type().NumIn() == 2 {                    // has parameters struct
		structType := f.Type().In(1)
		if structType.Kind() != reflect.Struct {
			return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
				"error": fmt.Sprintf("invokeGenTool: %s expected second argument to be a struct", fc.Name),
			}), nil
		}

		structVal := reflect.New(structType).Elem()
		for i := 0; i < structType.NumField(); i++ {
			field := structType.Field(i)
			name := field.Tag.Get("json")
			parts := strings.Split(name, ",")
			name = parts[0]
			if name == "" {
				name = field.Name
			}
			isOptional := false
			if field.Type.Kind() == reflect.Ptr {
				isOptional = true
			}
			if len(parts) > 1 {
				for _, p := range parts[1:] {
					if p == "omitempty" {
						isOptional = true
					}
				}
			}

			argVal, ok := fc.Args[name]
			if !ok {
				if !isOptional {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s missing parameter: '%s'", fc.Name, name),
					}), nil
				}
				continue
			}

			// Dereference / assign values to fields
			targetField := structVal.Field(i)
			fieldType := field.Type
			var val reflect.Value
			isPtr := fieldType.Kind() == reflect.Ptr
			var baseType reflect.Type
			if isPtr {
				baseType = fieldType.Elem()
				val = reflect.New(baseType).Elem()
			} else {
				baseType = fieldType
				val = targetField
			}

			switch baseType.Kind() {
			case reflect.Slice, reflect.Struct:
				jsonBytes, err := json.Marshal(argVal)
				if err != nil {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s error marshaling parameter '%s': %v", fc.Name, name, err),
					}), nil
				}
				targetPtr := reflect.New(baseType)
				if err := json.Unmarshal(jsonBytes, targetPtr.Interface()); err != nil {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s error unmarshaling parameter '%s' as %s: %v", fc.Name, name, baseType, err),
					}), nil
				}
				val.Set(targetPtr.Elem())
			case reflect.String:
				if s, ok := argVal.(string); ok {
					val.SetString(s)
				} else {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s type mismatch: '%s' expected string, got %T", fc.Name, name, argVal),
					}), nil
				}
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				if fv, ok := argVal.(float64); ok {
					val.SetInt(int64(fv))
				} else if iv, ok := argVal.(int64); ok {
					val.SetInt(iv)
				} else {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s type mismatch: '%s' expected integer, got %T", fc.Name, name, argVal),
					}), nil
				}
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				var uintVal uint64
				if fv, ok := argVal.(float64); ok {
					if fv < 0 {
						return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
							"error": fmt.Sprintf("%s error: negative value for unsigned integer '%s'", fc.Name, name),
						}), nil
					}
					uintVal = uint64(fv)
				} else if iv, ok := argVal.(int64); ok {
					if iv < 0 {
						return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
							"error": fmt.Sprintf("%s error: negative value for unsigned integer '%s'", fc.Name, name),
						}), nil
					}
					uintVal = uint64(iv)
				} else {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s type mismatch: '%s' expected unsigned integer, got %T", fc.Name, name, argVal),
					}), nil
				}
				val.SetUint(uintVal)
			case reflect.Float32, reflect.Float64:
				if fv, ok := argVal.(float64); ok {
					val.SetFloat(fv)
				} else if iv, ok := argVal.(int64); ok {
					val.SetFloat(float64(iv))
				} else {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s type mismatch: '%s' expected float, got %T", fc.Name, name, argVal),
					}), nil
				}
			case reflect.Bool:
				if b, ok := argVal.(bool); ok {
					val.SetBool(b)
				} else {
					return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
						"error": fmt.Sprintf("%s type mismatch: '%s' expected boolean, got %T", fc.Name, name, argVal),
					}), nil
				}
			}

			if isPtr {
				targetField.Set(val.Addr())
			}
		}
		args = append(args, structVal)
	}

	vals := f.Call(args)
	if !vals[1].IsNil() {
		var paramErr *core.ParamError
		err := vals[1].Interface().(error)
		if errors.As(err, &paramErr) {
			return nil, paramErr
		}
		return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"error": fmt.Sprintf("%s error: %v", fc.Name, err),
		}), nil
	}

	outVal := vals[0].Interface()
	if outVal == nil {
		return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
			"output": "SUCCESS",
		}), nil
	}
	if p, ok := outVal.(*genai.Part); ok {
		return p, nil
	}
	// fallback
	return genai.NewPartFromFunctionResponse(fc.Name, map[string]any{
		"output": fmt.Sprintf("%v", outVal),
	}), nil
}
