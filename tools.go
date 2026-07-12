package main

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"google.golang.org/genai"
)

// goTypeToGenAIType maps Go basic types to GenAI schema types.
func goTypeToGenAIType(k reflect.Kind) (genai.Type, error) {
	switch k {
	case reflect.String:
		return genai.TypeString, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return genai.TypeInteger, nil
	case reflect.Float32, reflect.Float64:
		return genai.TypeNumber, nil
	case reflect.Bool:
		return genai.TypeBoolean, nil
	default:
		return "", fmt.Errorf("unsupported kind: %s", k)
	}
}

// buildFieldSchema parses a struct field tag and type to construct a GenAI schema.
func buildFieldSchema(field reflect.StructField) (*genai.Schema, string, bool, error) {
	fieldType := field.Type
	isOptional := false

	if fieldType.Kind() == reflect.Ptr {
		isOptional = true
		fieldType = fieldType.Elem()
	}

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

	genaiType, err := goTypeToGenAIType(fieldType.Kind())
	if err != nil {
		return nil, "", false, fmt.Errorf("field '%s': %v", field.Name, err)
	}

	return &genai.Schema{
		Type:     genaiType,
		Nullable: &isOptional,
	}, name, isOptional, nil
}

// knownTools returns string of comma-separated function names.
func knownTools(ctx context.Context) (string, error) {
	params, ok := ctx.Value(paramsKey).(*Parameters)
	if !ok {
		return "", fmt.Errorf("knownTools: params not found in context")
	}

	var res []string

	// gen tools
	genTool := reflect.TypeOf(Tool{})
	for i := 0; i < genTool.NumMethod(); i++ {
		res = append(res, sigGenTool(genTool.Method(i)))
	}

	// MCP tools
	for _, sess := range params.MCPSessions {
		if sess == nil {
			return "", fmt.Errorf("knownTools: nil session pointer found")
		}
		ltr, err := sess.ListTools(ctx, nil)
		if err != nil {
			break
		}
		for _, tool := range ltr.Tools {
			res = append(res, sigMCPTool(tool))
		}
	}

	return strings.Join(res, "\n"), nil
}

// sigGenTool inspects a native Gen Tool and returns its signature.
func sigGenTool(m reflect.Method) string {
	f := reflect.ValueOf(Tool{}).MethodByName(m.Name)
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
	f := reflect.ValueOf(Tool{}).MethodByName(fc.Name)
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

			// Dereference / assign values to fields recursively
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
		var paramErr *ParamError
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
