package main

import (
	"bytes"
	"fmt"
	"reflect"
	"time"
)

// In extracts a tuple from the tuple space using `m` as template.
// The parameter must be a pointer so the value can be overwritten.
// It blocks until a matching value is found in the tuple space.
// The matching tuple is removed from the space.
// TODO handle `Input` channel closed; add Context support
func (a *Amanda) In(m Tuple) {
	for t := range a.Input {
		if match(m, t) {
			assign(m, t)
			return
		}
		// no match, put tuple back
		a.Output <- t
	}
}

// Rd is similar to In, except that it does not remove the matched tuple
// from the tuple space.
// TODO handle `Input` channel closed; add Context support
func (a *Amanda) Rd(m Tuple) {
	for t := range a.Input {
		if match(m, t) {
			assign(m, t)
			a.Output <- t
			return
		}
		a.Output <- t
	}
}

// Out inserts a tuple into the tuple space.
// This is a non-blocking operation.
func (a *Amanda) Out(t Tuple) {
	a.Output <- t
}

// Eval is similar to Out except it treats the tumple as a
// function signature to be launched within a goroutine.
// The function must be in the first argument in the `sig` slice.
// Remaining entries are treated as arguments.
func (a *Amanda) Eval(fn interface{}, args ...interface{}) {
	fnVal := reflect.ValueOf(fn)
	if fnVal.Kind() != reflect.Func {
		return
	}
	var argVals []reflect.Value
	for _, arg := range args {
		argVals = append(argVals, reflect.ValueOf(arg))
	}

	go func() {
		select {
		case <-a.Done:
			return
		case <-a.Timeout:
			return
		default:
			results := fnVal.Call(argVals)
			for _, result := range results {
				// TODO handle channel full
				a.Output <- result.Interface()
			}
			a.Done <- struct{}{}
		}
	}()
}

// match compares template `m` and tuple `t` for equality.
// It recursively handles nested structs, arrays and slices.
// Ensure the fields of structs are exported.
// `nil` acts as wildcard (aka formal in Linda), matching any
// value in that position. The order of fields in structs is
// significant but ignores their names.
func match(m, t interface{}) bool {
	mVal := reflect.ValueOf(m)
	tVal := reflect.ValueOf(t)

	// nil matches anything
	if m == nil {
		return true
	}

	// Dereference pointers
	if mVal.Kind() == reflect.Ptr {
		if mVal.IsNil() {
			return true
		}
		mVal = mVal.Elem()
	}
	if tVal.Kind() == reflect.Ptr {
		tVal = tVal.Elem()
	}

	// Check if types match
	if mVal.Kind() != tVal.Kind() {
		//fmt.Println("type mismatch")
		return false
	}

	switch mVal.Kind() {
	case reflect.Array, reflect.Slice:
		//fmt.Println("Array or Slice case")
		if mVal.Len() != tVal.Len() {
			return false
		}
		for i := 0; i < mVal.Len(); i++ {
			if !match(mVal.Index(i).Interface(), tVal.Index(i).Interface()) {
				return false
			}
		}
		return true
	case reflect.Struct:
		//fmt.Printf("struct case %v\n", mVal.Type())
		if mVal.Type() == reflect.TypeOf(bytes.Buffer{}) {
			//fmt.Println("bytes.Buffer case")
			// if template is an empty bytes.Buffer, match anything
			mBuffer, ok1 := m.(*bytes.Buffer)
			tBuffer, ok2 := t.(*bytes.Buffer)
			//fmt.Printf("ok? %v %v\n", ok1, ok2)
			if !ok1 || !ok2 {
				return false
			}
			//fmt.Printf("%v equal? %v\n", mBuffer.Bytes(), tBuffer.Bytes())
			return bytes.Equal(mBuffer.Bytes(), tBuffer.Bytes())
		} else {
			if mVal.NumField() != tVal.NumField() {
				return false
			}
			for i := 0; i < mVal.NumField(); i++ {
				//fmt.Printf("matching %v %v\n", mVal.Field(i).Interface(), tVal.Field(i).Interface())
				if !match(mVal.Field(i).Interface(), tVal.Field(i).Interface()) {
					return false
				}
			}
			return true
		}
	default:
		//fmt.Println("default case")
		return reflect.DeepEqual(mVal.Interface(), tVal.Interface())
	}
}

// assign recursively copies the value of a source variable (`src`) into
// a destination variable (`dest`). Note: `dest` must be a pointer.
// If `dest` is nil, it allocates a new value of the appropriate type before copying.
// It uses reflection to achieve this generic copying behavior.
func assign(dest, src interface{}) {
	destVal := reflect.ValueOf(dest)
	srcVal := reflect.ValueOf(src)
	if destVal.Kind() != reflect.Ptr {
		panic("Argument must be a pointer")
	}
	if destVal.IsNil() {
		destVal.Set(reflect.New(srcVal.Type()))
	}
	if srcVal.Kind() == reflect.Ptr {
		srcVal = srcVal.Elem()
	}
	assignRecur(destVal, srcVal)
}

// assignRecur performs a deep copy of the `src` value into the `dest` value.
// The function handles various data types including structs, slices, and
// pointers, ensuring type safety and correctly handling nested structures.
// Type mismatch will trigger a panic.
func assignRecur(dest, src reflect.Value) {
	if !src.IsValid() {
		return
	}
	if src.Kind() == reflect.Ptr && src.IsNil() {
		return
	}
	// Dereference pointer in source
	if src.Kind() == reflect.Ptr {
		src = src.Elem()
	}
	if dest.Kind() != reflect.Ptr && dest.Kind() != src.Kind() {
		panic(fmt.Sprintf("Type mismatch during assign: cannot set %v to %v\n", dest.Type(), src.Type()))
	}

	switch dest.Kind() {
	case reflect.Ptr:
		if dest.IsNil() {
			dest.Set(reflect.New(dest.Type().Elem()))
		}
		assignRecur(dest.Elem(), src)
	case reflect.Array:
		for i := 0; i < src.Len(); i++ {
			assignRecur(dest.Index(i), src.Index(i))
		}
	case reflect.Slice:
		dest := reflect.MakeSlice(dest.Type(), src.Len(), src.Len())
		for i := 0; i < src.Len(); i++ {
			assignRecur(dest.Index(i), src.Index(i))
		}
	case reflect.Struct:
		if dest.Type() == reflect.TypeOf(bytes.Buffer{}) {
			if !dest.CanSet() {
				panic("Cannot set bytes.Buffer in destination")
			}
			srcBuffer, ok := src.Interface().(bytes.Buffer)
			if !ok {
				panic("Source is not a bytes.Buffer")
			}
			destPtr := dest.Addr().Interface().(*bytes.Buffer)
			destPtr.Reset()
			_, err := destPtr.Write(srcBuffer.Bytes())
			if err != nil {
				panic(fmt.Sprintf("Error writing to bytes.Buffer: %v", err))
			}
		} else {
			for i := 0; i < dest.NumField(); i++ {
				assignRecur(dest.Field(i), src.Field(i))
			}
		}
	default:
		if dest.Type().AssignableTo(src.Type()) {
			dest.Set(src)
		} else {
			panic(fmt.Sprintf("Type mismatch during assign: cannot set %v to %v\n", dest.Type(), src.Type()))
		}
	}
}

func (a *Amanda) StartWithSecondsTimeout(timeout int) int {
	a.Timeout = time.After(time.Duration(timeout) * time.Second)
	select {
	case <-a.Timeout:
		return 1
	case <-a.Done:
		return 0
	}
}
