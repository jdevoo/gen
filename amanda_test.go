package main

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestMatch(t *testing.T) {
	tests := []struct {
		m      interface{}
		t      interface{}
		expect bool
	}{
		// template, tuple, outcome
		{nil, nil, true},
		{nil, 1, true},
		{1, 1, true},
		{nil, "hello", true},
		{1, nil, false},
		{[]interface{}{1, nil}, []interface{}{1, "hello"}, true},
		{[]interface{}{nil, "hello"}, []interface{}{2, "world"}, false},
		{[]interface{}{nil, "x", nil}, []interface{}{1, "x", 3}, true},
		{[]interface{}{1, 2}, []interface{}{3, 4}, false},
		{struct {
			A *string
			B *int
		}{String("hello"), nil}, struct {
			A *string
			B *int
		}{String("hello"), Decimal(2)}, true},
		{struct {
			A *string
			B *int
		}{nil, nil}, struct {
			A *string
			B *int
		}{String("hello"), Decimal(2)}, true},
		{"hello", nil, false},
		// NewBufferString returns *bytes.Buffer
		{nil, bytes.NewBufferString("foo"), true},
		{bytes.Buffer{}, bytes.NewBufferString("world"), false},
		{bytes.NewBufferString("hello"), bytes.NewBufferString("hello"), true},
		{bytes.NewBufferString("hello"), bytes.NewBufferString("world"), false},
		{nil, bytes.NewBufferString("world"), true},
	}

	for i, tc := range tests {
		t.Run(fmt.Sprintf("Test %d", i), func(t *testing.T) {
			result := match(tc.m, tc.t)
			if result != tc.expect {
				t.Errorf("match(%v, %v) = %t; want %t", tc.m, tc.t, result, tc.expect)
			}
		})
	}
}

func TestEval(t *testing.T) {
	type testCase struct {
		sig      []interface{}
		expected []interface{}
	}

	tests := []testCase{
		{
			sig: []interface{}{
				func(a int, b int) int { return a + b },
				1,
				2,
			},
			expected: []interface{}{3},
		},
		{
			sig: []interface{}{
				func(a, b string) string { return a + b },
				"hello",
				" world",
			},
			expected: []interface{}{"hello world"},
		},
		{
			sig: []interface{}{
				func(a int) (int, string) { return a * 2, fmt.Sprintf("%d", a*2) },
				3,
			},
			expected: []interface{}{6, "6"},
		},
		{
			sig: []interface{}{
				func() int { return 42 },
			},
			expected: []interface{}{42},
		},
		{
			// Test with no arguments
			sig:      []interface{}{func() {}},
			expected: []interface{}{},
		},
		{
			// Test with invalid signature (not a function)
			sig:      []interface{}{"not a function"},
			expected: []interface{}{},
		},
	}

	for i, tc := range tests {
		t.Run(fmt.Sprintf("Test case %d", i+1), func(t *testing.T) {
			input := make(chan interface{}, 10)
			output := make(chan interface{}, 10)
			a := &Amanda{Input: input, Output: output}
			var wg sync.WaitGroup
			wg.Add(1)

			a.Eval(tc.sig[0], tc.sig[1:]...)

			go func() {
				defer wg.Done()
				var results []interface{}
				timeout := time.After(100 * time.Millisecond)
				if len(tc.expected) == 0 {
					return
				}
				for {
					select {
					case res := <-output:
						results = append(results, res)
					case <-timeout:
						t.Errorf("Test case %d timed out waiting for results", i+1)
						return
					}
					if len(results) == len(tc.expected) {
						break
					}
				}
				if !reflect.DeepEqual(results, tc.expected) {
					t.Errorf("Test case %d failed: Expected %v, got %v", i+1, tc.expected, results)
				}
			}()

			wg.Wait()

			close(input)
			close(output)
		})
	}
}

// This test simulates the Dining Philosophers using an Amanda tuple space.
// It expires after 5s at which point philosphers will be done thinking and eating.
func TestPhilosophers(t *testing.T) {
	type (
		chopstick int
		ticket    struct{}
	)

	num := 5
	ts := TupleSpace()
	for i := 0; i < num; i++ {
		ts.Out(chopstick(i))
		ts.Eval(
			// do is a function for an active Amanda tuple.
			// A philosopher who is ready to enter the dining room uses In() to grab a ticket.
			// If there are no free tickets, she will block until some other philosopher leaves
			// and releases his ticket.
			// Once inside, she uses In() to grab chopsticks on each side.
			// Left and right chopsticks are represented by separate tuples.
			// When done eating, the philosopher returns both chopsticks and the ticket.
			func(i int) {
				for {
					time.Sleep(time.Duration(rand.Int32N(100)) * time.Millisecond) // think
					t := ticket{}
					ts.In(&t)
					c1 := chopstick(i)
					ts.In(&c1)
					c2 := chopstick((i + 1) % num)
					ts.In(&c2)
					time.Sleep(time.Duration(rand.Int32N(100)) * time.Millisecond) // eat
					ts.Out(c1)
					ts.Out(c2)
					ts.Out(t)
				}
			},
			i,
		)
		// issue one less ticket as there are philosophers
		if i < (num - 1) {
			ts.Out(ticket{})
		}
	}
	res := ts.StartWithSecondsTimeout(5)
	if res != 1 {
		t.Errorf("expected timeout, got %d", res)
	}
}

// This test uses Amanda to pass work from agent to agent.
// The tuple space is populated with instruction structs that can
// be picked up by specific agents.
func TestWorkflow(t *testing.T) {
	type Instructions struct {
		Agent  string
		Result *string
	}

	ts := TupleSpace()
	ts.Out(Instructions{"Alice", nil})

	task := func(agent string, t *testing.T) {
		var exp int

		t.Logf("%s is running\n", agent)
		i := Instructions{agent, nil}
		for {
			ts.In(&i)
			time.Sleep(time.Duration(rand.Int32N(1000)) * time.Millisecond)
			switch agent {
			case "Alice":
				ts.Out(Instructions{"Bob", nil})
				ts.Out(Instructions{"Charlie", nil})
			case "Bob":
				ts.Out(Instructions{"Dave", nil})
			case "Charlie":
				ts.Out(Instructions{"Dave", nil})
			case "Dave":
				res := "task complete"
				ts.Out(Instructions{"Master", &res})
			case "Master":
				exp += 1
				t.Logf("%s\n", *i.Result)
				if exp == 2 {
					return
				}
			}
		}
	}
	ts.Eval(task, "Alice", t)
	ts.Eval(task, "Bob", t)
	ts.Eval(task, "Charlie", t)
	ts.Eval(task, "Dave", t)
	ts.Eval(task, "Master", t)

	res := ts.StartWithSecondsTimeout(30)
	if res != 0 {
		t.Errorf("expected done, got %d", res)
	}
}

type board struct {
	Fixed *int
	Val   [8]*int
	Col   [8]*int
	Fdiag [15]*int
	Bdiag [15]*int
}

// This test approaches the N queens problem using a tuple space as a blackboard
// shared by agents exploring various queen positions.
// In `main` a goroutine is launched that expires after a set duration. After this delay,
// it sends a signal to the `done` channel. When `main` receives this signal
// agents will stop exploring options.
// FIXME some times outputs 2 boards instead of one
func TestQueens(t *testing.T) {
	var crewSize = 5
	var n = 6
	var timeout = 120
	var mu sync.Mutex

	t.Logf("Solving for %d queens with a crew of %d for %ds...\n", n, crewSize, timeout)
	ts := TupleSpace()
	for i := 0; i < crewSize; i++ {
		b := board{Fixed: new(int)}
		for i := 0; i < 8; i++ {
			b.Val[i] = new(int)
			b.Col[i] = new(int)
		}
		for i := 0; i < 15; i++ {
			b.Fdiag[i] = new(int)
			b.Bdiag[i] = new(int)
		}
		ts.Out(b)
		ts.Eval(
			func(i int, want int, t *testing.T) {
				for {
					// pick any board from the blackboard
					var b board
					ts.In(&b)
					// consider zapping queens from the board
					if *b.Fixed > 3 && rand.Float32() < 0.1 {
						b.zapQueens(rand.IntN(2))
					}
					if b.foundAcceptableHint() {
						if *b.Fixed == want {
							mu.Lock()
							b.print(t)
							mu.Unlock()
							return
						}
					}
					ts.Out(b)
				}
			},
			i,
			n,
			t,
		)
	}
	res := ts.StartWithSecondsTimeout(timeout)
	if res != 0 {
		t.Errorf("expected done, got %d", res)
	}
}

func (b *board) foundAcceptableHint() bool {
	i := rand.IntN(8)
	j := rand.IntN(8)
	if b.isPromising(i, j) {
		*b.Col[j] = 1
		*b.Fdiag[i-j+7] = 1
		*b.Bdiag[i+j] = 1
		*b.Val[i] = j + 1
		*b.Fixed++
		return true
	}
	return false
}

func (b *board) zapQueens(n int) {
	todo := min(n, *b.Fixed)
	for _, i := range rand.Perm(8) {
		if *b.Val[i] != 0 {
			*b.Col[*b.Val[i]-1] = 0
			*b.Fdiag[i-*b.Val[i]+8] = 0
			*b.Bdiag[i+*b.Val[i]-1] = 0
			*b.Fixed--
			*b.Val[i] = 0
			todo -= 1
		}
		if todo == 0 {
			break
		}
	}
}

func (b board) isPromising(i int, j int) bool {
	return *b.Val[i] == 0 && *b.Col[j] == 0 && *b.Fdiag[i-j+7] == 0 && *b.Bdiag[i+j] == 0
}

func (b board) print(t *testing.T) {
	var row string
	for i := 0; i < 8; i++ {
		switch *b.Val[i] {
		case 0:
			row = strings.Repeat("_ ", 8)
		case 1:
			row = "󰡚 " + strings.Repeat("_ ", 7)
		case 8:
			row = strings.Repeat("_ ", 7) + "󰡚 "
		default:
			row = strings.Repeat("_ ", *b.Val[i]-1) + "󰡚 " + strings.Repeat("_ ", 8-*b.Val[i])
		}
		t.Log(row)
	}
	t.Logf("%d fixed\n", *b.Fixed)
}

func String(s string) *string {
	return &s
}

func Decimal(d int) *int {
	return &d
}
