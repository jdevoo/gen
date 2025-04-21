package main

import (
	"math/rand/v2"
	"time"
)

// Amanda is a tuple space that holds the communication channels
// that allow getting and putting tuples into the space.
// It appears as a shared associative memory that supports
// four operations: Out, In, Rd, and Eval.

type Amanda struct {
	Input   <-chan interface{}
	Output  chan<- interface{}
	RNG     *rand.Rand
	Timeout <-chan time.Time
	Done    chan struct{}
}

// Tuple shoud be a flat structure composed of Go types
type Tuple interface{}

type Src int64

func (s Src) Uint64() uint64 {
	return uint64(s)
}

// Amanda Tuple Space Limitations
//
// No Error Handling in `In` and `Rd`
// If the tuple space is closed (the `output` channel is closed), the `In` and `Rd` functions
// will infinitely loop and consume CPU resources without returning.
// This can lead to a deadlock or other issues.
//
// Potential Deadlock in `Eval`
// If the `Eval` function's executed goroutine produces a large number of results that
// fill the `output` channel, it could potentially block indefinitely waiting for the `input`
// channel to have space, leading to a deadlock. (less likely due to output buffer size but
// theoretically possible)
//
// Performance
// The heavy reliance on reflection can impact performance, especially when dealing with
// large tuples or frequent operations.
//
// Tuple Ordering
// The code provides no explicit ordering of tuples within the tuple space.
// Tuple extraction relies on matching, so the order in which tuples are added and extracted
// may not be deterministic. This might be a desired behavior for some use cases, but it
// should be documented or configurable if needed.
//
// Limited Matching Capabilities
// The `match` function performs exact matching (or nil wildcard matching).
// More complex matching, such as regular expressions or range-based matching, is not supported.
//
// Lack of Context Support
// Operations lack context support. It would be beneficial to have context aware functions.
//
// Dropped Tuple
// The `TupleSpace` constructor drops tuples instead of blocking.
// If blocking is the desired behaviour, this should be changed.
func TupleSpace() *Amanda {
	input := make(chan interface{}, 10)
	output := make(chan interface{}, 10)
	rng := rand.New(Src(time.Now().UnixNano()))
	ts := &Amanda{
		Input:   input,
		Output:  output,
		RNG:     rng,
		Timeout: nil,
		Done:    make(chan struct{}),
	}
	go func() {
		for t := range output {
			select {
			case input <- t:
			default:
				// drop tuple if the input channel is full
			}
		}
		close(input)
	}()
	return ts
}
