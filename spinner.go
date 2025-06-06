package main

import (
	"fmt"
	"io"
	"os"
	"sync/atomic"
	"time"
)

// ClearLine go to the beginning of the line and clear it
const ClearLine = "\r\033[K"

// Spinner main type
type Spinner struct {
	frames []rune
	pos    int
	active uint64
	text   string
	tpf    time.Duration
	writer io.Writer
}

// Option describes an option to override a default
// when creating a new Spinner.
type SpinnerOption func(s *Spinner)

// New creates a Spinner object with the provided
// text. By default, the line frames are used, and
// new frames are rendered every 100 milliseconds.
func NewSpinner(text string, opts ...SpinnerOption) *Spinner {
	s := &Spinner{
		text:   ClearLine + text,
		frames: []rune(`|/-\`),
		tpf:    100 * time.Millisecond,
		writer: os.Stdout,
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

// WithFrames sets the frames string.
func WithFrames(frames string) SpinnerOption {
	return func(s *Spinner) {
		s.Set(frames)
	}
}

// Set frames to the given string which must not use spaces.
func (s *Spinner) Set(frames string) {
	s.frames = []rune(frames)
}

// Start shows the spinner.
func (s *Spinner) Start() *Spinner {
	if atomic.LoadUint64(&s.active) > 0 {
		return s
	}
	atomic.StoreUint64(&s.active, 1)
	go func() {
		for atomic.LoadUint64(&s.active) > 0 {
			fmt.Fprintf(s.writer, s.text, s.next())
			time.Sleep(s.tpf)
		}
	}()
	return s
}

// Stop hides the spinner.
func (s *Spinner) Stop() bool {
	if x := atomic.SwapUint64(&s.active, 0); x > 0 {
		fmt.Fprint(s.writer, ClearLine)
		return true
	}
	return false
}

func (s *Spinner) next() string {
	r := s.frames[s.pos%len(s.frames)]
	s.pos++
	return string(r)
}
