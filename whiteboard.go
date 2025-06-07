package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// amandaGen is a wrapper for emitGen
// It uses tuples on the whiteboard to set arguments for the various prompts
// One tuple holds content, the other holds a value used by the sentinel to terminate
// FIXME experimental
// TODO break case statement into separate functions; explore more cases
func amandaGen(ctx context.Context, in io.Reader, out io.Writer, params *Parameters) int {
	var this string
	// determine current role from prompt filename
	_, file := filepath.Split(params.FilePaths[0])
	prompt := strings.TrimSuffix(file, SPExt)
	s := NewSpinner("%s "+prompt, WithWriter(os.Stderr), WithFrames(`⣾⣽⣻⢿⡿⣟⣯⣷`))
	s.Start()
	defer s.Stop()
	for {
		// fresh Env at each iteration
		var e Env
		var buf bytes.Buffer
		switch prompt {
		case "tool":
			// pick any tuple
			Whiteboard.Rd(&e)
			// ignore if it's a sentinel tuple
			if e.Receiver != nil && *e.Receiver == "sentinel" {
				continue
			}
			// set args from tuple data
			params.Args = []string{e.Args.String()}
			params.Tool = true
			// invoke tool and capture gen out
			res := emitGen(ctx, in, &buf, params)
			if res != 0 {
				return res
			}
			// create tuple for sentinel
			rcv := "sentinel"
			Whiteboard.Out(Env{
				Args:     &buf,
				Receiver: &rcv,
			})
		case "sentinel":
			if this == "" {
				this = prompt
			}
			e.Receiver = &this
			Whiteboard.In(&e)
			// set args from tuple data
			params.Args = []string{e.Args.String()}
			// capture gen out in buffer
			res := emitGen(ctx, in, &buf, params)
			if res != 0 {
				return res
			}
			if rcv := strings.ToLower(strings.TrimSpace(buf.String())); rcv != "continue" {
				var final Env
				final.Receiver = &rcv
				Whiteboard.Rd(&final)
				fmt.Fprintf(out, ShowCursor+ClearLine+"\033[97m%s\033[0m\n", final.Args.String())
				return 0
			}
		case "code":
			// TODO
			continue
		default:
			// fetch tuple for this receiver; any on first iteration
			if this != "" {
				e.Receiver = &this
			}
			Whiteboard.In(&e)
			// set args from tuple data
			params.Args = []string{e.Args.String()}
			// capture gen out in buffer
			res := emitGen(ctx, in, &buf, params)
			if res != 0 {
				return res
			}
			// last line decides new tuple receiver
			next := lastWord(&buf)
			// put result back on whiteboard
			if next != "" {
				Whiteboard.Out(Env{
					Args:     &buf,
					Receiver: &next,
				})
			} else {
				Whiteboard.Out(Env{
					Args: &buf,
					// Receiver nil i.e. any
				})
			}
			// for following iterations, request tuples for this receiver
			if this == "" {
				this = prompt
			}
		}
	}
}
