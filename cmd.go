package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"syscall"
)

// Version information, populated by make
// Token count accumulator in case of CTRL-C
// Parameter map shared with tools
var (
	version    string
	golang     string
	githash    string
	tokenCount int32
	keyVals    ParamMap
)

const (
	siExt     = ".sprompt"
	pExt      = ".prompt"
	embModel  = "text-embedding-004"
	digestKey = "{digest}"
)

// Parameters holds gen flag values
type Parameters struct {
	ChatMode          bool
	Code              bool
	DigestPaths       ParamArray
	Embed             bool
	FilePaths         ParamArray
	GenModel          string
	Help              bool
	JSON              bool
	K                 int
	Lambda            float64
	OnlyKvs           bool
	Stdin             bool
	SystemInstruction bool
	TokenCount        bool
	Temp              float64
	Tool              bool
	TopP              float64
	Unsafe            bool
	Verbose           bool
	Version           bool
	WhiteboardMode    bool
	Args              []string
	Whiteboard        *Amanda
}

type Env struct {
	Card *bytes.Buffer
	Next *string
}

func main() {
	// Check for API key
	if val, ok := os.LookupEnv("GEMINI_API_KEY"); !ok || len(val) == 0 {
		fmt.Fprintf(os.Stderr, "Environment variable GEMINI_API_KEY not set!\n")
		os.Exit(1)
	}

	// Define parameter map
	keyVals = ParamMap{}

	// Define params
	params := &Parameters{}
	flag.BoolVar(&params.Verbose, "V", false, "output model details, system instructions and chat history")
	flag.BoolVar(&params.ChatMode, "c", false, "enter chat mode after content generation")
	flag.BoolVar(&params.Code, "code", false, "allow code execution (incompatible with -json and -tool)")
	flag.Var(&params.DigestPaths, "d", "path to a digest folder")
	flag.BoolVar(&params.Embed, "e", false, fmt.Sprintf("write embeddings to digest (default model \"%s\")", embModel))
	flag.Var(&params.FilePaths, "f", "file, directory or quoted matching pattern of files to attach")
	flag.BoolVar(&params.Help, "h", false, "show this help message and exit")
	// TODO add -i for image generation
	flag.BoolVar(&params.JSON, "json", false, "response in JavaScript Object Notation (incompatible with -tool and -code)")
	flag.IntVar(&params.K, "k", 3, "maximum number of entries from digest to retrieve")
	flag.Float64Var(&params.Lambda, "l", 0.5, "trade off accuracy for diversity when querying digests [0.0,1.0]")
	flag.StringVar(&params.GenModel, "m", "gemini-2.0-flash", "embedding or generative model name")
	flag.BoolVar(&params.OnlyKvs, "o", false, "only store metadata with embeddings and ignore the content")
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val")
	flag.BoolVar(&params.SystemInstruction, "s", false, "treat argument as system instruction")
	flag.BoolVar(&params.TokenCount, "t", false, "output total number of tokens")
	flag.Float64Var(&params.Temp, "temp", 1.0, "changes sampling during response generation [0.0,2.0]")
	flag.BoolVar(&params.Tool, "tool", false, "invoke one of the tools (incompatible with -json and -code)")
	flag.Float64Var(&params.TopP, "top_p", 0.95, "changes how the model selects tokens for generation [0.0,1.0]")
	flag.BoolVar(&params.Unsafe, "unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	flag.BoolVar(&params.Version, "v", false, "show version and exit")
	flag.BoolVar(&params.WhiteboardMode, "w", false, "enter whiteboard mode for content generation")
	flag.Parse()
	params.Args = flag.Args()
	params.Stdin = hasInputFromStdin(os.Stdin)

	// Handle help and version flags before any further processing
	if params.Help {
		emitUsage(os.Stdout)
		os.Exit(0)
	}
	if params.Version {
		fmt.Fprintf(os.Stdout, "gen version %s (%s %s)\n", version, golang, githash)
		os.Exit(0)
	}

	// Argument validation
	if !isParamsValid(params) {
		emitUsage(os.Stderr)
		os.Exit(1)
	}

	// TODO use context WithValue for retrieving values
	// TODO replace Background() with WithCancel()
	ctx := context.Background()

	// Handle token count as separate Go routine
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-done
		if tokenCount > 0 {
			fmt.Printf("\n\033[31m%d tokens\033[0m\n", tokenCount)
		}
		os.Exit(1)
	}()

	// handle multiple gen instances whiteboarding
	if params.WhiteboardMode {
		var text bytes.Buffer
		for i, arg := range params.Args {
			text.WriteString(arg)
			if i < len(params.Args)-1 {
				text.WriteString(" ")
			}
		}
		params.Whiteboard = TupleSpace()
		params.Whiteboard.Out(Env{
			Card: &text,
		})
		for _, thisPath := range params.FilePaths {
			thisParams := *params // deep copy, ignore DigestPaths
			thisParams.FilePaths = ParamArray{thisPath}
			thisParams.Args = []string{} // arguments are obtain via In()
			params.Whiteboard.Eval(amandaGen, ctx, os.Stdin, os.Stdout, &thisParams)
		}
		os.Exit(params.Whiteboard.SecondsTimeout(30))
	} else {
		os.Exit(emitGen(ctx, os.Stdin, os.Stdout, params))
	}
}

// Usage overrides PrintDefaults to provide custom usage information.
func emitUsage(out io.Writer) {
	fmt.Fprintln(out, "Usage: gen [options] <prompt>")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Command-line interface to Google Gemini large language models")
	fmt.Fprintln(out, "  Requires a valid GEMINI_API_KEY environment variable set.")
	fmt.Fprintln(out, "  Content is generated by a prompt and optional system instructions.")
	fmt.Fprintln(out, "  Use - to assign stdin as prompt argument or as attached file.")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Tools:")
	fmt.Fprintln(out, knownTools())
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Parameters:")
	fmt.Fprintf(out, "\n")
	flag.PrintDefaults()
}

func isParamsValid(params *Parameters) bool {
	// Handle invalid arguments/option combinations, starting with no embed flag,
	// prompt as stdin, argument or file
	if (!params.Embed &&
		!params.Stdin &&
		len(params.Args) == 0 && !oneMatches(params.FilePaths, pExt) &&
		!oneMatches(params.FilePaths, siExt)) ||
		// embeddings with chat, prompts, no files, no argument or generative settings
		(params.Embed && (params.WhiteboardMode || params.ChatMode || params.Unsafe ||
			params.Code || params.Tool || params.JSON ||
			len(params.DigestPaths) != 1 ||
			(params.OnlyKvs && len(keyVals) == 0) ||
			isFlagSet("temp") || isFlagSet("top_p") || isFlagSet("k") || isFlagSet("l") ||
			anyMatches(params.FilePaths, pExt) || anyMatches(params.FilePaths, siExt) ||
			(len(params.Args) == 0 && !oneMatches(params.FilePaths, "-")))) ||
		// whiteboard mode only with system instruction files, argument,
		// no system instruction flag, chat mode, json output or stdin
		(params.WhiteboardMode && (params.SystemInstruction ||
			params.JSON || params.ChatMode || params.Stdin ||
			!allMatch(params.FilePaths, siExt) || len(params.Args) == 0)) ||
		// simultaneous use of -code and -tool
		(params.Code && params.Tool) ||
		// tool registration with json output
		(params.Tool && params.JSON) ||
		// code execution with json output
		(params.Code && params.JSON) ||
		// invalid k values
		(params.K < 0 || params.K > 10) ||
		// invalid lambda values
		(params.Lambda < 0 || params.Lambda > 1) ||
		// invalid temperature values
		(params.Temp < 0 || params.Temp > 2) ||
		// invalid topP values
		(params.TopP < 0 || params.TopP > 1) ||
		// lack of /dev/tty on Windows prevents this flag combination
		(runtime.GOOS == "windows" && params.Stdin && params.ChatMode) ||
		// stdin set but neither used as file nor as argument
		(params.Stdin && !(len(params.Args) == 1 && params.Args[0] == "-") &&
			!oneMatches(params.FilePaths, "-")) ||
		// one of file or argument as system instruction - look for a prompt
		(params.SystemInstruction &&
			// no stdin, no argument
			((!params.Stdin && len(params.Args) == 0) ||
				// no stdin, argument as system instruction, no prompt as file, no chat mode
				(!params.Stdin && len(params.Args) > 0 && !anyMatches(params.FilePaths, pExt) && !params.ChatMode) ||
				// stdin as file, no prompt as file or argument
				(params.Stdin && oneMatches(params.FilePaths, "-") && len(params.Args) == 0 && !oneMatches(params.FilePaths, pExt)) ||
				// stdin as argument, no prompt as file
				(params.Stdin && len(params.Args) == 1 && params.Args[0] == "-" && !oneMatches(params.FilePaths, pExt) && !params.ChatMode))) {
		return false
	}
	return true
}
