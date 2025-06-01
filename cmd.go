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
	"runtime/debug"
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
	Whiteboard *Amanda
)

const (
	siExt     = ".sprompt"
	pExt      = ".prompt"
	digestKey = "{digest}"
	dotGen    = ".gen"
)

// Parameters holds gen flag values
type Parameters struct {
	ChatMode          bool
	Code              bool
	DigestPaths       ParamArray
	Embed             bool
	EmbModel          string
	FilePaths         ParamArray
	GenModel          string
	Help              bool
	ImgModality       bool
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
}

type Env struct {
	Args     *bytes.Buffer
	Receiver *string
}

func main() {
	// Define parameter map
	keyVals = ParamMap{}

	// Define params
	params := &Parameters{
		EmbModel: "text-embedding-004",
	}
	flag.BoolVar(&params.Verbose, "V", false, "output model details, system instructions and chat history")
	flag.BoolVar(&params.ChatMode, "c", false, "enter chat mode after content generation (incompatible with -img)")
	flag.BoolVar(&params.Code, "code", false, "allow code execution (incompatible with -img, -json and -tool)")
	flag.Var(&params.DigestPaths, "d", "path to a digest folder")
	flag.BoolVar(&params.Embed, "e", false, fmt.Sprintf("write embeddings to digest (default model \"%s\")", params.EmbModel))
	flag.Var(&params.FilePaths, "f", "file, directory or quoted matching pattern of files to attach")
	flag.BoolVar(&params.Help, "h", false, "show this help message and exit")
	flag.BoolVar(&params.ImgModality, "img", false, "generate an image instead of text")
	flag.BoolVar(&params.JSON, "json", false, "response in JavaScript Object Notation (incompatible with -img, -tool and -code)")
	flag.IntVar(&params.K, "k", 3, "maximum number of entries from digest to retrieve")
	flag.Float64Var(&params.Lambda, "l", 0.5, "trade off accuracy for diversity when querying digests [0.0,1.0]")
	flag.StringVar(&params.GenModel, "m", "gemini-2.0-flash", "embedding or generative model name")
	flag.BoolVar(&params.OnlyKvs, "o", false, "only store metadata with embeddings and ignore the content")
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val")
	flag.BoolVar(&params.SystemInstruction, "s", false, "treat argument as system instruction")
	flag.BoolVar(&params.TokenCount, "t", false, "output total number of tokens")
	flag.Float64Var(&params.Temp, "temp", 1.0, "changes sampling during response generation [0.0,2.0]")
	flag.BoolVar(&params.Tool, "tool", false, "invoke one of the tools (incompatible with -img, -json and -code)")
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
		var m debug.Module
		binfo, ok := debug.ReadBuildInfo()
		if ok {
			for _, dep := range binfo.Deps {
				if dep.Path == "google.golang.org/genai" {
					m = *dep
					break
				}
			}
		}
		fmt.Fprintf(os.Stdout, "gen version %s (%s %s sdk %s)\n", version, golang, githash, m.Version)
		os.Exit(0)
	}

	// Argument validation
	if !isParamsValid(params) {
		emitUsage(os.Stderr)
		os.Exit(1)
	}

	// Look for API key
	if val, ok := os.LookupEnv("GOOGLE_CLOUD_PROJECT"); !ok || len(val) == 0 {
		if val, ok := os.LookupEnv("GOOGLE_API_KEY"); !ok || len(val) == 0 {
			fmt.Fprintf(os.Stderr, "Environment variable GOOGLE_API_KEY not set!\n")
			os.Exit(1)
		}
	}
	// if VertexAI project ID, then look for cloud location
	if val, ok := os.LookupEnv("GOOGLE_CLOUD_PROJECT"); ok && len(val) != 0 {
		if val, ok := os.LookupEnv("GOOGLE_CLOUD_LOCATION"); !ok || len(val) == 0 {
			fmt.Fprintf(os.Stderr, "Environment variable GOOGLE_CLOUD_LOCATION not set!\n")
			os.Exit(1)
		}
		// if both backends are possible, chose which one is to be used
		if val, ok := os.LookupEnv("GOOGLE_API_KEY"); ok && len(val) != 0 {
			if val, ok := os.LookupEnv("GOOGLE_GENAI_USE_VERTEXAI"); !ok || len(val) == 0 {
				fmt.Fprintf(os.Stderr, "Environment variable GOOGLE_GENAI_USE_VERTEXAI not set!\n")
				os.Exit(1)
			}
		}
	}

	// TODO replace Background() with WithCancel()
	ctx := context.Background()

	// Handle token count in case of CTRL-C
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-done
		if params.TokenCount {
			fmt.Printf("\n\033[31m%d tokens\033[0m\n", tokenCount)
		}
		os.Exit(1)
	}()

	// Handle multiple gen instances
	if params.WhiteboardMode {
		var text bytes.Buffer
		for i, arg := range params.Args {
			text.WriteString(arg)
			if i < len(params.Args)-1 {
				text.WriteString(" ")
			}
		}
		Whiteboard = TupleSpace()
		// Place args in tuple on the whiteboard
		Whiteboard.Out(Env{
			Args: &text,
			// Next nil i.e. anyone can pick it up
			// Done false
		})
		// Start one gen instance per system prompt file
		for _, thisPath := range params.FilePaths {
			thisParams := *params        // deep copy ignoring DigestPaths and FilePaths
			thisParams.Args = []string{} // arguments are obtained via Amanda's In()
			thisParams.FilePaths = ParamArray{thisPath}
			Whiteboard.Eval(amandaGen, ctx, os.Stdin, os.Stdout, &thisParams)
		}
		// FIXME hardcoded timeout
		os.Exit(Whiteboard.StartWithSecondsTimeout(30))
	}
	// Handle regular gen usage
	os.Exit(emitGen(ctx, os.Stdin, os.Stdout, params))
}

// Usage overrides PrintDefaults to provide custom usage information.
func emitUsage(out io.Writer) {
	fmt.Fprintln(out, "Usage: gen [options] <prompt>")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Command-line interface to Google Gemini large language models")
	fmt.Fprintln(out, "  Requires a valid GOOGLE_API_KEY environment variable set.")
	fmt.Fprintln(out, "  Also supports VertexAI with valid GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
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

// TODO check -img with other flags
func isParamsValid(params *Parameters) bool {
	// Handle invalid arguments/option combinations
	// starting with no embed flag and no prompt as stdin, argument or file
	if (!params.Embed &&
		!params.Stdin &&
		len(params.Args) == 0 && !oneMatches(params.FilePaths, pExt) &&
		!oneMatches(params.FilePaths, siExt)) ||
		// embeddings with chat, prompts, no files, no argument or generative settings
		(params.Embed && (params.WhiteboardMode || params.ChatMode || params.Unsafe ||
			params.Code || params.Tool || params.JSON || params.ImgModality ||
			len(params.DigestPaths) != 1 ||
			(params.OnlyKvs && len(keyVals) == 0) ||
			isFlagSet("temp") || isFlagSet("top_p") || isFlagSet("k") || isFlagSet("l") ||
			anyMatches(params.FilePaths, pExt) || anyMatches(params.FilePaths, siExt) ||
			(len(params.Args) == 0 && !oneMatches(params.FilePaths, "-")))) ||
		// whiteboard mode only with system instruction files, argument,
		// no system instruction flag, chat mode, image modality, json output or stdin
		(params.WhiteboardMode && (params.SystemInstruction ||
			params.JSON || params.ChatMode || params.ImgModality || params.Stdin ||
			!allMatch(params.FilePaths, siExt) || len(params.Args) == 0)) ||
		// tool registration with code execution
		(params.Code && params.Tool) ||
		// json output with tool registration
		(params.Tool && params.JSON) ||
		// json output with code execution
		(params.Code && params.JSON) ||
		// image modality with tool registration, json output, code execution or chat mode
		(params.ImgModality && (params.Code || params.Tool || params.JSON || params.ChatMode)) ||
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
		// one of file or argument as system instruction - looking for a prompt
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
