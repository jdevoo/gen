package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
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
)

// gen constants
const (
	SPExt     = ".sprompt" // system prompt extension
	PExt      = ".prompt"  // regular prompt extension
	DigestKey = "{digest}" // key to replace with embedded content
	DotGen    = ".gen"     // name of chat history file
)

// Parameters holds gen flag values
type Parameters struct {
	Args              []string
	ChatMode          bool
	CodeGen           bool
	DigestPaths       ParamArray
	Embed             bool
	EmbModel          string
	FilePaths         ParamArray
	GenModel          string
	GoogleSearch      bool
	Help              bool
	ImgModality       bool
	JSON              bool
	K                 int
	Lambda            float64
	OnlyKvs           bool
	Interactive       bool
	SystemInstruction bool
	TokenCount        bool
	Temp              float64
	Tool              bool
	TopP              float64
	Unsafe            bool
	Verbose           bool
	Version           bool
}

func main() {
	// Define parameter map for variable substitutions in prompts
	keyVals = ParamMap{}

	// Define gen parameters
	params := &Parameters{
		EmbModel: "text-embedding-004",
	}
	flag.BoolVar(&params.Verbose, "V", false, "output model details, system instructions and chat history")
	flag.BoolVar(&params.ChatMode, "c", false, "enter chat mode after content generation (incompatible with -json, -img, -code or -g)")
	flag.BoolVar(&params.CodeGen, "code", false, "code execution tool (incompatible with -g, -json, -img or -tool)")
	flag.Var(&params.DigestPaths, "d", "path to a digest folder")
	flag.BoolVar(&params.Embed, "e", false, fmt.Sprintf("write embeddings to digest (default model \"%s\")", params.EmbModel))
	flag.Var(&params.FilePaths, "f", "file, directory or quoted matching pattern of files to attach")
	flag.BoolVar(&params.GoogleSearch, "g", false, "Google search tool (incompatible with -code, -json, -img and -tool)")
	flag.BoolVar(&params.Help, "h", false, "show this help message and exit")
	flag.BoolVar(&params.ImgModality, "img", false, "generate a jpeg image (use -m with a supported model)")
	flag.BoolVar(&params.JSON, "json", false, "response in JavaScript Object Notation (incompatible with -g, -code, -img and -tool)")
	flag.IntVar(&params.K, "k", 3, "maximum number of entries from digest to retrieve")
	flag.Float64Var(&params.Lambda, "l", 0.5, "trade off accuracy for diversity when querying digests [0.0,1.0]")
	flag.StringVar(&params.GenModel, "m", "gemini-2.0-flash", "embedding or generative model name")
	flag.BoolVar(&params.OnlyKvs, "o", false, "only store metadata with embeddings and ignore the content")
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val")
	flag.BoolVar(&params.SystemInstruction, "s", false, "treat prompt as system instruction")
	flag.BoolVar(&params.TokenCount, "t", false, "output total number of tokens")
	flag.Float64Var(&params.Temp, "temp", 1.0, "changes sampling during response generation [0.0,2.0]")
	flag.BoolVar(&params.Tool, "tool", false, "invoke one of the tools (incompatible with -s, -g, -json, -img or -code)")
	flag.Float64Var(&params.TopP, "top_p", 0.95, "changes how the model selects tokens for generation [0.0,1.0]")
	flag.BoolVar(&params.Unsafe, "unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	flag.BoolVar(&params.Version, "v", false, "show version and exit")
	flag.Parse()
	params.Args = flag.Args()
	params.Interactive = hasInteractiveInput(os.Stdin)

	// Handle help and version flags before any further processing
	if params.Help {
		emitUsage(os.Stdout)
		os.Exit(0)
	}

	// Handle version option
	if params.Version {
		var m debug.Module
		if binfo, ok := debug.ReadBuildInfo(); ok {
			for _, dep := range binfo.Deps {
				if dep.Path == "google.golang.org/genai" {
					m = *dep
					break
				}
			}
		}
		fmt.Fprintf(os.Stdout, "gen %s (%s sdk %s %s)\n", version, githash, m.Version, golang)
		os.Exit(0)
	}

	// Argument validation
	if !isParamsValid(params) {
		emitUsage(os.Stderr)
		os.Exit(1)
	}

	// Look for API key staring with VertexAI followed by Google AI Studio
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

	// Create a root context
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
	fmt.Fprintln(out, "  Use - to assign stdin as prompt or as attached file.")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Tools:")
	fmt.Fprintln(out, knownTools())
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Parameters:")
	fmt.Fprintf(out, "\n")
	flag.PrintDefaults()
}

// isParamsValid looks for invalid params combinations and/or values.
func isParamsValid(params *Parameters) bool {
	if
	// invalid k values
	(params.K < 0 || params.K > 10) ||

		// invalid lambda values
		(params.Lambda < 0 || params.Lambda > 1) ||

		// invalid temperature values
		(params.Temp < 0 || params.Temp > 2) ||

		// invalid topP values
		(params.TopP < 0 || params.TopP > 1) ||

		// code execution with incompatible flags
		(params.CodeGen &&
			(params.JSON || params.Tool || params.GoogleSearch)) ||

		// tool registration with incompatible flags
		(params.Tool &&
			(params.JSON || params.CodeGen || params.GoogleSearch || params.SystemInstruction)) ||

		// search with incompatible flags
		(params.GoogleSearch &&
			(params.JSON || params.Tool || params.CodeGen)) ||

		// image modality with incompatible flags
		(params.ImgModality &&
			(params.GoogleSearch || params.CodeGen || params.Tool || params.JSON || params.ChatMode)) ||

		// chat with incompatible flags
		(params.ChatMode &&
			(params.JSON || params.GoogleSearch || params.CodeGen)) ||

		// embeddings
		(params.Embed &&
			// incompatible flags
			(params.ChatMode || params.Unsafe || params.CodeGen || params.Tool ||
				params.JSON || params.ImgModality || params.GoogleSearch ||
				isFlagSet("temp") || isFlagSet("top_p") || isFlagSet("k") || isFlagSet("l") ||
				// no digest set
				len(params.DigestPaths) != 1 ||
				// metadata missing
				(params.OnlyKvs && len(keyVals) == 0) ||
				// prompts set
				anyMatches(params.FilePaths, PExt) || anyMatches(params.FilePaths, SPExt) ||
				// no arguments or files to digest
				(len(params.Args) == 0 && !oneMatches(params.FilePaths, "-")))) ||

		// redirected or piped stdin
		(!params.Interactive &&
			// not set as file
			(!oneMatches(params.FilePaths, "-") ||
				// not set as argument
				(len(params.Args) == 1 && params.Args[0] != "-") ||
				// no embed flag, no regular prompt, argument or file
				(!params.Embed && len(params.Args) == 0 && !oneMatches(params.FilePaths, PExt)))) ||

		// one of file or argument as system instruction - looking for a prompt
		(params.SystemInstruction &&
			// no redirect, no argument
			((params.Interactive && len(params.Args) == 0) ||
				// no redirect, argument as system instruction, no prompt as file, no chat mode
				(params.Interactive && len(params.Args) > 0 && !anyMatches(params.FilePaths, PExt) && !params.ChatMode) ||
				// redirect as file, but no prompt as file or argument
				(!params.Interactive && oneMatches(params.FilePaths, "-") && len(params.Args) == 0 && !oneMatches(params.FilePaths, PExt)) ||
				// redirect as argument, no prompt as file
				(!params.Interactive && len(params.Args) == 1 && params.Args[0] == "-" && !oneMatches(params.FilePaths, PExt) && !params.ChatMode))) {

		return false
	}

	return true
}
