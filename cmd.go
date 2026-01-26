package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime/debug"
	"sync/atomic"
	"syscall"
	"time"
)

// Version, Golang and Githash are populated by make
// Token count accumulator in case of CTRL-C
var (
	Version    string
	Githash    string
	TokenCount atomic.Int32
)

// gen constants
const (
	SPExt     = ".sprompt" // system prompt extension
	PExt      = ".prompt"  // regular prompt extension
	DigestKey = "{digest}" // key to replace with embedded content
	DotGen    = ".gen"     // name of chat history file
	DotGenRc  = ".genrc"   // name of preferences file
)

// Parameters holds gen flag values as well as Args and MCP sessions.
type Parameters struct {
	Args              []string // non-flag command-line arguments i.e. prompt
	ChatMode          bool
	CodeGen           bool
	DigestPaths       ParamArray // RAG
	Embed             bool       // RAG
	EmbModel          string
	FilePaths         ParamArray
	GenModel          string
	GoogleSearch      bool
	Help              bool
	ImgModality       bool
	JSON              bool
	K                 int
	Lambda            float64
	MCPServers        ParamArray
	MCPSessions       SessionArray
	OnlyKvs           bool // RAG
	Interactive       bool // terminal session?
	SystemInstruction bool
	TokenCount        bool
	Temp              float64
	Timeout           time.Duration
	Tool              bool
	TopP              float64
	Unsafe            bool
	Verbose           bool
	Version           bool
	Walk              bool // used with FilePaths
}

// main gen entry point.
// Parameters are stored as `ParamMap` and passed as context values for tool state injection.
func main() {
	params, keyVals := parseFlags()

	// Create the root context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle version option
	if params.Version {
		printVersion()
		os.Exit(0)
	}

	// store keyVals and params in context
	ctx = context.WithValue(ctx, "keyVals", keyVals)
	ctx = context.WithValue(ctx, "params", params)

	// stash MCP client sessions in params.MCPSessionsa
	if err := initMCPSessions(ctx, params); err != nil {
		fmt.Fprintf(os.Stderr, "MCP Error: %v\n", err)
		os.Exit(1)
	}
	defer genCleanup(params)

	// Handle help and version flags before any further processing
	if params.Help {
		emitUsage(ctx, os.Stdout) // context includes params with list to known tools
		os.Exit(0)
	}

	// Argument validation
	if isParamsInvalid(params, keyVals) {
		emitUsage(ctx, os.Stderr) // context includes params with list to known tools
		os.Exit(1)
	}

	if err := validateEnv(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// handle CTRL-C
	setupSignalHandler(ctx, cancel, params)

	os.Exit(emitGen(ctx, os.Stdin, os.Stdout))
}

// parseFlags handles flag definitions and parameter map for variable substitutions in prompts.
func parseFlags() (*Parameters, ParamMap) {
	keyVals := ParamMap{}

	// default parameter values
	params := &Parameters{
		K: 3, Lambda: 0.5, Temp: 1.0, TopP: 0.95, Timeout: 300 * time.Second,
		EmbModel: "gemini-embedding-001", GenModel: "gemini-2.5-flash",
	}

	if err := loadPrefs(params); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: %s: %v\n", DotGenRc, err)
	}

	flag.BoolVar(&params.Verbose, "V", false, "output model details, system instructions, chat history and thoughts")
	flag.BoolVar(&params.ChatMode, "c", false, "enter chat mode after content generation (incompatible with -json, -img, -code or -g)")
	flag.BoolVar(&params.CodeGen, "code", false, "code execution tool (incompatible with -g, -json, -img or -tool)")
	flag.Var(&params.DigestPaths, "d", "path to a digest folder")
	flag.BoolVar(&params.Embed, "e", false, fmt.Sprintf("write text embeddings to digest (default model \"%s\")", params.EmbModel))
	flag.Var(&params.FilePaths, "f", "file, directory or quoted pattern of files to attach")
	flag.BoolVar(&params.GoogleSearch, "g", false, "Google search tool (incompatible with -code, -json, -img and -tool)")
	flag.BoolVar(&params.Help, "h", false, "show this help message and exit")
	flag.BoolVar(&params.ImgModality, "img", false, "generate jpeg images (use -m to set a supported model)")
	flag.BoolVar(&params.JSON, "json", false, "response in JavaScript Object Notation (incompatible with -g, -code, -img and -tool)")
	flag.IntVar(&params.K, "k", params.K, "maximum number of entries from digest to retrieve")
	flag.Float64Var(&params.Lambda, "l", params.Lambda, "trade off accuracy for diversity when querying digests [0.0,1.0]")
	flag.StringVar(&params.GenModel, "m", params.GenModel, "embedding or generative model name")
	flag.Var(&params.MCPServers, "mcp", "mcp stdio server command")
	flag.BoolVar(&params.OnlyKvs, "o", false, "only store metadata with embeddings and ignore the content")
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val")
	flag.BoolVar(&params.SystemInstruction, "s", false, "treat argument as system prompt")
	flag.BoolVar(&params.TokenCount, "t", false, "output total number of tokens")
	flag.Float64Var(&params.Temp, "temp", params.Temp, "changes sampling during response generation [0.0,2.0]")
	flag.DurationVar(&params.Timeout, "to", params.Timeout, "timeout value in milliseconds")
	flag.BoolVar(&params.Tool, "tool", false, "invoke one of the tools (incompatible with -s, -g, -json, -img or -code)")
	flag.Float64Var(&params.TopP, "top_p", params.TopP, "changes how the model selects tokens for generation [0.0,1.0]")
	flag.BoolVar(&params.Unsafe, "unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	flag.BoolVar(&params.Version, "v", false, "show version and exit")
	flag.BoolVar(&params.Walk, "w", false, "process directories declared with -f recursively")
	flag.Parse()

	params.Args = flag.Args()
	params.Interactive = !isRedirected(os.Stdin)

	return params, keyVals
}

func printVersion() {
	var genaiVer, mcpVer string
	if binfo, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range binfo.Deps {
			switch dep.Path {
			case "google.golang.org/genai":
				genaiVer = dep.Version
			case "github.com/modelcontextprotocol/go-sdk":
				mcpVer = dep.Version
			}
		}
	}
	fmt.Printf("gen %s (%s sdk %s mcp %s)\n", Version, Githash, genaiVer, mcpVer)
}

// emitUsage overrides PrintDefaults to provide custom usage information.
func emitUsage(ctx context.Context, out io.Writer) {
	tools, err := knownTools(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "MCP server error: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintln(out, "Usage: gen [options] <prompt>")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Command-line interface to Google Gemini large language models")
	fmt.Fprintln(out, "  Requires a valid GOOGLE_API_KEY environment variable set.")
	fmt.Fprintln(out, "  Also supports VertexAI with valid GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
	fmt.Fprintln(out, "  Content is generated by a prompt and optional system instructions.")
	fmt.Fprintln(out, "  Use - to assign stdin as prompt or as attached file.")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, fmt.Sprintf("Tools:\n%s", tools))
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Parameters:")
	fmt.Fprintf(out, "\n")
	flag.PrintDefaults()
}

// validateEnv checks for required Google Cloud/AI Studio credentials.
func validateEnv() error {
	hasCloudProject := os.Getenv("GOOGLE_CLOUD_PROJECT") != ""
	hasAPIKey := os.Getenv("GOOGLE_API_KEY") != ""

	if !hasCloudProject && !hasAPIKey {
		return fmt.Errorf("neither GOOGLE_CLOUD_PROJECT nor GOOGLE_API_KEY is set")
	}

	if hasCloudProject {
		if os.Getenv("GOOGLE_CLOUD_LOCATION") == "" {
			return fmt.Errorf("GOOGLE_CLOUD_LOCATION must be set when using GOOGLE_CLOUD_PROJECT")
		}
		if hasAPIKey && os.Getenv("GOOGLE_GENAI_USE_VERTEXAI") == "" {
			return fmt.Errorf("set GOOGLE_GENAI_USE_VERTEXAI to 'true' or 'false' when both API Key and Project ID are present")
		}
	}
	return nil
}

func genCleanup(params *Parameters) {
	for _, sess := range params.MCPSessions {
		sess.Close()
	}
	// Final token count report
	if params.TokenCount {
		fmt.Printf("\n\033[31m%d tokens\033[0m\n", TokenCount.Load())
	}
}

func setupSignalHandler(ctx context.Context, cancel context.CancelFunc, params *Parameters) {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		if params.TokenCount {
			if count := TokenCount.Load(); count > 0 {
				fmt.Printf("\n\033[31m%d tokens at CTRL-C\033[0m\n", count)
			}
		}
		cancel()
		os.Exit(1)
	}()
}
