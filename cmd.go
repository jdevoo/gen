package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
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
	Tool              bool
	TopP              float64
	Unsafe            bool
	Verbose           bool
	Version           bool
	Walk              bool // used with FilePaths
}

// main gen entry point.
// Parameters are stored as `ParamMap` and passed as context values for tool stete injection.
func main() {
	// Define parameter map for variable substitutions in prompts
	keyVals := ParamMap{}

	// Define gen parameters
	params := &Parameters{
		K:           3,
		Lambda:      0.5,
		Temp:        1.0,
		TopP:        0.95,
		EmbModel:    "gemini-embedding-001",
		GenModel:    "gemini-2.5-flash",
		DigestPaths: ParamArray{},
		MCPServers:  ParamArray{},
		MCPSessions: SessionArray{},
	}

	if err := loadPrefs(params); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading %s: %v\n", DotGenRc, err)
		os.Exit(1)
	}

	flag.BoolVar(&params.Verbose, "V", false, "output model details, system instructions and chat history")
	flag.BoolVar(&params.ChatMode, "c", false, "enter chat mode after content generation (incompatible with -json, -img, -code or -g)")
	flag.BoolVar(&params.CodeGen, "code", false, "code execution tool (incompatible with -g, -json, -img or -tool)")
	flag.Var(&params.DigestPaths, "d", "path to a digest folder")
	flag.BoolVar(&params.Embed, "e", false, fmt.Sprintf("write embeddings to digest (default model \"%s\")", params.EmbModel))
	flag.Var(&params.FilePaths, "f", "file, directory or quoted pattern of files to attach")
	flag.BoolVar(&params.GoogleSearch, "g", false, "Google search tool (incompatible with -code, -json, -img and -tool)")
	flag.BoolVar(&params.Help, "h", false, "show this help message and exit")
	flag.BoolVar(&params.ImgModality, "img", false, "generate a jpeg image (use -m to set a supported model)")
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
	flag.BoolVar(&params.Tool, "tool", false, "invoke one of the tools (incompatible with -s, -g, -json, -img or -code)")
	flag.Float64Var(&params.TopP, "top_p", params.TopP, "changes how the model selects tokens for generation [0.0,1.0]")
	flag.BoolVar(&params.Unsafe, "unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	flag.BoolVar(&params.Version, "v", false, "show version and exit")
	flag.BoolVar(&params.Walk, "w", false, "process directories delcared with -f recursively")
	flag.Parse()
	params.Args = flag.Args()
	params.Interactive = isInteractive(os.Stdin)

	// Create the root context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle version option
	if params.Version {
		var genaiModule debug.Module
		var mcpModule debug.Module
		if binfo, ok := debug.ReadBuildInfo(); ok {
			for _, dep := range binfo.Deps {
				if dep.Path == "google.golang.org/genai" {
					genaiModule = *dep
					continue
				}
				if dep.Path == "github.com/modelcontextprotocol/go-sdk" {
					mcpModule = *dep
					continue
				}
			}
		}
		fmt.Fprintf(os.Stdout, "gen %s (%s sdk %s mcp %s)\n",
			Version, Githash, genaiModule.Version, mcpModule.Version)
		os.Exit(0)
	}

	// store keyVals and params in context
	ctx = context.WithValue(ctx, "keyVals", keyVals)
	ctx = context.WithValue(ctx, "params", params)

	// stash MCP client sessions in params.MCPSessions
	for _, s := range params.MCPServers {
		cmd := exec.Command(s)
		name := filepath.Base(os.Args[0])
		client := mcp.NewClient(
			&mcp.Implementation{Name: name, Version: Version},
			&mcp.ClientOptions{
				CreateMessageHandler:  genSampling,
				ElicitationHandler:    genElicitation,
				LoggingMessageHandler: genLoggingHandler,
			},
		)
		mcpCtx, mcpCancel := context.WithTimeout(ctx, 30*time.Second)
		defer mcpCancel()
		session, err := client.Connect(mcpCtx, &mcp.CommandTransport{Command: cmd}, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "MCP client connect error: %v\n", err)
			os.Exit(1)
		}
		params.MCPSessions = append(params.MCPSessions, session)
	}

	// close any stashed MCP sessions before exit
	defer func() {
		for _, sess := range params.MCPSessions {
			sess.Close()
		}
		if params.TokenCount && ctx.Err() == nil {
			fmt.Printf("\n\033[31m%d tokens\033[0m\n", TokenCount.Load())
		}
	}()

	// Handle help and version flags before any further processing
	// requires context with params to list known tools
	if params.Help {
		emitUsage(ctx, os.Stdout)
		os.Exit(0)
	}

	// Argument validation
	// requires context with params to list known tools
	if isParamsInvalid(params, keyVals) {
		emitUsage(ctx, os.Stderr)
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

	// handle CTRL-C
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-done
		if params.TokenCount {
			if count := TokenCount.Load(); count > 0 {
				fmt.Printf("\n\033[31m%d tokens at CTRL-C\033[0m\n", count)
			}
		}
		cancel()
		os.Exit(1)
	}()

	os.Exit(emitGen(ctx, os.Stdin, os.Stdout))
}

// Usage overrides PrintDefaults to provide custom usage information.
func emitUsage(ctx context.Context, out io.Writer) {
	fmt.Fprintln(out, "Usage: gen [options] <prompt>")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Command-line interface to Google Gemini large language models")
	fmt.Fprintln(out, "  Requires a valid GOOGLE_API_KEY environment variable set.")
	fmt.Fprintln(out, "  Also supports VertexAI with valid GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
	fmt.Fprintln(out, "  Content is generated by a prompt and optional system instructions.")
	fmt.Fprintln(out, "  Use - to assign stdin as prompt or as attached file.")
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Tools:")
	if res, err := knownTools(ctx); err == nil {
		fmt.Fprintln(out, res)
	}
	fmt.Fprintf(out, "\n")
	fmt.Fprintln(out, "Parameters:")
	fmt.Fprintf(out, "\n")
	flag.PrintDefaults()
}
