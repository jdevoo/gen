package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"google.golang.org/genai"
)

// Version and Githash are populated by make
var (
	Version    string
	Githash    string
	TokenCount atomic.Int32
)

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
	Segment           bool // SegmentForeground by default
	SegmentBackground bool
	SegModel          string
	SystemInstruction bool
	TokenCount        bool
	Temp              float64
	ThinkingLevel     genai.ThinkingLevel
	Timeout           time.Duration
	Tool              bool
	ToolRegistry      ToolMap
	TopP              float64
	Unsafe            bool
	Verbose           bool
	Version           bool
	Walk              bool // used with FilePaths
}

func main() {
	// create context that listens for OS interrupt signals
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := run(ctx); err != nil {
		if err.Error() != "" {
			fmt.Fprintf(os.Stderr, "%v\n", err)
		}
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	params := &Parameters{}
	keyVals := ParamMap{}

	if err := parseFlags(params, &keyVals); err != nil {
		return err
	}

	if params.Version {
		printVersion()
		return nil
	}

	defer cleanup(params)

	// store keyVals and params in context
	ctx = context.WithValue(ctx, "keyVals", keyVals)
	ctx = context.WithValue(ctx, "params", params)

	// stash MCP client sessions in params.MCPSessionsa
	if params.Help || params.Tool {
		if err := initMCPSessions(ctx, params); err != nil {
			return fmt.Errorf("MCP error: %w", err)
		}
	}

	// handle help and version flags before any further processing
	// context includes params with list to known tools
	if params.Help {
		emitUsage(ctx, os.Stdout, true)
		return nil
	}

	// argument validation
	if err := isArgsInvalid(params, keyVals); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n\n", err)
		emitUsage(ctx, os.Stdout, false)
		return fmt.Errorf("")
	}

	if err := validateEnv(); err != nil {
		return fmt.Errorf("Environment error: %w", err)
	}

	if err := genContent(ctx, os.Stdin, os.Stdout); err != nil {
		return fmt.Errorf("Generation error: %w", err)
	}

	return nil
}

// parseFlags handles flag definitions and parameter map for variable substitutions in prompts.
func parseFlags(params *Parameters, keyVals *ParamMap) error {
	// default parameter values
	params.K = 3
	params.Lambda = 0.5
	params.Temp = 1.0
	params.TopP = 0.95
	params.ThinkingLevel = genai.ThinkingLevelUnspecified
	params.Timeout = 300 * time.Second
	params.EmbModel = "gemini-embedding-001"
	params.GenModel = "gemini-2.5-flash"
	params.SegModel = "image-segmentation-001"

	if err := loadPrefs(params); err != nil {
		return fmt.Errorf("Error loading preferences from %s: %v\n", DotGenRc, err)
	}

	flag.BoolVar(&params.Verbose, "V", false, "output model details, system instructions, chat history and thoughts")
	flag.BoolVar(&params.SegmentBackground, "b", false, "background segmentation mode (default: foreground)")
	flag.BoolVar(&params.ChatMode, "c", false, "enter chat mode (incompatible with -json, -img, -code or -g)")
	flag.BoolVar(&params.CodeGen, "code", false, "code execution tool (incompatible with -g, -json, -img or -tool)")
	flag.Var(&params.DigestPaths, "d", "path to a digest folder")
	flag.BoolVar(&params.Embed, "e", false, fmt.Sprintf("write text embeddings to digest (default model \"%s\")", params.EmbModel))
	flag.Var(&params.FilePaths, "f", "GCS URI, file, directory or quoted pattern of files to attach")
	flag.BoolVar(&params.GoogleSearch, "g", false, "Google search tool (incompatible with -code, -json, -img and -tool)")
	flag.BoolVar(&params.Help, "h", false, "show available tools, this help message and exit")
	flag.BoolVar(&params.OnlyKvs, "i", false, "only store metadata with embeddings and ignore the content")
	flag.BoolVar(&params.ImgModality, "img", false, "generate jpeg images (use -m to set a supported model)")
	flag.BoolVar(&params.JSON, "json", false, "structured output (incompatible with -g, -code, -img and -tool)")
	flag.IntVar(&params.K, "k", params.K, "maximum number of entries from digest to retrieve")
	flag.Float64Var(&params.Lambda, "l", params.Lambda, "balance accuracy and diversity querying digests [0.0,1.0]")
	flag.Func("think", fmt.Sprintf("%s, %s, %s or %s (default: %s)",
		genai.ThinkingLevelMinimal,
		genai.ThinkingLevelLow,
		genai.ThinkingLevelMedium,
		genai.ThinkingLevelHigh,
		params.ThinkingLevel), func(val string) error {
		params.ThinkingLevel = genai.ThinkingLevel(strings.ToUpper(val))
		return nil
	})
	flag.StringVar(&params.GenModel, "m", params.GenModel, "model name")
	flag.Var(&params.MCPServers, "mcp", "mcp stdio or streamable server command")
	flag.Var(keyVals, "p", "prompt parameter value in format key=val")
	flag.BoolVar(&params.Walk, "r", false, "process directory declared with -f recursively")
	flag.BoolVar(&params.SystemInstruction, "s", false, "treat argument as system prompt")
	flag.BoolVar(&params.Segment, "seg", false, fmt.Sprintf("segment image on VertexAI (default model \"%s\")", params.SegModel))
	flag.BoolVar(&params.TokenCount, "t", false, "output total number of tokens")
	flag.Float64Var(&params.Temp, "temp", params.Temp, "sampling during response generation [0.0,2.0]")
	flag.DurationVar(&params.Timeout, "timeout", params.Timeout, "time limit for single turn content generation")
	flag.BoolVar(&params.Tool, "tool", false, "invoke one of the tools (incompatible with -s, -g, -json, -img or -code)")
	flag.Float64Var(&params.TopP, "top_p", params.TopP, "how the model selects tokens for generation [0.0,1.0]")
	flag.BoolVar(&params.Unsafe, "unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	flag.BoolVar(&params.Version, "v", false, "show version and exit")
	flag.Parse()

	params.Args = flag.Args()
	params.Interactive = !isRedirected(os.Stdin)
	params.ToolRegistry = ToolMap{}

	return nil
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
func emitUsage(ctx context.Context, out io.Writer, emitTools bool) {
	var tools string
	if emitTools {
		tools, _ = knownTools(ctx)
		fmt.Fprintln(out, "Command-line interface to Google Gemini large language models")
		fmt.Fprintln(out, "  Requires a valid GOOGLE_API_KEY environment variable set.")
		fmt.Fprintln(out, "  VertexAI backend with valid GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
		fmt.Fprintln(out, "  Content is generated by a prompt and optional system instructions.")
		fmt.Fprintln(out, "  Use - to assign stdin as prompt or as attached file.")
		fmt.Fprintf(out, "\n")
	}
	fmt.Fprintln(out, "Usage: gen [options] <prompt>")
	fmt.Fprintf(out, "\n")
	if emitTools {
		fmt.Fprintln(out, fmt.Sprintf("Tools:\n%s", tools))
		fmt.Fprintf(out, "\n")
	}
	fmt.Fprintln(out, "Options:")
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

func cleanup(params *Parameters) {
	for _, sess := range params.MCPSessions {
		if sess != nil {
			sess.Close()
		}
	}
	// final token count report
	if params.TokenCount {
		fmt.Printf("\n"+important("%d tokens")+"\n", TokenCount.Load())
	}
}
