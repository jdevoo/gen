package core

import (
	"fmt"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

type ContextKey string

const (
	ParamsKey    ContextKey = "params"
	KeyValsKey   ContextKey = "keyVals"
	ElicitErrKey ContextKey = "elicitError"
)

// Parameters holds gen flag values as well as Args and MCP sessions.
type Parameters struct {
	Args              []string // non-flag command-line arguments i.e. prompt
	ChatMode          bool
	CodeGen           bool
	CountTokens       bool
	DigestPaths       ParamArray // RAG
	Embed             bool       // RAG
	EmbModel          string
	FilePaths         ParamArray
	FileURIs          []string
	GenModel          string
	GoogleSearch      bool
	Help              bool
	ImgModality       bool
	Interactive       bool // terminal session?
	JSON              bool
	K                 int
	Lambda            float64
	MCPServers        ParamArray
	MCPSessions       SessionArray
	OutPath           string
	OutRedirected     bool
	OnlyKvs           bool // RAG
	SystemInstruction bool
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

type Tool struct{}

type ParamError struct {
	Message string
}

func (e *ParamError) Error() string {
	return e.Message
}

// ParamMap holds key-value pairs for string replacement.
type ParamMap map[string]string

// String implements the flag.Value interface for ParamMap.
func (*ParamMap) String() string { return "" }

// Set implements the flag.Value interface for ParamMap.
func (m *ParamMap) Set(kv string) error {
	parts := strings.SplitN(kv, "=", 2) // limit splits to 2
	if len(parts) != 2 {
		return fmt.Errorf("invalid parameter %s", kv)
	}
	(*m)[parts[0]] = parts[1]
	return nil
}

// ParamArray holds a list of strings e.g. file paths.
type ParamArray []string

// String implements the flag.Value interface for ParamArray.
func (*ParamArray) String() string { return "" }

// Set implements the flag.Value interface for ParamArray.
func (a *ParamArray) Set(val string) error {
	*a = append(*a, val)
	return nil
}

// ToolRegistry maps tool names to the session
type ToolMap map[string]*mcp.ClientSession

// SessionArray holds a list of MCP client session
type SessionArray []*mcp.ClientSession
