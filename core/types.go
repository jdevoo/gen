package core

import (
	"context"
	"fmt"
	"strconv"
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
	ClientKey    ContextKey = "genClient"
)

// GetGenClient retrieves the shared *genai.Client from context, or initializes one if absent.
func GetGenClient(ctx context.Context) (*genai.Client, error) {
	if client, ok := ctx.Value(ClientKey).(*genai.Client); ok && client != nil {
		return client, nil
	}
	return genai.NewClient(ctx, nil)
}

// Parameters holds gen flag values as well as Args and MCP sessions.
type Parameters struct {
	Args              []string // non-flag command-line arguments i.e. prompt
	Cache             bool
	ChatMode          bool
	CodeGen           bool
	CountTokens       bool
	DigestPaths       ParamArray // RAG
	Embed             bool       // RAG
	EmbModel          string
	FilePaths         ParamArray
	FileURIs          ParamMap
	FrequencyPenalty  OptionalFloat32
	GenModel          string
	GoogleSearch      bool
	Help              bool
	ImgModality       bool
	Interactive       bool // terminal session?
	JSON              bool
	K                 int
	Lambda            float64
	MaxEmbedBatchSize int
	MCPServers        ParamArray
	MCPSessions       SessionArray
	NoOp              bool
	OutPath           string
	OutRedirected     bool
	OnlyKvs           bool // RAG
	PresencePenalty   OptionalFloat32
	SystemInstruction bool
	Seed              OptionalInt32
	StopSequences     ParamArray
	Temp              OptionalFloat32
	ThinkingLevel     genai.ThinkingLevel
	Timeout           time.Duration
	Tool              bool
	ToolRegistry      ToolMap
	TopP              OptionalFloat32
	Unsafe            bool
	Usage             TokenUsage
	Verbose           bool
	Version           bool
	Walk              bool // used with FilePaths
}

// TokenUsage holds token metrics across prompt, candidates, thinking, and tools.
type TokenUsage struct {
	PromptTokens        int32 `json:"promptTokens"`
	CandidatesTokens    int32 `json:"candidatesTokens"`
	ThoughtsTokens      int32 `json:"thoughtsTokens,omitempty"`
	CachedContentTokens int32 `json:"cachedContentTokens,omitempty"`
	ToolUsePromptTokens int32 `json:"toolUsePromptTokens,omitempty"`
	TotalTokens         int32 `json:"totalTokens"`
}

// Add accumulates token metrics from another TokenUsage instance.
func (u *TokenUsage) Add(other *TokenUsage) {
	if other == nil {
		return
	}
	u.PromptTokens += other.PromptTokens
	u.CandidatesTokens += other.CandidatesTokens
	u.ThoughtsTokens += other.ThoughtsTokens
	u.CachedContentTokens += other.CachedContentTokens
	u.ToolUsePromptTokens += other.ToolUsePromptTokens
	u.TotalTokens += other.TotalTokens
}

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

// OptionalFloat32 implements the flag.Value interface
type OptionalFloat32 struct {
	Value   *float32
	Defined bool
}

func (o *OptionalFloat32) String() string {
	if !o.Defined {
		return ""
	}
	return strconv.FormatFloat(float64(*o.Value), 'g', -1, 32)
}

// Set is called by the flag package when the flag is provided
func (o *OptionalFloat32) Set(s string) error {
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return err
	}
	val := float32(v)
	o.Value = &val
	o.Defined = true
	return nil
}

// OptionalInt32 implements the flag.Value interface
type OptionalInt32 struct {
	Value   *int32
	Defined bool
}

// String is required by the flag.Value interface
func (o *OptionalInt32) String() string {
	if !o.Defined {
		return ""
	}
	return strconv.Itoa(int(*o.Value))
}

// Set is called by the flag package when the flag is provided
func (o *OptionalInt32) Set(s string) error {
	v, err := strconv.ParseInt(s, 10, 32)
	if err != nil {
		return err
	}
	val := int32(v)
	o.Value = &val
	o.Defined = true
	return nil
}

// ToolRegistry maps tool names to the session
type ToolMap map[string]*mcp.ClientSession

// SessionArray holds a list of MCP client session
type SessionArray []*mcp.ClientSession
