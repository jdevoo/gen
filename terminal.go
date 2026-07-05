package main

import (
	"strings"
	"sync"
)

const (
	ansiReset  = "\033[0m"
	ansiBold   = "\033[1m"
	ansiNoBold = "\033[22m"
	ansiYellow = "\033[93m"
	ansiWhite  = "\033[97m"

	colorCyan = "\033[36m"
	colorRed  = "\033[31m"
	colorRole = "\033[1;37;46m"
)

// MarkdownParser holds state across Parse calls.
type MarkdownParser struct {
	mu          sync.Mutex
	isBold      bool
	inCodeBlock bool
	inThought   bool
	buffer      string
}

// setThought safely updates the MarkdownParser state
func (p *MarkdownParser) setThought(b bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.inThought = b
}

// ParseMarkdown turns pairs of `**` to terminal escape sequences
func (p *MarkdownParser) parse(s string) string {
	p.mu.Lock()
	defer p.mu.Unlock()

	s = p.buffer + s
	p.buffer = ""

	var sb strings.Builder
	if p.inCodeBlock {
		sb.WriteString(ansiReset)
	} else {
		if p.inThought {
			sb.WriteString(ansiYellow)
		} else {
			sb.WriteString(ansiWhite)
		}
		if p.isBold {
			sb.WriteString(ansiBold)
		}
	}
	n := len(s)
	for i := 0; i < n; {
		// protection for splits at the end of stream
		if i == n-1 && (s[i] == '*' || s[i] == '`' || s[i] == '$') {
			p.buffer = s[i:]
			break
		}
		if i == n-2 && (s[i:i+2] == "``" || (s[i] == '$' && s[i+1] != '$')) {
			p.buffer = s[i:]
			break
		}
		// block code
		if i+2 < n && s[i:i+3] == "```" {
			if p.inCodeBlock {
				if p.inThought {
					sb.WriteString(ansiYellow)
				} else {
					sb.WriteString(ansiWhite)
				}
				sb.WriteString("```")
			} else {
				sb.WriteString("```")
				sb.WriteString(ansiReset)
				if p.isBold {
					sb.WriteString(ansiBold)
				}
			}
			p.inCodeBlock = !p.inCodeBlock
			i += 3
			continue
		}
		// skip LaTeX, bold if inside code block
		if p.inCodeBlock {
			sb.WriteByte(s[i])
			i++
			continue
		}
		// bold delimiter
		if i+1 < n && s[i:i+2] == "**" {
			if p.isBold {
				sb.WriteString(ansiNoBold)
			} else {
				sb.WriteString(ansiBold)
			}
			p.isBold = !p.isBold
			i += 2
			continue
		}
		// normal character
		sb.WriteByte(s[i])
		i++
	}
	sb.WriteString(ansiReset)

	return sb.String()
}

func (p *MarkdownParser) flush(isRedirected bool) string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.buffer == "" {
		return ""
	}
	var sb strings.Builder
	if !isRedirected {
		if p.inCodeBlock {
			sb.WriteString(ansiReset)
		} else {
			if p.inThought {
				sb.WriteString(ansiYellow) // Yellow for thoughts
			} else {
				sb.WriteString(ansiWhite)
			}
			if p.isBold {
				sb.WriteString(ansiBold)
			}
		}
	}
	sb.WriteString(p.buffer)
	if !isRedirected {
		sb.WriteString(ansiReset)
	}
	p.buffer = ""
	p.isBold = false
	p.inCodeBlock = false
	p.inThought = false
	return sb.String()
}

func infos(s string) string     { return colorCyan + s + ansiReset }
func important(s string) string { return colorRed + s + ansiReset }
func roles(s string) string     { return colorRole + s + ansiReset }
