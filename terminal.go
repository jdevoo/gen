package main

import (
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

const (
	ansiReset  = "\033[0m"
	ansiBold   = "\033[1m"
	ansiNoBold = "\033[22m"
	ansiYellow = "\033[93m"
	ansiWhite  = "\033[97m"

	colorInfo      = "\033[36m"
	colorImportant = "\033[1;31m"
	colorRole      = "\033[1;37;46m"
)

// MarkdownParser holds state across Parse calls.
type MarkdownParser struct {
	mu          sync.Mutex
	isBold      bool
	inCodeBlock bool
	inThought   bool
	buffer      string
	lastRune    rune
	imageFound  bool
}

// setThought safely updates the MarkdownParser state
func (p *MarkdownParser) setThought(b bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.inThought = b
}

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

func (p *MarkdownParser) getPrevRune(s string, i int) rune {
	if i > 0 {
		r, _ := utf8.DecodeLastRuneInString(s[:i])
		return r
	}
	return p.lastRune
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
	lastRune := p.lastRune
	for i := 0; i < n; {
		// protection for splits at the end of stream
		if i == n-1 {
			if s[i] == '*' || s[i] == '`' || s[i] == '$' {
				p.buffer = s[i:]
				break
			}
			// buffer trailing ! if it could start an image tag
			if s[i] == '!' && (i == 0 || s[i-1] == '\n' || s[i-1] == ' ' || s[i-1] == '\t') {
				p.buffer = s[i:]
				break
			}
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
			lastRune = '`'
			i += 3
			continue
		}
		// skip LaTeX, bold if inside code block
		if p.inCodeBlock {
			if s[i] < utf8.RuneSelf {
				sb.WriteByte(s[i])
				lastRune = rune(s[i])
				i++
			} else {
				r, size := utf8.DecodeRuneInString(s[i:])
				sb.WriteString(s[i : i+size])
				lastRune = r
				i += size
			}
			continue
		}
		if i+1 < n && s[i:i+2] == "![" {
			closeBracket := strings.Index(s[i+2:], "](")
			if closeBracket != -1 {
				parenStart := i + 2 + closeBracket + 1
				closeParen := strings.IndexByte(s[parenStart+1:], ')')
				if closeParen != -1 {
					parenEnd := parenStart + 1 + closeParen
					i = parenEnd + 1
					if i < n && s[i] == '\n' {
						i++
					}
					p.buffer = s[i:]
					p.imageFound = true
					lastRune = 0
					break
				}
			}
			if !strings.Contains(s[i:], "\n\n") && len(s[i:]) < 500 {
				p.buffer = s[i:]
				break
			}
		}
		// bold delimiter or asterisk redaction
		if i+1 < n && s[i:i+2] == "**" {
			prev := p.getPrevRune(s, i)
			var next rune
			if i+2 < n {
				next, _ = utf8.DecodeRuneInString(s[i+2:])
			}
			prevIsWord := isWordRune(prev)
			nextIsWord := i+2 < n && isWordRune(next)
			// intra-word redaction eg f**king or f**?
			if (prevIsWord && nextIsWord) || (!p.isBold && prevIsWord) {
				sb.WriteString("**")
				lastRune = '*'
				i += 2
				continue
			}
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
		if s[i] < utf8.RuneSelf {
			sb.WriteByte(s[i])
			lastRune = rune(s[i])
			i++
		} else {
			r, size := utf8.DecodeRuneInString(s[i:])
			sb.WriteString(s[i : i+size])
			lastRune = r
			i += size
		}
	}
	p.lastRune = lastRune
	sb.WriteString(ansiReset)

	return sb.String()
}

func (p *MarkdownParser) flush(isRedirected bool) string {
	p.mu.Lock()
	defer p.mu.Unlock()
	defer func() {
		p.buffer = ""
		p.isBold = false
		p.inCodeBlock = false
		p.inThought = false
		p.lastRune = 0
		p.imageFound = false
	}()
	if p.buffer == "" {
		return ""
	}
	var sb strings.Builder
	if !isRedirected {
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
	}
	sb.WriteString(p.buffer)
	if !isRedirected {
		sb.WriteString(ansiReset)
	}
	return sb.String()
}

func infos(s string) string     { return colorInfo + s + ansiReset }
func important(s string) string { return colorImportant + s + ansiReset }
