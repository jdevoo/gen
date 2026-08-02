package sparql

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ntParser is a parser for N-Triples lines.
type ntParser struct {
	s   string
	pos int
}

func (p *ntParser) skipSpace() {
	for p.pos < len(p.s) && (p.s[p.pos] == ' ' || p.s[p.pos] == '\t') {
		p.pos++
	}
}

func (p *ntParser) parseValue() (Value, error) {
	p.skipSpace()
	if p.pos >= len(p.s) {
		return nil, io.EOF
	}

	if p.s[p.pos] == '<' {
		// Parse URI
		start := p.pos + 1
		end := strings.IndexByte(p.s[start:], '>')
		if end == -1 {
			return nil, fmt.Errorf("malformed URI in N-Triples")
		}
		uriStr := p.s[start : start+end]
		p.pos = start + end + 1
		return URI(uriStr), nil
	}

	if p.s[p.pos] == '_' {
		// Parse Blank Node, format _:name
		if p.pos+1 < len(p.s) && p.s[p.pos+1] == ':' {
			start := p.pos + 2
			end := start
			for end < len(p.s) && !isNTSpaceOrDot(p.s[end]) {
				end++
			}
			bnodeStr := p.s[start:end]
			p.pos = end
			return BNode(bnodeStr), nil
		}
		return nil, fmt.Errorf("malformed blank node in N-Triples")
	}

	if p.s[p.pos] == '"' {
		// Parse Literal
		start := p.pos + 1
		end := -1
		escaped := false
		for i := start; i < len(p.s); i++ {
			if escaped {
				escaped = false
				continue
			}
			if p.s[i] == '\\' {
				escaped = true
				continue
			}
			if p.s[i] == '"' {
				end = i
				break
			}
		}
		if end == -1 {
			return nil, fmt.Errorf("malformed literal (unclosed quote) in N-Triples")
		}
		litVal := p.s[start:end]
		litVal = unescapeNTString(litVal)
		p.pos = end + 1

		// Check for language tag or datatype
		if p.pos < len(p.s) && p.s[p.pos] == '@' {
			p.pos++
			langStart := p.pos
			for p.pos < len(p.s) && !isNTSpaceOrDot(p.s[p.pos]) {
				p.pos++
			}
			lang := p.s[langStart:p.pos]
			return Literal{Value: litVal, LanguageTag: lang}, nil
		}

		if p.pos+1 < len(p.s) && p.s[p.pos] == '^' && p.s[p.pos+1] == '^' {
			p.pos += 2
			dtVal, err := p.parseValue()
			if err != nil {
				return nil, err
			}
			dtURI, ok := dtVal.(URI)
			if !ok {
				return nil, fmt.Errorf("expected URI datatype after ^^")
			}
			return Literal{Value: litVal, DataType: dtURI}, nil
		}

		return Literal{Value: litVal}, nil
	}

	return nil, fmt.Errorf("unexpected character %q at position %d", p.s[p.pos], p.pos)
}

func isNTSpaceOrDot(c byte) bool {
	return c == ' ' || c == '\t' || c == '.' || c == '\r' || c == '\n'
}

func unescapeNTString(s string) string {
	r := strings.NewReplacer(
		`\"`, `"`,
		`\\`, `\`,
		`\t`, "\t",
		`\n`, "\n",
		`\r`, "\r",
	)
	return r.Replace(s)
}

func parseNTriplesTriple(line string) (Value, URI, Value, error) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return nil, "", nil, nil
	}
	p := &ntParser{s: line}
	s, err := p.parseValue()
	if err != nil {
		return nil, "", nil, err
	}
	predVal, err := p.parseValue()
	if err != nil {
		return nil, "", nil, err
	}
	pred, ok := predVal.(URI)
	if !ok {
		return nil, "", nil, fmt.Errorf("predicate must be a URI, got %T", predVal)
	}
	o, err := p.parseValue()
	if err != nil {
		return nil, "", nil, err
	}
	// Verify the trailing dot
	p.skipSpace()
	if p.pos < len(p.s) && p.s[p.pos] == '.' {
		p.pos++
	}
	return s, pred, o, nil
}

// NTriplesQueryResult is a SPARQL query result for CONSTRUCT/DESCRIBE graph queries.
type NTriplesQueryResult struct {
	r       io.ReadCloser
	scanner *bufio.Scanner
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewNTriplesQueryResult returns a QueryResult for N-Triples streams.
func NewNTriplesQueryResult(ctx context.Context, r io.ReadCloser) QueryResult {
	ctx, cancel := context.WithCancel(ctx)
	return &NTriplesQueryResult{
		r:       r,
		scanner: bufio.NewScanner(r),
		ctx:     ctx,
		cancel:  cancel,
	}
}

// Variables returns the variables representing graph triples.
func (n *NTriplesQueryResult) Variables() []string {
	return []string{"subject", "predicate", "object"}
}

// Next retrieves the next RDF triple as a binding map.
func (n *NTriplesQueryResult) Next() (map[string]Value, error) {
	for {
		select {
		case <-n.ctx.Done():
			return nil, n.ctx.Err()
		default:
			if !n.scanner.Scan() {
				if err := n.scanner.Err(); err != nil {
					return nil, err
				}
				return nil, io.EOF
			}
			line := n.scanner.Text()
			s, p, o, err := parseNTriplesTriple(line)
			if err != nil {
				return nil, err
			}
			if s == nil {
				continue // skip empty/comment lines
			}
			return map[string]Value{
				"subject":   s,
				"predicate": p,
				"object":    o,
			}, nil
		}
	}
}

// Boolean returns an error because graph queries do not return a boolean value.
func (n *NTriplesQueryResult) Boolean() (bool, error) {
	return false, errors.New("sparql: Boolean() called on a graph query result")
}

// Close closes the underlying reader.
func (n *NTriplesQueryResult) Close() error {
	if n.cancel != nil {
		n.cancel()
	}
	return n.r.Close()
}
