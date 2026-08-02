package sparql

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
)

// detectQueryType detects whether the query is SELECT, ASK, CONSTRUCT, or DESCRIBE.
func detectQueryType(query string) string {
	var tokens []string
	lines := strings.Split(query, "\n")
	for _, line := range lines {
		if idx := strings.Index(line, "#"); idx != -1 {
			line = line[:idx]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		tokens = append(tokens, strings.Fields(line)...)
	}

	for i := 0; i < len(tokens); i++ {
		token := strings.ToUpper(tokens[i])
		if token == "PREFIX" {
			i += 2
			continue
		}
		if token == "BASE" {
			i += 1
			continue
		}
		switch token {
		case "SELECT", "ASK", "CONSTRUCT", "DESCRIBE":
			return token
		}
	}
	return "SELECT"
}

// Query queries the endpoint.
func (c *Client) Query(
	ctx context.Context,
	query string,
	params ...Param,
) (QueryResult, error) {
	prefix := ""
	if len(c.Prefixes) > 0 {
		ss := make([]string, 0, len(c.Prefixes)*5)
		for prefix, uri := range c.Prefixes {
			ss = append(ss, "PREFIX ")
			ss = append(ss, prefix)
			ss = append(ss, ": ")
			ss = append(ss, uri.Ref())
			ss = append(ss, "\n")
		}
		prefix = strings.Join(ss, "")
	}

	request, err := c.buildRequest(ctx, prefix, query, params...)
	if err != nil {
		return nil, err
	}

	resp, err := c.HTTPClient.Do(request)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		defer resp.Body.Close()
		errMsg := cleanErrorBody(resp.Body)
		return nil, fmt.Errorf(
			"SPARQL QUERY error. status code: %d msg: %s",
			resp.StatusCode,
			errMsg,
		)
	}

	queryType := detectQueryType(query)
	if queryType == "CONSTRUCT" || queryType == "DESCRIBE" {
		return NewNTriplesQueryResult(ctx, resp.Body), nil
	}

	// For SELECT and ASK, delegate to c.ResultParser which manages the response body.
	return c.ResultParser.Parse(ctx, resp.Body)
}

// interpolate builds the prefix string and replaces parameters by sorting them descending by key length
func (c *Client) interpolate(prefix, query string, params ...Param) (string, error) {
	const defaultBufferSize = 1024
	b := bytes.NewBuffer(make([]byte, 0, defaultBufferSize))

	// Write prefix
	if _, err := b.Write([]byte(prefix)); err != nil {
		return "", err
	}
	// Replace parameters by sorting them descending by key length to avoid prefix overlap (e.g. $1 replacing part of $10)
	type pair struct {
		key string
		val string
	}
	var pairs []pair
	for _, p := range params {
		v := p.Serialize()
		for _, key := range p.Placeholders() {
			pairs = append(pairs, pair{key: key, val: v})
		}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return len(pairs[i].key) > len(pairs[j].key)
	})

	replacePairs := make([]string, 0, 2*len(pairs))
	for _, pr := range pairs {
		replacePairs = append(replacePairs, pr.key, pr.val)
	}

	_, err := strings.NewReplacer(replacePairs...).WriteString(b, query)
	if err != nil {
		return "", err
	}
	return b.String(), nil
}

// buildRequest is a helper function for Query
func (c *Client) buildRequest(ctx context.Context, prefix, query string, params ...Param) (*http.Request, error) {
	method := c.Method
	if method == "" {
		method = http.MethodGet
	}

	built, err := c.interpolate(prefix, query, params...)
	if err != nil {
		return nil, err
	}

	queryType := detectQueryType(query)
	format := c.ResultParser.Format()
	if queryType == "CONSTRUCT" || queryType == "DESCRIBE" {
		format = "text/plain"
	}

	var req *http.Request
	if strings.ToUpper(method) == http.MethodPost {
		form := url.Values{}
		form.Set("query", built)
		form.Set("format", format)
		body := form.Encode()

		req, err = http.NewRequest(http.MethodPost, c.Endpoint, strings.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	} else {
		req, err = http.NewRequest(http.MethodGet, c.Endpoint, nil)
		if err != nil {
			return nil, err
		}
		urlParams := req.URL.Query()
		urlParams.Set("query", built)
		urlParams.Set("format", format)
		req.URL.RawQuery = urlParams.Encode()
	}

	req = req.WithContext(ctx)
	return req, nil
}

// Update executes a SPARQL 1.1 Update operation on the endpoint.
func (c *Client) Update(ctx context.Context, update string, params ...Param) error {
	prefix := ""
	if len(c.Prefixes) > 0 {
		ss := make([]string, 0, len(c.Prefixes)*5)
		for prefix, uri := range c.Prefixes {
			ss = append(ss, "PREFIX ")
			ss = append(ss, prefix)
			ss = append(ss, ": ")
			ss = append(ss, uri.Ref())
			ss = append(ss, "\n")
		}
		prefix = strings.Join(ss, "")
	}

	built, err := c.interpolate(prefix, update, params...)
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Endpoint, strings.NewReader(built))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/sparql-update")

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		errMsg := cleanErrorBody(resp.Body)
		return fmt.Errorf(
			"SPARQL UPDATE error. status code: %d msg: %s",
			resp.StatusCode,
			errMsg,
		)
	}

	return nil
}

var htmlTagRegex = regexp.MustCompile("<[^>]*>")

func cleanErrorBody(r io.Reader) string {
	// Read up to 2KB
	buf := make([]byte, 2048)
	n, _ := io.ReadAtLeast(r, buf, 1)
	if n == 0 {
		return "empty response body"
	}
	body := string(buf[:n])

	// If it contains html tags, let's clean it
	bodyLower := strings.ToLower(body)
	if strings.Contains(bodyLower, "<html") || strings.Contains(bodyLower, "<!doctype") {
		// Try to find the title or body heading
		reTitle := regexp.MustCompile(`(?i)<title>([^<]+)</title>`)
		if m := reTitle.FindStringSubmatch(body); len(m) > 1 {
			return strings.TrimSpace(m[1])
		}
		reH1 := regexp.MustCompile(`(?i)<h1>([^<]+)</h1>`)
		if m := reH1.FindStringSubmatch(body); len(m) > 1 {
			return strings.TrimSpace(m[1])
		}
		rePre := regexp.MustCompile(`(?i)<pre>([^<]+)</pre>`)
		if m := rePre.FindStringSubmatch(body); len(m) > 1 {
			return strings.TrimSpace(m[1])
		}
		// Strip all HTML tags
		body = htmlTagRegex.ReplaceAllString(body, " ")
		body = strings.Join(strings.Fields(body), " ")
	}

	// Truncate to 500 chars for LLM safety
	body = strings.TrimSpace(body)
	if len(body) > 500 {
		return body[:500] + "..."
	}
	return body
}
