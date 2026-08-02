package sparql

import (
	"context"
	"fmt"
	"net/http"
	"time"
)

// Client queries to its SPARQL endpoint.
type Client struct {
	HTTPClient   http.Client
	Endpoint     string
	Prefixes     map[string]URI
	ResultParser ResultParser
	Method       string
}

// WithResultParser sets the parser.
func WithResultParser(resultParser ResultParser) func(*Client) {
	return func(c *Client) { c.ResultParser = resultParser }
}

// HTTPClient replaces default HTTP client.
func WithHTTPClient(httpClient *http.Client) func(*Client) {
	return func(c *Client) {
		c.HTTPClient = *httpClient
	}
}

// WithPrefix sets a global PREFIX for all queries.
func WithPrefix(prefix string, uri URI) func(*Client) {
	return func(c *Client) {
		if c.Prefixes == nil {
			c.Prefixes = make(map[string]URI)
		}
		c.Prefixes[prefix] = uri
	}
}

// WithMaxIdleConns sets the maximum idle connections for the HTTP client's transport.
func WithMaxIdleConns(maxIdleConns int) func(*Client) {
	return func(c *Client) {
		if c.HTTPClient.Transport == nil {
			c.HTTPClient.Transport = &http.Transport{}
		}
		if t, ok := c.HTTPClient.Transport.(*http.Transport); ok {
			t.MaxIdleConns = maxIdleConns
		}
	}
}

// WithIdleConnTimeout sets the idle connection timeout for the HTTP client's transport.
func WithIdleConnTimeout(idleConnTimeout time.Duration) func(*Client) {
	return func(c *Client) {
		if c.HTTPClient.Transport == nil {
			c.HTTPClient.Transport = &http.Transport{}
		}
		if t, ok := c.HTTPClient.Transport.(*http.Transport); ok {
			t.IdleConnTimeout = idleConnTimeout
		}
	}
}

// WithTimeout sets the timeout for the HTTP client.
func WithTimeout(timeout time.Duration) func(*Client) {
	return func(c *Client) {
		c.HTTPClient.Timeout = timeout
	}
}

// WithMethod sets the query request method.
func WithMethod(method string) func(*Client) {
	return func(c *Client) {
		c.Method = method
	}
}

// New returns `sparql.Client`.
func New(endpoint string, opts ...func(*Client)) (*Client, error) {
	transport := &http.Transport{}
	client := &Client{
		HTTPClient:   http.Client{Transport: transport},
		Endpoint:     endpoint,
		Prefixes:     make(map[string]URI),
		ResultParser: NewXMLResultParser(),
		Method:       "GET",
	}
	for _, opt := range opts {
		opt(client)
	}
	return client, nil
}

// Close and nothing to do as http.Client manages its own connections.
func (c *Client) Close() error {
	return nil
}

// Ping sends an HTTP HEAD request to the endpoint.
func (c *Client) Ping(ctx context.Context) (err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, c.Endpoint, nil)
	if err != nil {
		return err
	}

	res, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	if res.StatusCode != http.StatusOK {
		return fmt.Errorf("SPARQL ping error. status code %d", res.StatusCode)
	}

	return nil
}
