package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"
)

// authUIMutex sequences user prompts when multiple MCP servers request OAuth authorization.
var authUIMutex sync.Mutex

// tokenStore manages persisting OAuth tokens to ~/.cache/gen/mcp_tokens.json.
type tokenStore struct {
	mu       sync.Mutex
	filePath string
	tokens   map[string]*oauth2.Token
}

func newTokenStore() (*tokenStore, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil || cacheDir == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get user home directory: %w", err)
		}
		cacheDir = filepath.Join(homeDir, ".cache")
	}
	genDir := filepath.Join(cacheDir, "gen")
	if err := os.MkdirAll(genDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create cache directory %s: %w", genDir, err)
	}

	store := &tokenStore{
		filePath: filepath.Join(genDir, "mcp_tokens.json"),
		tokens:   make(map[string]*oauth2.Token),
	}
	store.load()
	return store, nil
}

func (s *tokenStore) load() {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.filePath)
	if err == nil {
		_ = json.Unmarshal(data, &s.tokens)
	}
}

func (s *tokenStore) save() error {
	data, err := json.MarshalIndent(s.tokens, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.filePath, data, 0600)
}

func (s *tokenStore) Get(key string) *oauth2.Token {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.tokens[key]
}

func (s *tokenStore) Put(key string, token *oauth2.Token) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tokens == nil {
		s.tokens = make(map[string]*oauth2.Token)
	}
	s.tokens[key] = token
	return s.save()
}

type cachingTokenSource struct {
	base      oauth2.TokenSource
	serverURL string
	store     *tokenStore
}

func (c *cachingTokenSource) Token() (*oauth2.Token, error) {
	tok, err := c.base.Token()
	if err == nil && tok != nil {
		_ = c.store.Put(c.serverURL, tok)
	}
	return tok, err
}

// buildOAuthHandler creates an OAuthHandler.
func buildOAuthHandler(serverURL string, httpClient *http.Client, spinner *Spinner) (auth.OAuthHandler, error) {
	store, err := newTokenStore()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize token store: %w", err)
	}

	cachedToken := store.Get(serverURL)
	var initialTS oauth2.TokenSource
	if cachedToken != nil && cachedToken.AccessToken != "" {
		initialTS = oauth2.StaticTokenSource(cachedToken)
	}

	// loopback port for the redirect URI
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("failed to allocate loopback port for OAuth: %w", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()

	redirectURI := fmt.Sprintf("http://127.0.0.1:%d/callback", port)

	fetcher := func(fetchCtx context.Context, args *auth.AuthorizationArgs) (*auth.AuthorizationResult, error) {
		authUIMutex.Lock()
		defer authUIMutex.Unlock()

		if spinner != nil {
			spinner.Stop()
		}

		listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err != nil {
			return nil, fmt.Errorf("failed to open loopback listener on port %d: %w", port, err)
		}
		defer listener.Close()

		resChan := make(chan *auth.AuthorizationResult, 1)
		errChan := make(chan error, 1)

		mux := http.NewServeMux()
		server := &http.Server{Handler: mux}

		mux.HandleFunc("/callback", func(w http.ResponseWriter, r *http.Request) {
			q := r.URL.Query()
			if errCode := q.Get("error"); errCode != "" {
				desc := q.Get("error_description")
				errChan <- fmt.Errorf("oauth error %s: %s", errCode, desc)
				http.Error(w, "Authorization failed. You may close this tab.", http.StatusUnauthorized)
				return
			}
			code := q.Get("code")
			state := q.Get("state")
			iss := q.Get("iss")

			resChan <- &auth.AuthorizationResult{
				Code:  code,
				State: state,
				Iss:   iss,
			}

			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			fmt.Fprint(w, "<html><body><h2>Authentication successful!</h2><p>You can close this tab and return to gen.</p></body></html>")
		})

		go func() {
			if serveErr := server.Serve(listener); serveErr != nil && serveErr != http.ErrServerClosed {
				select {
				case errChan <- serveErr:
				default:
				}
			}
		}()

		fmt.Fprintf(os.Stderr, "To authorize %s, visit: %s\n", serverURL, args.URL)

		select {
		case <-fetchCtx.Done():
			_ = server.Shutdown(context.Background())
			return nil, fetchCtx.Err()
		case err := <-errChan:
			_ = server.Shutdown(context.Background())
			return nil, err
		case res := <-resChan:
			go func() {
				time.Sleep(500 * time.Millisecond)
				_ = server.Shutdown(context.Background())
			}()
			return res, nil
		}
	}

	handler, err := auth.NewAuthorizationCodeHandler(&auth.AuthorizationCodeHandlerConfig{
		Client:                   httpClient,
		RedirectURL:              redirectURI,
		InitialTokenSource:       initialTS,
		RequestRefreshToken:      true,
		AuthorizationCodeFetcher: fetcher,
		DynamicClientRegistrationConfig: &auth.DynamicClientRegistrationConfig{
			Metadata: &oauthex.ClientRegistrationMetadata{
				ClientName:   "gen",
				RedirectURIs: []string{redirectURI},
				GrantTypes:   []string{"authorization_code", "refresh_token"},
			},
		},
		NewTokenSource: func(tsCtx context.Context, cfg *oauth2.Config, tok *oauth2.Token) (oauth2.TokenSource, error) {
			_ = store.Put(serverURL, tok)
			return &cachingTokenSource{
				base:      cfg.TokenSource(tsCtx, tok),
				serverURL: serverURL,
				store:     store,
			}, nil
		},
	})
	if err != nil {
		return nil, err
	}

	return handler, nil
}
