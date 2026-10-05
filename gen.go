package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

type Generator struct {
	ctx      context.Context
	params   *core.Parameters
	keyVals  core.ParamMap
	client   *genai.Client
	in       io.Reader
	out      io.Writer
	parts    []*genai.Part
	sysParts []*genai.Part
	schema   map[string]any
}

func genContent(ctx context.Context, in io.Reader, out io.Writer) error {
	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return fmt.Errorf("missing params")
	}
	if !params.ChatMode {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, params.Timeout)
		defer cancel()
	}
	g, err := newGenerator(ctx, in, out)
	if err != nil {
		return err
	}
	return g.run()
}

func newGenerator(ctx context.Context, in io.Reader, out io.Writer) (*Generator, error) {
	params, ok := ctx.Value(core.ParamsKey).(*core.Parameters)
	if !ok {
		return nil, fmt.Errorf("missing params")
	}
	keyVals, ok := ctx.Value(core.KeyValsKey).(core.ParamMap)
	if !ok {
		return nil, fmt.Errorf("missing keyVals")
	}

	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		return nil, err
	}

	return &Generator{
		ctx:     ctx,
		params:  params,
		keyVals: keyVals,
		client:  client,
		in:      in,
		out:     out,
	}, nil
}

func (g *Generator) run() error {
	if g.params.Verbose {
		if err := g.emitModelDetails(); err != nil {
			return err
		}
	}

	if err := g.setPromptsAndFiles(); err != nil {
		return err
	}

	if g.params.Embed {
		return g.saveEmbeddings() // exit after embeddings added
	}

	if len(g.params.DigestPaths) > 0 {
		if err := g.searchDigests(); err != nil {
			return err
		}
	}

	config := genai.GenerateContentConfig{
		Temperature: genai.Ptr(float32(g.params.Temp)),
		TopP:        genai.Ptr(float32(g.params.TopP)),
	}
	if err := g.buildConfig(&config); err != nil {
		return err
	}

	if g.params.NoOp {
		if g.params.CountTokens {
			return g.countTokensNoOp(&config)
		}
		return nil
	}

	if err := g.generateContent(&config); err != nil {
		return err
	}

	if g.params.CountTokens {
		g.emitUsageSummary()
	}

	return nil
}

func (g *Generator) emitModelDetails() error {
	backend := "GeminiAPI"
	if g.client.ClientConfig().Backend == genai.BackendVertexAI {
		backend = "VertexAI"
	}
	modelName := g.params.GenModel
	if g.params.Embed {
		modelName = g.params.EmbModel
	}
	m, err := g.client.Models.Get(g.ctx, modelName, nil)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, infos("%s backend | %s | %d/%d in/out token limit | %s\n\n"),
		backend, m.Name, m.InputTokenLimit, m.OutputTokenLimit, g.params.ThinkingLevel)
	return nil
}

func (g *Generator) setPromptsAndFiles() error {
	var stdinData []byte
	var err error

	// handle redirect/piped data
	if !g.params.Interactive {
		stdinData, err = io.ReadAll(g.in)
		if err != nil {
			return err
		}
		g.params.Interactive = len(stdinData) == 0 // ignore redirect
	}

	// handle prompts from argument
	if len(g.params.Args) > 0 {
		text := searchReplace(strings.Join(g.params.Args, " "), g.keyVals)
		if !g.params.Interactive && text == "-" {
			text = string(stdinData)
		}
		if g.params.SystemInstruction && (g.params.Interactive || !oneMatches(g.params.FilePaths, "-")) {
			// argument used as system prompt for chat session unless `-f -` is set
			g.sysParts = append(g.sysParts, &genai.Part{Text: text})
		} else {
			g.parts = append(g.parts, &genai.Part{Text: text})
		}
	}

	// handle files
	if len(g.params.FilePaths) > 0 {
		for _, filePathVal := range g.params.FilePaths {
			// case of redirect passed as file
			if filePathVal == "-" {
				if g.params.SystemInstruction {
					// `-f -` takes precedence over `-s`
					g.sysParts = append(g.sysParts, &genai.Part{Text: searchReplace(string(stdinData), g.keyVals)})
				} else {
					g.parts = append(g.parts, &genai.Part{Text: searchReplace(string(stdinData), g.keyVals)})
				}
				continue
			}
			// case of regular file, json schema, .prompt, .sprompt or directory
			if err = glob(g.ctx, g.client, filePathVal, &g.parts, &g.sysParts, &g.schema); err != nil {
				return err
			}
		}
		// stash URIs of FileData parts for removal
		for _, p := range g.parts {
			if p.FileData != nil {
				g.params.FileURIs = append(g.params.FileURIs, p.FileData.FileURI)
			}
		}
	}

	return nil
}

func (g *Generator) saveEmbeddings() error {
	chunks := extractChunks(g.parts)
	if len(chunks) == 0 {
		return fmt.Errorf("no content found to embed")
	}

	contents := make([]*genai.Content, len(chunks))
	for i, chunk := range chunks {
		contents[i] = &genai.Content{
			Parts: []*genai.Part{
				{Text: chunk},
			},
		}
	}

	var allEmbeddings []*genai.ContentEmbedding
	for i := 0; i < len(contents); i += g.params.MaxEmbedBatchSize {
		end := min(i+g.params.MaxEmbedBatchSize, len(contents))
		batch := contents[i:end]

		res, err := g.client.Models.EmbedContent(g.ctx, g.params.EmbModel, batch, nil)
		if err != nil {
			return err
		}
		if len(res.Embeddings) != len(batch) {
			return fmt.Errorf("expected %d embeddings, got %d", len(batch), len(res.Embeddings))
		}
		allEmbeddings = append(allEmbeddings, res.Embeddings...)
	}

	if err := appendToDigest(g.params.DigestPaths[0], allEmbeddings, chunks, g.keyVals, g.params.OnlyKvs, g.params.Verbose); err != nil {
		return err
	}
	return nil
}

func (g *Generator) searchDigests() error {
	query, err := g.client.Models.EmbedContent(g.ctx, g.params.EmbModel, []*genai.Content{{Parts: g.parts}}, nil)
	if err != nil {
		return err
	}
	var res []QueryResult
	for _, digestPathVal := range g.params.DigestPaths {
		res, err = queryDigest(digestPathVal, query.Embeddings[0], res, g.params.K, float32(g.params.Lambda), g.params.Verbose)
		if err != nil {
			return err
		}
	}
	if len(res) > 0 {
		// inject digest into a prompt or append as text
		if idx := partWithKey(g.sysParts, DigestKey); idx != -1 {
			replacePart(&g.sysParts, idx, DigestKey, res)
		} else if idx := partWithKey(g.parts, DigestKey); idx != -1 {
			replacePart(&g.parts, idx, DigestKey, res)
		} else {
			prependToParts(&g.parts, res)
		}
	}
	return nil
}

func (g *Generator) buildConfig(config *genai.GenerateContentConfig) error {
	var err error

	if g.params.ImgModality {
		config.ResponseModalities = []string{"TEXT", "IMAGE"}
	} else {
		config.ResponseModalities = []string{"TEXT"}
	}
	if g.params.JSON && !g.params.Tool {
		config.ResponseMIMEType = "application/json"
		if g.schema != nil {
			config.ResponseJsonSchema = g.schema
		}
	}
	if g.params.Tool {
		if err = registerGenTools(config); err != nil { // see tools.go
			return err
		}
		if err = registerMCPTools(g.ctx, config); err != nil { // declared with -mcp
			return err
		}
		conjTexts(&g.parts)
	}
	if g.params.CodeGen {
		config.Tools = append(config.Tools, &genai.Tool{CodeExecution: &genai.ToolCodeExecution{}})
	}
	if g.params.GoogleSearch {
		config.Tools = append(config.Tools,
			&genai.Tool{GoogleSearch: &genai.GoogleSearch{}},
			&genai.Tool{URLContext: &genai.URLContext{}},
		)
	}
	if g.params.Unsafe {
		config.SafetySettings = []*genai.SafetySetting{
			{
				Category:  genai.HarmCategoryDangerousContent,
				Threshold: genai.HarmBlockThresholdBlockNone,
			},
			{
				Category:  genai.HarmCategoryHarassment,
				Threshold: genai.HarmBlockThresholdBlockNone,
			},
			{
				Category:  genai.HarmCategoryHateSpeech,
				Threshold: genai.HarmBlockThresholdBlockNone,
			},
			{
				Category:  genai.HarmCategorySexuallyExplicit,
				Threshold: genai.HarmBlockThresholdBlockNone,
			},
		}
	}
	if len(g.sysParts) > 0 {
		config.SystemInstruction = &genai.Content{
			Parts: g.sysParts,
		}
		if g.params.Verbose {
			_ = emitContent(os.Stderr, config.SystemInstruction, false, false, true, nil, nil, "")
		}
	}
	if g.params.ThinkingLevel != genai.ThinkingLevelUnspecified {
		config.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   genai.ThinkingLevel(g.params.ThinkingLevel),
		}
	}
	if g.params.Seed.Defined && g.params.Seed.Value != nil {
		config.Seed = g.params.Seed.Value
	}
	if g.params.PresencePenalty.Defined && g.params.PresencePenalty.Value != nil {
		config.PresencePenalty = g.params.PresencePenalty.Value
	}
	if g.params.FrequencyPenalty.Defined && g.params.FrequencyPenalty.Value != nil {
		config.FrequencyPenalty = g.params.FrequencyPenalty.Value
	}
	if len(g.params.StopSequences) > 0 {
		config.StopSequences = g.params.StopSequences
	}

	return nil
}

func (g *Generator) generateContent(config *genai.GenerateContentConfig) error {
	var chat *genai.Chat
	var userAcc []*genai.Part
	var modelAcc []*genai.Part
	var inlDatAcc []*genai.Part
	var turnUsage core.TokenUsage
	var err error

	// retrieve previous session, if any
	history := []*genai.Content{}
	if g.params.ChatMode {
		if err = retrieveHistory(&history); err != nil {
			return err
		}
		if len(history) > 0 {
			fmt.Fprintf(g.out, important("%s found\n"), DotGen)
			if g.params.Verbose {
				emitHistory(os.Stderr, history)
			}
		}
	}

	tty := g.in // assume in is terminal for chat

	if !g.params.Interactive && g.params.ChatMode {
		// in is a redirect, look for a terminal to open
		tty, err = openConsole()
		if err != nil {
			return err
		}
	}

	// main interaction loop
	for {
		chat, err = g.client.Chats.Create(g.ctx, g.params.GenModel, config, history)
		if err != nil {
			return err
		}
		if len(g.parts) > 0 {
			i := 0
			userAcc = g.parts
			streamAcc := []*genai.Part{}
			g.parts = []*genai.Part{}
			funCalls := []*genai.FunctionCall{}
			var textBuilder strings.Builder
			var thoughtBuilder strings.Builder
			var sig []byte
			mp := &MarkdownParser{}

			var streamUsage *genai.GenerateContentResponseUsageMetadata
			for resp, err := range chat.SendStream(g.ctx, userAcc...) {
				if err != nil {
					fmt.Fprintf(g.out, "\n")
					return err
				}

				if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
					partsRetained := []*genai.Part{}
					for _, p := range resp.Candidates[0].Content.Parts {
						if !isValidPart(p) {
							continue
						}
						if p.Thought {
							thoughtBuilder.WriteString(p.Text)
							if len(p.ThoughtSignature) > 0 {
								sig = p.ThoughtSignature
							}
						} else if p.Text != "" {
							textBuilder.WriteString(p.Text)
						} else if p.InlineData != nil {
							isDupe := false
							for _, existing := range inlDatAcc {
								if existing.InlineData != nil && bytes.Equal(existing.InlineData.Data, p.InlineData.Data) {
									isDupe = true
									break
								}
							}
							if !isDupe {
								inlDatAcc = append(inlDatAcc, p)
								streamAcc = append(streamAcc, p)
							}
							continue
						} else {
							streamAcc = append(streamAcc, p)
						}
						partsRetained = append(partsRetained, p)
					}
					resp.Candidates[0].Content.Parts = partsRetained
				}

				if fc := resp.FunctionCalls(); len(fc) > 0 {
					funCalls = append(funCalls, fc...)
				}
				if len(funCalls) == 0 {
					if len(resp.Candidates) > 0 {
						err := emitCandidate(g.out, resp.Candidates[0], g.params.OutRedirected, g.params.ImgModality, g.params.Verbose, &i, mp, g.params.OutPath)
						if err != nil {
							fmt.Fprintf(g.out, "\n")
							return err
						}
						for mp != nil && mp.imageFound {
							mp.imageFound = false
							if len(inlDatAcc) > 0 {
								img := inlDatAcc[0]
								inlDatAcc = inlDatAcc[1:]
								imgContent := &genai.Content{
									Parts: []*genai.Part{img},
								}
								_ = emitContent(g.out, imgContent, g.params.OutRedirected, g.params.ImgModality, g.params.Verbose, &i, nil, g.params.OutPath)
							}
							if !g.params.OutRedirected {
								// drain what's left in mp's buffer
								fmt.Fprint(g.out, mp.parse(""))
							}
						}
					}
				}

				if resp.UsageMetadata != nil {
					streamUsage = resp.UsageMetadata
				}
			}

			if streamUsage != nil {
				stepUsage := &core.TokenUsage{
					PromptTokens:        streamUsage.PromptTokenCount,
					CandidatesTokens:    streamUsage.CandidatesTokenCount,
					ThoughtsTokens:      streamUsage.ThoughtsTokenCount,
					CachedContentTokens: streamUsage.CachedContentTokenCount,
					ToolUsePromptTokens: streamUsage.ToolUsePromptTokenCount,
					TotalTokens:         streamUsage.TotalTokenCount,
				}
				g.params.Usage.Add(stepUsage)
				turnUsage.Add(stepUsage)
			}

			modelAcc = []*genai.Part{}
			if thoughtBuilder.Len() > 0 {
				modelAcc = append(modelAcc, &genai.Part{
					Text:             thoughtBuilder.String(),
					Thought:          true,
					ThoughtSignature: sig,
				})
			}
			if textBuilder.Len() > 0 {
				modelAcc = append(modelAcc, &genai.Part{
					Text: textBuilder.String(),
				})
			}
			if len(streamAcc) > 0 {
				modelAcc = append(modelAcc, streamAcc...)
			}
			// fallback in case stream returns completely empty content
			if len(modelAcc) == 0 {
				modelAcc = append(modelAcc, &genai.Part{Text: " "})
			}

			history = append(history, &genai.Content{
				Role:  "user",
				Parts: userAcc,
			})
			history = append(history, &genai.Content{
				Role:  "model",
				Parts: modelAcc,
			})

			if len(funCalls) > 0 {
				resCand, err := processFunctionCalls(g.ctx, funCalls)
				if err != nil {
					return err
				}
				if len(resCand.Content.Parts) > 0 {
					displayCand := &genai.Candidate{
						Content: &genai.Content{
							Role: resCand.Content.Role,
						},
						FinishReason: resCand.FinishReason,
					}
					for _, p := range resCand.Content.Parts {
						if p.FunctionResponse != nil {
							var partsRetained []*genai.FunctionResponsePart
							for _, frp := range p.FunctionResponse.Parts {
								if frp.InlineData != nil && len(frp.InlineData.Data) > 0 && p.FunctionResponse.Name != "CaptureScreen" {
									isDupe := false
									for _, existing := range inlDatAcc {
										if existing.InlineData != nil && bytes.Equal(existing.InlineData.Data, frp.InlineData.Data) {
											isDupe = true
											break
										}
									}
									if !isDupe {
										inlDatAcc = append(inlDatAcc, &genai.Part{
											InlineData: &genai.Blob{
												DisplayName: frp.InlineData.DisplayName,
												Data:        frp.InlineData.Data,
												MIMEType:    frp.InlineData.MIMEType,
											},
										})
									}
								} else {
									partsRetained = append(partsRetained, frp)
								}
							}
							clonedFR := *p.FunctionResponse
							clonedFR.Parts = partsRetained
							displayCand.Content.Parts = append(displayCand.Content.Parts, &genai.Part{
								FunctionResponse: &clonedFR,
							})
						} else {
							displayCand.Content.Parts = append(displayCand.Content.Parts, p)
						}
					}
					err := emitCandidate(g.out, displayCand, g.params.OutRedirected, g.params.ImgModality, g.params.Verbose, &i, mp, g.params.OutPath)
					if err != nil {
						fmt.Fprintf(g.out, "\n")
						return err
					}
					if mp != nil {
						fmt.Fprint(g.out, mp.flush(g.params.OutRedirected))
					}
					if g.params.JSON {
						// switch to JSON for final response generation AND disable tools so Gemini doesn't loop
						config.ResponseMIMEType = "application/json"
						if g.schema != nil {
							config.ResponseJsonSchema = g.schema
						}
						config.Tools = nil
					}
					// carry forward function response to next iteration
					g.parts = append(g.parts, resCand.Content.Parts...)
					continue
				}
			}

			if mp != nil {
				fmt.Fprint(g.out, mp.flush(g.params.OutRedirected))
			}

			if len(inlDatAcc) > 0 {
				imgContent := &genai.Content{
					Parts: inlDatAcc,
				}
				if err := emitContent(g.out, imgContent, g.params.OutRedirected, g.params.ImgModality, g.params.Verbose, &i, mp, g.params.OutPath); err != nil {
					return err
				}
				inlDatAcc = []*genai.Part{}
			}
		}

		// exit if not a chat
		if !g.params.ChatMode {
			break
		}

		if g.params.CountTokens && turnUsage.TotalTokens > 0 {
			dest := g.out
			if g.params.OutRedirected {
				dest = os.Stderr
			}
			msg := formatTokenUsage(&turnUsage)
			if g.params.OutRedirected {
				fmt.Fprintf(dest, "\n%s\n", msg)
			} else {
				fmt.Fprintf(dest, "\n%s\n", important(msg))
			}
			turnUsage = core.TokenUsage{}
		}

		input, err := readLine(tty)
		if err != nil {
			return err
		}
		// check for double blank line
		if input == "" {
			input, err = readLine(tty)
			if err != nil {
				return err
			}
			if input == "" {
				break // exit chat mode
			}
		}
		if g.params.OutRedirected {
			fmt.Fprintf(g.out, "\n%s\n\n", input)
		}

		g.parts = append(g.parts, &genai.Part{Text: input})
	} // end main interaction loop

	if g.params.ChatMode {
		if err = persistChat(history); err != nil {
			fmt.Fprintf(g.out, "\n")
			return err
		}
	}

	return nil
}

func (g *Generator) countTokensNoOp(config *genai.GenerateContentConfig) error {
	contents := []*genai.Content{}
	if len(g.parts) > 0 {
		contents = append(contents, &genai.Content{
			Role:  "user",
			Parts: g.parts,
		})
	}

	countConfig := &genai.CountTokensConfig{
		SystemInstruction: config.SystemInstruction,
		Tools:             config.Tools,
	}

	resp, err := g.client.Models.CountTokens(g.ctx, g.params.GenModel, contents, countConfig)
	if err != nil {
		return err
	}

	g.params.Usage = core.TokenUsage{
		PromptTokens:        resp.TotalTokens,
		TotalTokens:         resp.TotalTokens,
		CachedContentTokens: resp.CachedContentTokenCount,
	}

	if g.params.JSON {
		res := struct {
			Model               string `json:"model"`
			TotalTokens         int32  `json:"totalTokens"`
			PromptTokens        int32  `json:"promptTokens"`
			CachedContentTokens int32  `json:"cachedContentTokens,omitempty"`
		}{
			Model:               g.params.GenModel,
			TotalTokens:         resp.TotalTokens,
			PromptTokens:        resp.TotalTokens,
			CachedContentTokens: resp.CachedContentTokenCount,
		}
		data, err := json.MarshalIndent(res, "", "  ")
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(g.out, "%s\n", data)
		return err
	}

	msg := formatPreFlightUsage(resp.TotalTokens, resp.CachedContentTokenCount)
	if g.params.OutRedirected {
		_, err = fmt.Fprintf(g.out, "%s\n", msg)
	} else {
		_, err = fmt.Fprintf(g.out, "%s\n", important(msg))
	}
	return err
}

func (g *Generator) emitUsageSummary() {
	if g.params.Usage.TotalTokens == 0 {
		return
	}
	msg := formatTokenUsage(&g.params.Usage)
	if g.params.ChatMode {
		msg = "session total: " + msg
	}

	if g.params.OutRedirected || g.params.JSON {
		fmt.Fprintf(os.Stderr, "%s\n", important(msg))
	} else {
		fmt.Fprintf(g.out, "\n%s\n", important(msg))
	}
}

func formatTokenUsage(u *core.TokenUsage) string {
	if u.TotalTokens == 0 {
		return "0 tokens"
	}
	var details []string
	details = append(details, fmt.Sprintf("prompt: %d", u.PromptTokens))
	details = append(details, fmt.Sprintf("generation: %d", u.CandidatesTokens))
	if u.ThoughtsTokens > 0 {
		details = append(details, fmt.Sprintf("thoughts: %d", u.ThoughtsTokens))
	}
	if u.CachedContentTokens > 0 {
		details = append(details, fmt.Sprintf("cached: %d", u.CachedContentTokens))
	}
	if u.ToolUsePromptTokens > 0 {
		details = append(details, fmt.Sprintf("tool: %d", u.ToolUsePromptTokens))
	}
	return fmt.Sprintf("%d tokens (%s)", u.TotalTokens, strings.Join(details, ", "))
}

func formatPreFlightUsage(totalTokens, cachedTokens int32) string {
	if cachedTokens > 0 {
		return fmt.Sprintf("%d tokens (prompt: %d, cached: %d)", totalTokens, totalTokens, cachedTokens)
	}
	return fmt.Sprintf("%d tokens (prompt: %d)", totalTokens, totalTokens)
}
