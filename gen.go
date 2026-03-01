package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"google.golang.org/genai"
)

type Generator struct {
	ctx      context.Context
	params   *Parameters
	keyVals  ParamMap
	client   *genai.Client
	in       io.Reader
	out      io.Writer
	parts    []*genai.Part
	sysParts []*genai.Part
	schema   map[string]any
}

func genContent(ctx context.Context, in io.Reader, out io.Writer) error {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return fmt.Errorf("missing params")
	}
	if !params.ChatMode {
		var genCancel context.CancelFunc
		ctx, genCancel = context.WithTimeout(ctx, params.Timeout)
		defer genCancel()
	}
	g, err := newGenerator(ctx, in, out)
	if err != nil {
		return err
	}
	return g.run()
}

func newGenerator(ctx context.Context, in io.Reader, out io.Writer) (*Generator, error) {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return nil, fmt.Errorf("missing params")
	}
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
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

	if g.params.Segment {
		return g.segmentImage() // exit after segmentation
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

	// remove any uploaded media assets on exit
	if len(g.params.FilePaths) > 0 {
		var mediaAssets []string
		for _, p := range g.parts {
			if p.FileData != nil {
				mediaAssets = append(mediaAssets, p.FileData.FileURI)
			}
		}
		if len(mediaAssets) > 0 {
			defer func(verbose bool) {
				for _, fileURI := range mediaAssets {
					_, err := g.client.Files.Delete(g.ctx, fileURI, nil)
					if err != nil {
						fmt.Fprintf(os.Stderr, "failed to delete %s\n", fileURI)
						continue
					}
					if verbose {
						fmt.Fprintf(os.Stderr, infos("%s deleted\n"), fileURI)
					}
				}
			}(g.params.Verbose)
		}
	}

	// handle token count
	if g.params.TokenCount {
		defer func() {
			if g.params.TokenCount && g.ctx.Err() == nil {
				fmt.Fprintf(g.out, tokens("%d tokens")+"\n", TokenCount.Load())
			}
		}()
	}

	config := genai.GenerateContentConfig{
		Temperature: genai.Ptr(float32(g.params.Temp)),
		TopP:        genai.Ptr(float32(g.params.TopP)),
	}
	if err := g.buildConfig(&config); err != nil {
		return err
	}

	return g.generateContent(&config)
}

func (g *Generator) emitModelDetails() error {
	backend := "GeminiAPI"
	if g.client.ClientConfig().Backend == genai.BackendVertexAI {
		backend = "VertexAI"
	}

	name := g.params.GenModel
	if g.params.Segment {
		if !isFlagSet("m") {
			name = g.params.SegModel
		}
		fmt.Fprintf(os.Stderr, infos("%s backend | %s | -/- in/out token limit | %s\n\n"),
			backend, name, g.params.ThinkingLevel)
		return nil
	}

	var m *genai.Model
	var err error
	if (g.params.Embed || len(g.params.DigestPaths) > 0) && !isFlagSet("m") {
		m, err = g.client.Models.Get(g.ctx, g.params.EmbModel, nil)
	} else {
		m, err = g.client.Models.Get(g.ctx, g.params.GenModel, nil)
	}
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

		// handle files
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
	}

	return nil
}

func (g *Generator) segmentImage() error {
	var img *genai.Image
	var err error

	if strings.HasPrefix(g.params.FilePaths[0], "gs://") {
		img = &genai.Image{GCSURI: g.params.FilePaths[0]}
	} else {
		img, err = loadImage(g.params.FilePaths[0])
		if err != nil {
			return err
		}
	}

	res, err := g.client.Models.SegmentImage(g.ctx, g.params.SegModel,
		&genai.SegmentImageSource{
			Image: img,
		},
		&genai.SegmentImageConfig{
			Mode: genai.SegmentModeForeground,
		},
	)
	if err != nil {
		return err
	}

	for _, gim := range res.GeneratedMasks {
		if err = emitImage(g.out, gim.Mask); err != nil {
			return err
		}
		for _, el := range gim.Labels {
			fmt.Fprintln(g.out, el.Label)
		}
	}

	return nil
}

func (g *Generator) saveEmbeddings() error {
	res, err := g.client.Models.EmbedContent(g.ctx, g.params.EmbModel, []*genai.Content{{Parts: g.parts}}, nil)
	if err != nil {
		return err
	}
	if err := appendToDigest(g.params.DigestPaths[0], res.Embeddings[0], g.keyVals, g.params.OnlyKvs, g.params.Verbose, g.parts...); err != nil {
		return err
	}
	return nil
}

func (g *Generator) searchDigests() error {
	var res []QueryResult
	for _, digestPathVal := range g.params.DigestPaths {
		query, err := g.client.Models.EmbedContent(g.ctx, g.params.EmbModel, []*genai.Content{{Parts: g.parts}}, nil)
		if err != nil {
			return err
		}
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
	if g.params.JSON {
		config.ResponseMIMEType = "application/json"
		if g.schema != nil {
			config.ResponseJsonSchema = g.schema
		}
	}
	if g.params.Tool {
		// register tools with genai.FunctionCallingConfigModeAny
		config.Tools = []*genai.Tool{}
		if err = registerGenTools(config); err != nil { // declared in the tools.go file
			return err
		}
		if err = registerMCPTools(g.ctx, config); err != nil { // declared with -mcp
			return err
		}
		conjTexts(&g.parts)
	}
	if g.params.CodeGen {
		config.Tools =
			[]*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}}
	}
	if g.params.GoogleSearch {
		config.Tools = []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
			{URLContext: &genai.URLContext{}},
		}
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
			Role:  "model",
		}
		if g.params.Verbose {
			emitContent(os.Stderr, config.SystemInstruction, false, true, 0)
		}
	}
	if g.params.ThinkingLevel != genai.ThinkingLevelUnspecified {
		config.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   genai.ThinkingLevel(g.params.ThinkingLevel),
		}
	}

	return nil
}

func (g *Generator) generateContent(config *genai.GenerateContentConfig) error {
	var err error

	// retrieve previous session, if any
	history := []*genai.Content{}
	if g.params.ChatMode {
		if err = retrieveHistory(&history); err != nil {
			return err
		}
		if g.params.Verbose {
			emitHistory(os.Stderr, history)
		}
	}

	chat, err := g.client.Chats.Create(g.ctx, g.params.GenModel, config, history)
	if err != nil {
		return err
	}

	tty := g.in // assume in is terminal for chat

	if !g.params.Interactive && g.params.ChatMode {
		// in is a redirect, look for a terminal to open
		tty, err = openConsole()
		if err != nil {
			return err
		}
	}

	// main chat loop
	var onceOnly bool
	var addedFunRes bool
	for {
		if len(g.parts) > 0 {
			i := 0
			addedFunRes = false
			for resp, err := range chat.SendStream(g.ctx, g.parts...) {
				if err != nil {
					fmt.Fprintf(g.out, "\n")
					return err
				}
				if !onceOnly {
					if !g.params.ChatMode {
						onceOnly = true
					}
					if res := processFunCalls(g.ctx, resp); len(res) > 0 {
						resp = &genai.GenerateContentResponse{
							Candidates: []*genai.Candidate{
								&genai.Candidate{
									Content: &genai.Content{
										Parts: res,
									},
									Index: 0,
								},
							},
						}
						addedFunRes = true
						g.parts = append(g.parts, res...)
					}
				}
				if err := emitCandidates(g.out, resp.Candidates, g.params.ImgModality, g.params.Verbose, addedFunRes, i); err != nil {
					fmt.Fprintf(g.out, "\n")
					return err
				}
				if g.params.TokenCount && resp.UsageMetadata != nil {
					TokenCount.Store(resp.UsageMetadata.TotalTokenCount)
				}
				i += 1
			} // for range SendStream
		}
		// exit if not a chat and no function response to process
		if !g.params.ChatMode && !addedFunRes { //len(parts) == 0 {
			break
		}
		if g.params.ChatMode && !addedFunRes {
			input, err := readLine(tty)
			if err != nil {
				return err
			}
			// check for double blank line exit condition
			if input == "" {
				input, err = readLine(tty)
				if err != nil {
					return err
				}
				if input == "" {
					break // exit chat mode
				}
			}
			if isRedirected(g.out) {
				fmt.Fprintf(g.out, "\n%s\n\n", input)
			}
			g.parts = append(g.parts, &genai.Part{Text: input})
		}
	}

	if g.params.ChatMode {
		if err = persistChat(chat); err != nil {
			fmt.Fprintf(g.out, "\n")
			return err
		}
	}

	return nil
}
