package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"google.golang.org/genai"
)

// emitGen is the main gen content generator.
func emitGen(ctx context.Context, in io.Reader, out io.Writer) int {
	var genCtx context.Context
	var genCancel context.CancelFunc
	var err error
	var parts []*genai.Part
	var sysParts []*genai.Part
	var stdinData []byte
	var mediaAssets []string
	var schema map[string]any

	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		genLogFatal(fmt.Errorf("emitGen: params not found in context"))
	}
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		genLogFatal(fmt.Errorf("emitGen: keyVals not found in context"))
	}

	// Handle redirect/piped data
	if !params.Interactive {
		stdinData, err = io.ReadAll(in)
		if err != nil {
			genLogFatal(err)
		}
		params.Interactive = len(stdinData) == 0 // ignore redirect
	}

	// Create a genai client
	if !params.ChatMode {
		genCtx, genCancel = context.WithTimeout(ctx, params.Timeout)
		defer genCancel()
	} else {
		genCtx = ctx
	}
	client, err := genai.NewClient(genCtx, nil)
	if err != nil {
		genLogFatal(err)
	}

	// Handle verbose parameter
	if params.Verbose {
		var backend string
		if client.ClientConfig().Backend == genai.BackendVertexAI {
			backend = "VertexAI"
		} else {
			backend = "GeminiAPI"
		}
		if params.Segment {
			var m string
			if !isFlagSet("m") {
				m = params.SegModel
			} else {
				m = params.GenModel
			}
			fmt.Fprintf(os.Stderr, "\033[36m%s backend | %s | -/- in/out token limit | %s\033[0m\n\n",
				backend, m, params.ThinkingLevel)
		} else {
			var m *genai.Model
			if (params.Embed || len(params.DigestPaths) > 0) && !isFlagSet("m") {
				m, err = client.Models.Get(genCtx, params.EmbModel, nil)
			} else {
				m, err = client.Models.Get(genCtx, params.GenModel, nil)
			}
			if err != nil {
				genLogFatal(err)
			}
			fmt.Fprintf(os.Stderr, "\033[36m%s backend | %s | %d/%d in/out token limit | %s\033[0m\n\n",
				backend, m.Name, m.InputTokenLimit, m.OutputTokenLimit, params.ThinkingLevel)
		}
	}

	// Handle segmentation
	if params.Segment {
		var img *genai.Image
		if strings.HasPrefix(params.FilePaths[0], "gs://") {
			img = &genai.Image{GCSURI: params.FilePaths[0]}
		} else {
			img, err = loadImage(params.FilePaths[0])
			if err != nil {
				genLogFatal(err)
			}
		}
		res, err := client.Models.SegmentImage(genCtx, params.SegModel,
			&genai.SegmentImageSource{
				Image: img,
			},
			&genai.SegmentImageConfig{
				Mode: genai.SegmentModeForeground,
			},
		)
		if err != nil {
			genLogFatal(err)
		}
		for _, gim := range res.GeneratedMasks {
			if err = emitImage(out, gim.Mask); err != nil {
				genLogFatal(err)
			}
			for _, el := range gim.Labels {
				fmt.Fprintln(out, el.Label)
			}
		}
		return 0
	}

	// First, handle argument
	if len(params.Args) > 0 {
		text := searchReplace(strings.Join(params.Args, " "), keyVals)
		if !params.Interactive && text == "-" {
			text = string(stdinData)
		}
		if params.SystemInstruction && (params.Interactive || !oneMatches(params.FilePaths, "-")) {
			// argument used as system prompt for chat session unless `-f -` is set
			sysParts = append(sysParts, &genai.Part{Text: text})
		} else {
			parts = append(parts, &genai.Part{Text: text})
		}
	}

	// Next, handle file options
	if len(params.FilePaths) > 0 {
		for _, filePathVal := range params.FilePaths {
			// redirect passed as file
			if filePathVal == "-" {
				if params.SystemInstruction {
					// `-f -` takes precedence over argument with `-s`
					sysParts = append(sysParts, &genai.Part{Text: searchReplace(string(stdinData), keyVals)})
				} else {
					parts = append(parts, &genai.Part{Text: searchReplace(string(stdinData), keyVals)})
				}
				continue
			}
			// possible uploads include regular file, json schema, .prompt, .sprompt or directory
			if err = glob(genCtx, client, filePathVal, &parts, &sysParts, &schema); err != nil {
				genLogFatal(err)
			}
		}
	}

	// Handle token count
	if params.TokenCount {
		defer func() {
			if params.TokenCount && ctx.Err() == nil {
				fmt.Fprintf(out, "\033[31m%d tokens\033[0m\n", TokenCount.Load())
			}
		}()
	}

	// Handle embed parameter then exit
	if params.Embed {
		res, err := client.Models.EmbedContent(genCtx, params.EmbModel, []*genai.Content{{Parts: parts}}, nil)
		if err != nil {
			genLogFatal(err)
		}
		if err := appendToDigest(params.DigestPaths[0], res.Embeddings[0], keyVals, params.OnlyKvs, params.Verbose, parts...); err != nil {
			genLogFatal(err)
		}
		return 0
	}

	// Handle digest parameter and retrieve text from digest
	if len(params.DigestPaths) > 0 {
		var res []QueryResult
		for _, digestPathVal := range params.DigestPaths {
			query, err := client.Models.EmbedContent(genCtx, params.EmbModel, []*genai.Content{{Parts: parts}}, nil)
			if err != nil {
				genLogFatal(err)
			}
			res, err = queryDigest(digestPathVal, query.Embeddings[0], res, params.K, float32(params.Lambda), params.Verbose)
			if err != nil {
				genLogFatal(err)
			}
		}
		if len(res) > 0 {
			// inject digest into a prompt or append as text
			if idx := partWithKey(sysParts, DigestKey); idx != -1 {
				replacePart(&sysParts, idx, DigestKey, res)
			} else if idx := partWithKey(parts, DigestKey); idx != -1 {
				replacePart(&parts, idx, DigestKey, res)
			} else {
				prependToParts(&parts, res)
			}
		}
	}

	// Set temperature and top_p from args or model defaults
	config := &genai.GenerateContentConfig{
		Temperature: genai.Ptr(float32(params.Temp)),
		TopP:        genai.Ptr(float32(params.TopP)),
	}
	// Handle modality
	if params.ImgModality {
		config.ResponseModalities = []string{"TEXT", "IMAGE"}
	} else {
		config.ResponseModalities = []string{"TEXT"}
	}
	// Handle json parameter
	if params.JSON {
		config.ResponseMIMEType = "application/json"
		if schema != nil {
			config.ResponseJsonSchema = schema
		}
	}
	// Register tools with genai.FunctionCallingConfigModeAny
	if params.Tool {
		config.Tools = []*genai.Tool{}
		if err = registerGenTools(config); err != nil { // declared in the tools.go file
			genLogFatal(err)
		}
		if err = registerMCPTools(genCtx, config); err != nil { // declared with -mcp
			genLogFatal(err)
		}
		conjTexts(&parts)
	}
	// Allow code execution
	if params.CodeGen {
		config.Tools =
			[]*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}}
	}
	// Enable Google search retrieval
	if params.GoogleSearch {
		config.Tools = []*genai.Tool{
			{GoogleSearch: &genai.GoogleSearch{}},
			{URLContext: &genai.URLContext{}},
		}
	}
	// Handle unsafe parameter
	if params.Unsafe {
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

	// Set system instruction
	if len(sysParts) > 0 {
		config.SystemInstruction = &genai.Content{
			Parts: sysParts,
			Role:  "model",
		}
		if params.Verbose {
			emitContent(os.Stderr, config.SystemInstruction, false, true, 0)
		}
	}

	// Handle thinking level
	if params.ThinkingLevel != genai.ThinkingLevelUnspecified {
		// add thought summaries
		config.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingLevel:   genai.ThinkingLevel(params.ThinkingLevel),
		}
	}

	// retrieve previous session, if any
	history := []*genai.Content{}
	if params.ChatMode {
		if err = retrieveHistory(&history); err != nil {
			genLogFatal(err)
		}
		if params.Verbose {
			emitHistory(os.Stderr, history)
		}
	}

	// Start chat
	chat, err := client.Chats.Create(genCtx, params.GenModel, config, history)
	if err != nil {
		genLogFatal(err)
	}

	tty := in // assume in is terminal for chat

	if !params.Interactive && params.ChatMode {
		// in is a redirect, look for a terminal to open
		tty, err = openConsole()
		if err != nil {
			genLogFatal(err)
		}
	}

	// Remove any uploaded media assets on exit
	if len(params.FilePaths) > 0 {
		for _, p := range parts {
			if p.FileData != nil {
				mediaAssets = append(mediaAssets, p.FileData.FileURI)
			}
		}
		if len(mediaAssets) > 0 {
			defer func(verbose bool) {
				for _, fileURI := range mediaAssets {
					_, err := client.Files.Delete(genCtx, fileURI, nil)
					if err != nil {
						genLogFatal(err)
					}
					if verbose {
						fmt.Fprintf(os.Stderr, "\033[36m%s deleted\033[0m\n", fileURI)
					}
				}
			}(params.Verbose)
		}
	}

	// Main chat loop
	var onceOnly bool
	var addedFunRes bool
	for {
		if len(parts) > 0 {
			i := 0
			//sendParts := parts
			//parts = []*genai.Part{} // emtpy parts for next iteration
			addedFunRes = false
			//for resp, err := range chat.SendStream(genCtx, sendParts...) {
			for resp, err := range chat.SendStream(genCtx, parts...) {
				if err != nil {
					fmt.Fprintf(out, "\n")
					genLogFatal(err)
				}
				if !onceOnly {
					if !params.ChatMode {
						onceOnly = true
					}
					if res := processFunCalls(genCtx, resp); len(res) > 0 {
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
						//parts = append(parts, sendParts...) // send original parts back
						parts = append(parts, res...)
					}
				}
				if err := emitCandidates(out, resp.Candidates, params.ImgModality, params.Verbose, addedFunRes, i); err != nil {
					fmt.Fprintf(out, "\n")
					genLogFatal(err)
				}
				if params.TokenCount && resp.UsageMetadata != nil {
					TokenCount.Store(resp.UsageMetadata.TotalTokenCount)
				}
				i += 1
			} // for range SendStream
		}
		// exit if not a chat and no function response to process
		if !params.ChatMode && !addedFunRes { //len(parts) == 0 {
			break
		}
		if params.ChatMode && !addedFunRes {
			input, err := readLine(tty)
			if err != nil {
				genLogFatal(err)
			}
			// check for double blank line exit condition
			if input == "" {
				input, err = readLine(tty)
				if err != nil {
					genLogFatal(err)
				}
				if input == "" {
					break // exit chat mode
				}
			}
			if isRedirected(out) {
				fmt.Fprintf(out, "\n%s\n\n", input)
			}
			parts = append(parts, &genai.Part{Text: input})
		}
	}

	if params.ChatMode {
		if err = persistChat(chat); err != nil {
			fmt.Fprintf(out, "\n")
			genLogFatal(err)
		}
	}

	return 0
}
