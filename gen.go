package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"google.golang.org/genai"
)

// emitGen is the main gen content generator
func emitGen(ctx context.Context, in io.Reader, out io.Writer, params *Parameters) int {
	var err error
	var parts []*genai.Part
	var sysParts []*genai.Part
	var stdinData []byte
	var mediaAssets []string

	// Handle redirect/piped data
	if !params.Interactive {
		stdinData, err = io.ReadAll(in)
		if err != nil {
			genLogFatal(err)
		}
		params.Interactive = len(stdinData) == 0 // ignore redirect
	}

	// Create a genai client
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		genLogFatal(err)
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
			// possible uploads include regular file, .prompt, .sprompt or directory
			if err = glob(ctx, client, filePathVal, &parts, &sysParts, keyVals); err != nil {
				genLogFatal(err)
			}
		}
	}

	// Handle token count
	if params.TokenCount {
		defer func() {
			fmt.Fprintf(out, "\033[31m%d tokens\033[0m\n", tokenCount)
		}()
	}

	// Handle embed parameter then exit
	if params.Embed {
		res, err := client.Models.EmbedContent(ctx, params.EmbModel, []*genai.Content{{Parts: parts}}, nil)
		if err != nil {
			genLogFatal(err)
		}
		if err := AppendToDigest(params.DigestPaths[0], res.Embeddings[0], keyVals, params.OnlyKvs, params.Verbose, parts...); err != nil {
			genLogFatal(err)
		}
		return 0
	}

	// Handle digest parameter and retrieve text from digest
	if len(params.DigestPaths) > 0 {
		var res []QueryResult
		for _, digestPathVal := range params.DigestPaths {
			query, err := client.Models.EmbedContent(ctx, params.EmbModel, []*genai.Content{{Parts: parts}}, nil)
			if err != nil {
				genLogFatal(err)
			}
			res, err = QueryDigest(digestPathVal, query.Embeddings[0], res, params.K, float32(params.Lambda), params.Verbose)
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
	}
	// Register tools with genai.FunctionCallingConfigModeAny
	if params.Tool {
		config.Tools = []*genai.Tool{}
		registerGenTools(config)                                     // declared in the tools.go file
		if err = registerMcpTools(ctx, config, params); err != nil { // declared with -mcp
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
			fmt.Fprintf(os.Stderr, "\033[36m%+v\033[0m\n\n", *config)
		}
	}

	// Handle verbose parameter
	if params.Verbose {
		var m *genai.Model
		var backend string
		if (params.Embed || len(params.DigestPaths) > 0) && !isFlagSet("m") {
			m, err = client.Models.Get(ctx, params.EmbModel, nil)
		} else {
			m, err = client.Models.Get(ctx, params.GenModel, nil)
		}
		if err != nil {
			genLogFatal(err)
		}
		if client.ClientConfig().Backend == genai.BackendVertexAI {
			backend = "VertexAI"
		} else {
			backend = "GeminiAPI"
		}
		fmt.Fprintf(os.Stderr, "\033[36m%s | %s | %d | %d\033[0m\n\n", backend, m.Name, m.InputTokenLimit, m.OutputTokenLimit)
	}

	history := []*genai.Content{}
	if params.ChatMode {
		if err = retrieveHistory(&history); err != nil {
			genLogFatal(err)
		}
	}

	// Start chat
	chat, err := client.Chats.Create(ctx, params.GenModel, config, history)
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
			defer func() {
				for _, fileURI := range mediaAssets {
					_, err := client.Files.Delete(ctx, fileURI, nil)
					if err != nil {
						genLogFatal(err)
					}
				}
			}()
		}
	}

	// Main chat loop
	for {
		if len(parts) > 0 {
			for resp, err := range chat.SendMessageStream(ctx, derefParts(parts)...) {
				if err != nil {
					fmt.Fprintf(out, "\n")
					genLogFatal(err)
				}
				// emtpy parts for next iteration, if any
				parts = []*genai.Part{}
				if ok, res := hasInvokedTool(ctx, params, resp); ok {
					// if chat mode, send response to model
					parts = append(parts, &genai.Part{Text: res.Response["Response"].(string)})
					resp = &genai.GenerateContentResponse{
						Candidates: []*genai.Candidate{
							{
								Content: &genai.Content{
									Parts: parts,
								},
								Index: 0,
							},
						},
					}
				}
				if err := emitCandidates(out, resp.Candidates, params.ImgModality); err != nil {
					genLogFatal(err)
				}
				if params.TokenCount && resp.UsageMetadata != nil {
					tokenCount = resp.UsageMetadata.TotalTokenCount
				}
			}
		}
		fmt.Fprint(out, "\n")
		if !params.ChatMode {
			break
		}
		if params.Verbose {
			fmt.Fprintf(os.Stderr, "\033[36m")
			hist := chat.History(false)
			emitHistory(os.Stderr, hist)
			fmt.Fprintf(os.Stderr, "\033[0m")
		}
		input, err := readLine(tty)
		if err != nil {
			genLogFatal(err)
		}
		// Check for double blank line exit condition
		if input == "" {
			input, err = readLine(tty)
			if err != nil {
				genLogFatal(err)
			}
			if input == "" {
				if err = persistChat(chat); err != nil {
					genLogFatal(err)
				}
				break // exit chat mode
			}
		}
		parts = append(parts, &genai.Part{Text: input})
	}

	return 0
}
