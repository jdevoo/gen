package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/genai"
)

// amandaGen is a wrapper for emitGen
// FIXME experimental setup involving 2 instances and a word counting tool
// sequential: next role from last line in buf
// parallel execution of tool sets exit condition before timeout
func amandaGen(ctx context.Context, in io.Reader, out io.Writer, params *Parameters) int {
	var this string
	for {
		// determine current role from prompt filename
		_, file := filepath.Split(params.FilePaths[0])
		prompt := strings.TrimSuffix(file, siExt)
		var e Env
		var buf bytes.Buffer
		switch prompt {
		case "tool":
			Whiteboard.Rd(&e)
			if e.Receiver != nil && *e.Receiver == "sentinel" {
				continue
			}
			// set args from tuple data
			params.Args = []string{e.Args.String()}
			params.Tool = true
			// capture gen out in buffer
			res := emitGen(ctx, in, &buf, params)
			if res != 0 {
				return res
			}
			dest := "sentinel"
			Whiteboard.Out(Env{
				Args:     &buf,
				Receiver: &dest,
			})
		case "sentinel":
			if this == "" {
				this = prompt
			}
			e.Receiver = &this
			Whiteboard.In(&e)
			// set args from tuple data
			params.Args = []string{e.Args.String()}
			// capture gen out in buffer
			res := emitGen(ctx, in, &buf, params)
			if res != 0 {
				return res
			}
			if strings.ToLower(strings.TrimSpace(buf.String())) == "true" {
				return 0
			}
		case "code":
			// TODO
			continue
		default:
			// fetch tuple for this receiver; any on first iteration
			if this != "" {
				e.Receiver = &this
			}
			Whiteboard.In(&e)
			// set args from tuple data
			params.Args = []string{e.Args.String()}
			// capture gen out in buffer
			res := emitGen(ctx, in, &buf, params)
			if res != 0 {
				return res
			}
			next := lastWord(&buf)
			// put result back on whiteboard
			if next != "" {
				Whiteboard.Out(Env{
					Args:     &buf,
					Receiver: &next,
				})
			} else {
				Whiteboard.Out(Env{
					Args: &buf,
					// Receiver nil i.e. any
				})
			}
			// for following iterations, request tuples for this receiver
			if this == "" {
				this = prompt
			}
		}
		fmt.Fprintf(out, "[\033[36m%s\033[0m] \033[97m%s\033[0m\n", prompt, strings.TrimSpace(buf.String()))
	}
}

// emitGen is the main gen content generator
func emitGen(ctx context.Context, in io.Reader, out io.Writer, params *Parameters) int {
	var err error
	var parts []*genai.Part
	var sysParts []*genai.Part
	var stdinData []byte
	var mediaAssets []string

	// Handle stdin data
	if params.Stdin {
		stdinData, err = io.ReadAll(in)
		if err != nil {
			genLogFatal(err)
		}
		params.Stdin = len(stdinData) > 0
	}

	// Create a genai client
	client, err := genai.NewClient(ctx, nil)
	if err != nil {
		genLogFatal(err)
	}

	// Handle argument
	if len(params.Args) > 0 {
		text := searchReplace(strings.Join(params.Args, " "), keyVals)
		if params.Stdin && text == "-" {
			text = string(stdinData)
		}
		if params.SystemInstruction && !(params.Stdin && oneMatches(params.FilePaths, "-")) {
			sysParts = append(sysParts, &genai.Part{Text: text})
		} else {
			parts = append(parts, &genai.Part{Text: text})
		}
	}

	// Handle file option
	if len(params.FilePaths) > 0 {
		for _, filePathVal := range params.FilePaths {
			// stdin passed as file
			if filePathVal == "-" {
				if params.SystemInstruction {
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

	// Handle TokenCount
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
			if idx := partWithKey(sysParts, digestKey); idx != -1 {
				replacePart(&sysParts, idx, digestKey, res)
			} else if idx := partWithKey(parts, digestKey); idx != -1 {
				replacePart(&parts, idx, digestKey, res)
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
	// Handle Modality
	if params.ImgModality {
		config.ResponseModalities = []string{"IMAGE"}
	} else {
		config.ResponseModalities = []string{"TEXT"}
	}
	// Handle json parameter
	if params.JSON {
		config.ResponseMIMEType = "application/json"
	}
	// Register tools declared in the tools.go file
	if params.Tool {
		registerTools(config) // genai.FunctionCallingConfigModeAny
	}
	// Allow code execution
	if params.Code {
		config.Tools =
			[]*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}}
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
	tty := in

	// Set file descriptor for chat input
	if params.Stdin && params.ChatMode {
		tty, err = os.Open("/dev/tty")
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
				// emtpy parts for next iteration, if any
				parts = []*genai.Part{}
				if err != nil {
					fmt.Fprintf(out, "\n")
					genLogFatal(err)
				}
				if ok, res := hasInvokedTool(resp); ok {
					// if chat mode, send response to model
					//parts = append(parts, &genai.Part{FunctionResponse: res})
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
				emitCandidates(out, resp.Candidates)
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
