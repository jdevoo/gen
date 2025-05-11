package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"

	"google.golang.org/api/iterator"
	"google.golang.org/genai"
)

func amandaGen(ctx context.Context, in io.Reader, out io.Writer, params *Parameters) int {
	for {
		var buf bytes.Buffer
		var e Env
		// fetch tuple from whiteboard
		params.Whiteboard.In(&e)
		params.Args = []string{e.Card.String()}
		res := emitGen(ctx, in, &buf, params) // params.Stdin false
		if res != 0 {
			return res
		}
		// TODO revisit convention
		// determines next role from last line in buf
		next := lastWord(&buf)
		// determine current role from prompt filename
		_, file := filepath.Split(params.FilePaths[0])
		fmt.Fprintf(out, "[\033[36m%s\033[0m] \033[97m%s\033[0m\n", strings.TrimSuffix(file, siExt), buf.String())
		// put result back on whiteboard
		if next != "" {
			params.Whiteboard.Out(Env{
				Card: &buf,
				Next: &next,
			})
		} else {
			params.Whiteboard.Out(Env{
				Card: &buf,
			})
		}
	}
}

func emitGen(ctx context.Context, in io.Reader, out io.Writer, params *Parameters) int {
	var err error
	var parts []*genai.Part
	var sysParts []*genai.Part
	var stdinData []byte
	var config interface{}

	// Handle stdin data
	if params.Stdin {
		stdinData, err = io.ReadAll(in)
		if err != nil {
			log.Fatal(err)
		}
		params.Stdin = len(stdinData) > 0
	}

	// Create a genai client
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  os.Getenv("GEMINI_API_KEY"),
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		genLogFatal(err)
	}

	// Handle file option
	if len(params.FilePaths) > 0 {
		for _, filePathVal := range params.FilePaths {
			if filePathVal == "-" {
				if params.SystemInstruction {
					sysParts = append(sysParts, &genai.Part{Text: searchReplace(string(stdinData), keyVals)})
				} else {
					parts = append(parts, &genai.Part{Text: searchReplace(string(stdinData), keyVals)})
				}
				continue
			}
			if err := glob(ctx, client, filePathVal, parts, sysParts, keyVals); err != nil {
				genLogFatal(err)
			}
		}
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

	/*
		// Set embedding model if -e or -d are used
		if params.Embed || len(params.DigestPaths) > 0 {
			if isFlagSet("m") {
				model = client.EmbeddingModel(params.GenModel)
			} else {
				model = client.EmbeddingModel(embModel)
			}
		} else {
			model = client.GenerativeModel(params.GenModel)
		}
	*/

	// Handle embed parameter and exit
	if params.Embed {
		//res, err := model.(*genai.EmbeddingModel).EmbedContent(ctx, parts...)
		res, err := client.Models.EmbedContent(ctx, embModel, []*genai.Content{{Parts: parts}}, nil)
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
			//query, err := client.Models.EmbedContent(ctx, embModel,
			query, err := client.Models.EmbedContent(ctx, embModel, []*genai.Content{{Parts: parts}}, nil)
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
				replacePart(sysParts, idx, digestKey, res)
			} else if idx := partWithKey(parts, digestKey); idx != -1 {
				replacePart(parts, idx, digestKey, res)
			} else {
				parts = prependToParts(parts, res)
			}
		}
		// Switch to generative model for the remainder of this program
		//model = client.GenerativeModel(params.GenModel)
	}

	// Set temperature and top_p from args or model defaults
	config = &genai.GenerateContentConfig{
		Temperature: genai.Ptr[float32](float32(params.Temp)),
		TopP:        genai.Ptr[float32](float32(params.TopP)),
	}

	// Handle json parameter
	if params.JSON {
		config.(*genai.GenerateContentConfig).ResponseMIMEType = "application/json"
	}

	// Register tools declared in the tools.go file
	if params.Tool {
		registerTools(config.(*genai.GenerateContentConfig)) // FunctionCallingAny
	}

	// Allow code execution
	if params.Code {
		config.(*genai.GenerateContentConfig).Tools =
			[]*genai.Tool{{CodeExecution: &genai.ToolCodeExecution{}}}
	}

	// Handle unsafe parameter
	if params.Unsafe {
		config.(*genai.GenerateContentConfig).SafetySettings = []*genai.SafetySetting{
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

	/*
		// Handle verbose parameter
		if params.Verbose {
			var config genai.GetModelConfig
			if params.Embed || len(params.DigestPaths) > 0 {
				if isFlagSet("m") {
					_, err = client.Models.Get(ctx, params.GenModel, &config)
				} else {
					_, err = client.Models.Get(ctx, embModel, &config)
				}
			} else {
				_, err = client.Models.Get(ctx, params.GenModel, &config)
			}
			if err != nil {
				genLogFatal(err)
			}
			fmt.Fprintf(os.Stderr, "\033[36m%s | %d | %d | %.2f | %.2f | %d\033[0m\n\n", info.Name, info.InputTokenLimit, info.OutputTokenLimit, params.Temp, params.TopP, info.TopK)
		}
	*/

	// Start chat session
	// TODO add history
	sess, err := client.Chats.Create(ctx, params.GenModel, config.(*genai.GenerateContentConfig), nil)
	if err != nil {
		genLogFatal(err)
	}
	tty := in

	// Set file descriptor for chat input
	if params.Stdin && params.ChatMode {
		tty, err = os.Open("/dev/tty")
		if err != nil {
			log.Fatal(err)
		}
	}

	// Set system prompt parts
	if len(sysParts) > 0 {
		config.(*genai.GenerateContentConfig).SystemInstruction = &genai.Content{
			Parts: sysParts,
			Role:  "model",
		}
		if params.Verbose {
			fmt.Fprintf(os.Stderr, "\033[36m%+v\033[0m\n\n", *config.(*genai.GenerateContentConfig).SystemInstruction)
		}
	}

	// Main chat loop
	for len(parts) > 0 {
		iter := sess.SendStream(ctx, parts...)
		parts = []*genai.Part{}
		for {
			resp, err := iter.Next()
			if err == iterator.Done {
				break
			}
			if err != nil {
				fmt.Fprintf(out, "\n")
				genLogFatal(err)
			}
			if ok, res := hasInvokedTool(resp); ok {
				//if params.ChatMode {
				// send response to Gemini
				parts = append(parts, &genai.Part{FunctionResponse: &res})
				//}
				resp = &genai.GenerateContentResponse{
					Candidates: []*genai.Candidate{
						{
							Index: 0,
							Content: &genai.Content{
								Parts: parts,
							},
						},
					},
				}
			}
			emitGeneratedResponse(out, resp)
			if params.TokenCount {
				tokenCount = resp.UsageMetadata.TotalTokenCount
			}
		}
		if params.Verbose {
			fmt.Fprintf(os.Stderr, "\n\033[36m")
			content := sess.History(false)
			pen := len(content) - 1
			for i, c := range content {
				if i == pen {
					break
				}
				fmt.Fprintf(os.Stderr, "\033[97m%02d:\033[36m %+v\n", i, c)
			}
			fmt.Fprintf(os.Stderr, "\033[0m")
		}
		fmt.Fprint(out, "\n")
		if !params.ChatMode {
			break
		}
		input, err := readLine(tty)
		if err != nil {
			log.Fatal(err)
		}
		// Check for double blank line exit condition
		if input == "" {
			input, err = readLine(tty)
			if err != nil {
				log.Fatal(err)
			}
			if input == "" {
				break // exit chat mode
			}
		}
		parts = append(parts, &genai.Part{Text: input})
	}

	if params.TokenCount {
		fmt.Fprintf(out, "\033[31m%d tokens\033[0m\n", tokenCount)
	}

	return 0
}
