package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

func emitGen(in io.Reader, out io.Writer, flags *Flags) int {
	var err error
	var parts []genai.Part
	var sysParts []genai.Part
	var stdinData []byte
	var model interface{}

	// Check for API key
	if val, ok := os.LookupEnv("GEMINI_API_KEY"); !ok || len(val) == 0 {
		fmt.Fprintf(out, "Environment variable GEMINI_API_KEY not set!\n")
		return 1
	}

	// Handle stdin data
	if flags.Stdin {
		stdinData, err = io.ReadAll(in)
		if err != nil {
			log.Fatal(err)
		}
		flags.Stdin = len(stdinData) > 0
	}

	// Create a genai client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		genLogFatal(err)
	}
	defer client.Close()

	// Handle file option
	if len(flags.FilePaths) > 0 {
		for _, filePathVal := range flags.FilePaths {
			if filePathVal == "-" {
				if flags.SystemInstruction {
					sysParts = append(sysParts, genai.Text(searchReplace(string(stdinData), keyVals)))
				} else {
					parts = append(parts, genai.Text(searchReplace(string(stdinData), keyVals)))
				}
				continue
			}
			if err := glob(ctx, client, filePathVal, &parts, &sysParts, keyVals); err != nil {
				genLogFatal(err)
			}
		}
	}

	// Handle argument
	if len(flag.Args()) > 0 {
		text := searchReplace(strings.Join(flag.Args(), " "), keyVals)
		if flags.Stdin && text == "-" {
			text = string(stdinData)
		}
		if flags.SystemInstruction && !(flags.Stdin && oneMatches(flags.FilePaths, "-")) {
			sysParts = append(sysParts, genai.Text(text))
		} else {
			parts = append(parts, genai.Text(text))
		}
	}

	// Set embedding model if -e or -d are used
	if flags.Embed || len(flags.DigestPaths) > 0 {
		if isFlagSet("m") {
			model = client.EmbeddingModel(flags.GenModel)
		} else {
			model = client.EmbeddingModel(embModel)
		}
	} else {
		model = client.GenerativeModel(flags.GenModel)
	}

	// Handle verbose flag
	if flags.Verbose {
		var info *genai.ModelInfo
		if flags.Embed || len(flags.DigestPaths) > 0 {
			info, err = model.(*genai.EmbeddingModel).Info(ctx)
		} else {
			info, err = model.(*genai.GenerativeModel).Info(ctx)
		}
		if err != nil {
			genLogFatal(err)
		}
		fmt.Fprintf(os.Stderr, "\033[36m%s | %d | %d | %.2f | %.2f | %d\033[0m\n\n", info.Name, info.InputTokenLimit, info.OutputTokenLimit, flags.Temp, flags.TopP, info.TopK)
	}

	// Handle embed flag and exit
	if flags.Embed {
		res, err := model.(*genai.EmbeddingModel).EmbedContent(ctx, parts...)
		if err != nil {
			genLogFatal(err)
		}
		if err := AppendToDigest(flags.DigestPaths[0], res.Embedding.Values, keyVals, flags.OnlyKvs, flags.Verbose, parts...); err != nil {
			genLogFatal(err)
		}
		return 0
	}

	// Handle digest flag and retrieve text from digest
	if len(flags.DigestPaths) > 0 {
		var res []QueryResult
		for _, digestPathVal := range flags.DigestPaths {
			query, err := model.(*genai.EmbeddingModel).EmbedContent(ctx, parts...)
			if err != nil {
				genLogFatal(err)
			}
			res, err = QueryDigest(digestPathVal, query.Embedding.Values, res, flags.K, float32(flags.Lambda), flags.Verbose)
			if err != nil {
				genLogFatal(err)
			}
		}
		if len(res) > 0 {
			// inject digest into a prompt or append as text
			if idx := partWithKey(sysParts, digestKey); idx != -1 {
				sysParts = replacePart(sysParts, idx, digestKey, res)
			} else if idx := partWithKey(parts, digestKey); idx != -1 {
				parts = replacePart(parts, idx, digestKey, res)
			} else {
				parts = prependToParts(parts, res)
			}
		}
		// Switch to generative model for the remainder of this program
		model = client.GenerativeModel(flags.GenModel)
	}

	// Set temperature and top_p from args or model defaults
	model.(*genai.GenerativeModel).SetTemperature(float32(flags.Temp))
	model.(*genai.GenerativeModel).SetTopP(float32(flags.TopP))

	// Handle json flag
	if flags.JSON {
		model.(*genai.GenerativeModel).ResponseMIMEType = "application/json"
	}

	// Register tools declared in the tools.go file
	if flags.Tool {
		registerTools(model.(*genai.GenerativeModel)) // FunctionCallingAny
	}

	// Allow code execution
	if flags.Code {
		model.(*genai.GenerativeModel).Tools = []*genai.Tool{{CodeExecution: &genai.CodeExecution{}}}
	}

	// Handle unsafe flag
	if flags.Unsafe {
		model.(*genai.GenerativeModel).SafetySettings = []*genai.SafetySetting{
			{
				Category:  genai.HarmCategoryDangerousContent,
				Threshold: genai.HarmBlockNone,
			},
			{
				Category:  genai.HarmCategoryHarassment,
				Threshold: genai.HarmBlockNone,
			},
			{
				Category:  genai.HarmCategoryHateSpeech,
				Threshold: genai.HarmBlockNone,
			},
			{
				Category:  genai.HarmCategorySexuallyExplicit,
				Threshold: genai.HarmBlockNone,
			},
		}
	}

	// Start chat session
	sess := model.(*genai.GenerativeModel).StartChat()
	tty := in

	// Set file descriptor for chat input
	if flags.Stdin && flags.ChatMode {
		tty, err = os.Open("/dev/tty")
		if err != nil {
			log.Fatal(err)
		}
	}

	// Set system parts
	if len(sysParts) > 0 {
		model.(*genai.GenerativeModel).SystemInstruction = &genai.Content{
			Parts: sysParts,
			Role:  "model",
		}
		if flags.Verbose {
			fmt.Fprintf(os.Stderr, "\033[36m%+v\033[0m\n\n", *model.(*genai.GenerativeModel).SystemInstruction)
		}
	}

	// Main chat loop
	for {
		if len(parts) > 0 {
			iter := sess.SendMessageStream(ctx, parts...)
			if flags.TokenCount {
				res, err := model.(*genai.GenerativeModel).CountTokens(ctx, parts...)
				if err != nil {
					genLogFatal(err)
				}
				tokenCount += res.TotalTokens
			}
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
					// send response to Gemini and resume generation
					parts = []genai.Part{genai.Text(res)}
					iter = sess.SendMessageStream(ctx, parts...)
					continue
				}
				emitGeneratedResponse(out, resp)
			}
		}
		if flags.Verbose {
			fmt.Fprintf(os.Stderr, "\n\033[36m")
			for i, c := range sess.History {
				if i == len(sess.History)-1 {
					break
				}
				fmt.Fprintf(os.Stderr, "\033[97m%02d:\033[36m %+v\n", i, c)
			}
			fmt.Fprintf(os.Stderr, "\033[0m")
		}
		fmt.Fprint(out, "\n")
		if !flags.ChatMode {
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
		parts = []genai.Part{genai.Text(input)}
	}

	if flags.TokenCount {
		fmt.Fprintf(out, "\033[31m%d tokens\033[0m\n", tokenCount)
	}

	return 0
}
