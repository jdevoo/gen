package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// Version information, populated by make
// Token count accumulator in case of CTRL-C
var (
	version    string
	golang     string
	githash    string
	tokenCount int32
)

// Usage overrides PrintDefaults to provide custom usage information.
func emitUsage(out io.Writer) {
	fmt.Fprintf(out, "Usage: "+filepath.Base(os.Args[0])+" [options] <prompt>\n")
	fmt.Fprintf(out, "\n")
	fmt.Fprintf(out, "Command-line interface to Google Gemini large language models\n")
	fmt.Fprintf(out, "  Requires a valid GEMINI_API_KEY environment variable set.\n")
	fmt.Fprintf(out, "  Content is generated according to the prompt argument.\n")
	fmt.Fprintf(out, "  Additionally, supports stdin and .prompt files as valid prompt parts.\n")
	fmt.Fprintf(out, "\n")
	fmt.Fprintf(out, "Options:\n")
	flag.PrintDefaults()
}

func emitGen(in io.Reader, out io.Writer) int {
	var err error

	// Check for API key
	if val, ok := os.LookupEnv("GEMINI_API_KEY"); !ok || len(val) == 0 {
		fmt.Fprintf(out, "Environment variable GEMINI_API_KEY not set!\n")
		return 1
	}

	// Flag handling
	verboseFlag := flag.Bool("V", false, "output model | maxInputTokens | maxOutputTokens | temp | top_p | top_k")
	chatModeFlag := flag.Bool("c", false, "enter chat mode after content generation\ntype two consecutive blank lines to exit")
	filePathVal := flag.String("f", "", "attach file to prompt where string is the path to the file\nfile with the extension .prompt is treated as prompt")
	helpFlag := flag.Bool("h", false, "show this help message and exit")
	jsonFlag := flag.Bool("json", false, "response in JavaScript Object Notation")
	modelName := flag.String("m", "gemini-1.5-flash", "generative model name")
	keyVals := ParamMap{}
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val\nreplaces all occurrences of {key} in prompt with val")
	systemInstructionFlag := flag.Bool("s", false, "treat first of stdin or file option as system instruction")
	tokenCountFlag := flag.Bool("t", false, "output number of tokens for prompt")
	tempVal := flag.Float64("temp", 1.0, "changes sampling during response generation [0.0,2.0]")
	toolFlag := flag.Bool("tool", false, fmt.Sprintf("invoke one of the tools {%s}", knownTools()))
	topPVal := flag.Float64("top_p", 0.95, "changes how the model selects tokens for generation [0.0,1.0]")
	unsafeFlag := flag.Bool("unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	versionFlag := flag.Bool("v", false, "show version and exit")
	flag.Parse()

	// Handle version flag
	if *versionFlag {
		fmt.Fprintf(out, "gen version %s (%s %s)\n", version, golang, githash)
		return 0
	}

	// Set stdin as prompt, if provided
	var prompt []genai.Part
	stdinFlag := hasInputFromStdin(in)
	if stdinFlag {
		if data, err := io.ReadAll(in); err != nil {
			log.Fatal(err)
		} else {
			prompt = append(prompt, genai.Text(searchReplace(string(data), keyVals)))
			stdinFlag = len(prompt) > 0
		}
	}

	// Handle invalid combinations
	// no prompt at all, as argument, stdin or file
	// stdin as system prompt, no other prompt
	// argument as system prompt, no other prompt
	// file option as system prompt, no other prompt
	if *helpFlag ||
		(*tempVal < 0 || *tempVal > 2) ||
		(*topPVal < 0 || *topPVal > 1) ||
		(!stdinFlag && len(flag.Args()) == 0 && *filePathVal == "") ||
		*systemInstructionFlag && ((stdinFlag && len(flag.Args()) == 0 && *filePathVal == "") ||
			(!stdinFlag && len(flag.Args()) != 0 && *filePathVal == "") ||
			(!stdinFlag && len(flag.Args()) == 0 && path.Ext(*filePathVal) == ".prompt")) {
		emitUsage(out)
		return 1
	}

	// Create a genai client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		genLogFatal(err)
	}
	defer client.Close()

	model := client.GenerativeModel(*modelName)

	// Set temperature and top_p from args or model defaults
	model.SetTemperature(float32(*tempVal))
	model.SetTopP(float32(*topPVal))

	// Handle json flag
	if *jsonFlag {
		model.ResponseMIMEType = "application/json"
	}

	// Register tools declared in the tools.go file
	if *toolFlag {
		registerTools(model, genai.FunctionCallingAny)
	} else {
		registerTools(model, genai.FunctionCallingNone)
	}

	// Promote stdin prompt as system instruction
	if stdinFlag && *systemInstructionFlag {
		model.SystemInstruction = &genai.Content{
			Parts: prompt,
		}
		prompt = nil
	}

	// Handle file option and set as prompt or system instruction if file ends with .prompt
	var file *genai.File
	if *filePathVal != "" {
		f, err := os.Open(*filePathVal)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		if path.Ext(*filePathVal) == ".prompt" {
			if data, err := io.ReadAll(f); err != nil {
				log.Fatal(err)
			} else {
				prompt = append(prompt, genai.Text(searchReplace(string(data), keyVals)))
				if !stdinFlag && len(flag.Args()) > 0 && *systemInstructionFlag {
					model.SystemInstruction = &genai.Content{
						Parts: prompt,
					}
					prompt = nil
				}
			}
		} else {
			file, err = uploadFile(ctx, client, *filePathVal)
			if err != nil {
				genLogFatal(err)
			}
			defer func() {
				err := client.DeleteFile(ctx, file.Name)
				if err != nil {
					genLogFatal(err)
				}
			}()
			if err != nil {
				genLogFatal(err)
			}
		}
	}

	// Handle argument as prompt
	if len(flag.Args()) > 0 {
		prompt = append(prompt, genai.Text(searchReplace(strings.Join(flag.Args(), " "), keyVals)))
	}

	// Send FileData to model if available
	if file != nil {
		prompt = append(prompt, genai.FileData{MIMEType: file.MIMEType, URI: file.URI})
		if err != nil {
			genLogFatal(err)
		}
	}

	// Handle verbose flag and output model information
	if *verboseFlag {
		info, err := model.Info(ctx)
		if err != nil {
			genLogFatal(err)
		}
		fmt.Fprintf(out, "\033[36m%s | %d | %d | %.2f | %.2f | %d\033[0m\n", info.Name, info.InputTokenLimit, info.OutputTokenLimit, *tempVal, *topPVal, info.TopK)
	}

	// Chat session
	sess := model.StartChat()
	var fd io.Reader

	// Set file descriptor for chat input
	if stdinFlag {
		fd, err = os.Open("/dev/tty")
		if err != nil {
			log.Fatal(err)
		}
	} else {
		fd = in
	}

	// Main chat loop
	for {
		if *unsafeFlag {
			model.SafetySettings = []*genai.SafetySetting{
				{
					Category:  genai.HarmCategoryDangerousContent,
					Threshold: genai.HarmBlockNone,
				},
			}
		}
		iter := sess.SendMessageStream(ctx, prompt...)
		if *tokenCountFlag {
			resp, err := model.CountTokens(ctx, prompt...)
			if err != nil {
				genLogFatal(err)
			}
			tokenCount += resp.TotalTokens
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
			emitGeneratedResponse(resp, out)
		}
		if !*chatModeFlag {
			break
		}
		if *verboseFlag {
			for i, c := range sess.History {
				fmt.Fprintf(out, "\033[36m%02d: %+v\033[0m\n", i, c)
			}
		}
		fmt.Fprintf(out, "\n")
		input, err := readLine(fd)
		if err != nil {
			log.Fatal(err)
		}
		// Check for double blank line exit condition
		if input == "" {
			input, err = readLine(fd)
			if err != nil {
				log.Fatal(err)
			}
			if input == "" {
				break // exit chat mode
			}
		}
		prompt = []genai.Part{genai.Text(input)}
	}

	if *tokenCountFlag {
		fmt.Fprintf(out, "\n\033[31m%d tokens\033[0m\n", tokenCount)
	}

	return 0
}

func main() {
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-done
		if tokenCount > 0 {
			fmt.Printf("\n\033[31m%d tokens\033[0m\n", tokenCount)
		}
		os.Exit(1)
	}()
	os.Exit(emitGen(os.Stdin, os.Stdout))
}
