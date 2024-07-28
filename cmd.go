package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
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
	fmt.Fprintf(out, "  Requires a valid GEMINI_API_KEY environment variable set\n")
	fmt.Fprintf(out, "  The prompt is set from stdin and/or arguments.\n")
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
	chatModeFlag := flag.Bool("c", false, "enter chat mode using prompt\nenter 2 consecutive blank lines to exit")
	filePathVal := flag.String("f", "", "attach file to prompt where string is the path to the file")
	helpFlag := flag.Bool("h", false, "show this help message and exit")
	jsonFlag := flag.Bool("json", false, "response uses the application/json MIME type")
	modelName := flag.String("m", "gemini-1.5-flash", "generative model name")
	keyVals := ParamMap{}
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val\nreplaces all occurrences of {key} in prompt with val")
	systemInstructionFlag := flag.Bool("s", false, "treat prompt as system instruction\nstdin used if found")
	tokenCountFlag := flag.Bool("t", false, "output number of tokens for prompt")
	tempVal := flag.Float64("temp", 1.0, "changes sampling during response generation [0.0,2.0]")
	toolFlag := flag.Bool("tool", false, fmt.Sprintf("invoke one of the tools {%s}", knownTools()))
	topPVal := flag.Float64("top_p", 0.95, "change how the model selects tokens for generation [0.0,1.0]")
	unsafeFlag := flag.Bool("unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	versionFlag := flag.Bool("v", false, "show version and exit")
	flag.Parse()

	// Handle version flag
	if *versionFlag {
		fmt.Fprintf(out, "gen version %s (%s %s)\n", version, golang, githash)
		return 0
	}

	// Set prompt from stdin, if any
	var prompt string
	stdinFlag := hasInputFromStdin(in)
	if stdinFlag {
		if data, err := io.ReadAll(in); err != nil {
			log.Fatal(err)
		} else {
			prompt = string(data)
			stdinFlag = len(prompt) > 0
		}
	}

	// Handle invalid combinations
	// no prompt at all, as argument or via stdin
	// system instruction flag with prompt but no stdin
	if *helpFlag ||
		(*tempVal < 0 || *tempVal > 2) ||
		(*topPVal < 0 || *topPVal > 1) ||
		(!stdinFlag && len(flag.Args()) == 0) {
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

	// Handle safety flag
	if *unsafeFlag {
		model.SafetySettings = []*genai.SafetySetting{
			{
				Category:  genai.HarmCategoryDangerousContent,
				Threshold: genai.HarmBlockNone,
			},
		}
	}

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

	// Set system instruction
	if stdinFlag {
		prompt = searchReplace(prompt, keyVals)
		if *systemInstructionFlag {
			model.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(prompt)},
			}
			prompt = ""
		}
	}
	if len(flag.Args()) > 0 {
		prompt += strings.Join(flag.Args(), " ")
		prompt = searchReplace(prompt, keyVals)
		if !stdinFlag && *chatModeFlag && *systemInstructionFlag {
			model.SystemInstruction = &genai.Content{
				Parts: []genai.Part{genai.Text(prompt)},
			}
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
	var r io.Reader

	// Set descriptor for chat input
	if stdinFlag {
		r, err = os.Open("/dev/tty")
		if err != nil {
			log.Fatal(err)
		}
	} else {
		r = in
	}

	// Handle attach flag
	var file *genai.File
	if *filePathVal != "" {
		f, err := os.Open(*filePathVal)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		mimeType := getMIMEType(f)
		opts := &genai.UploadFileOptions{MIMEType: mimeType}
		file, err = client.UploadFile(ctx, "", f, opts)
		if err != nil {
			genLogFatal(err)
		}
		defer func() {
			err := client.DeleteFile(ctx, file.Name)
			if err != nil {
				genLogFatal(err)
			}
		}()
		if *tokenCountFlag {
			resp, err := model.CountTokens(ctx, genai.FileData{URI: file.URI})
			if err != nil {
				genLogFatal(err)
			}
			tokenCount += resp.TotalTokens
		}
		_, err = sess.SendMessage(ctx, genai.FileData{URI: file.URI})
		if err != nil {
			genLogFatal(err)
		}
	}

	for {
		if *tokenCountFlag {
			resp, err := model.CountTokens(ctx, genai.Text(prompt))
			if err != nil {
				genLogFatal(err)
			}
			tokenCount += resp.TotalTokens
		}
		iter := sess.SendMessageStream(ctx, genai.Text(prompt))
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
		prompt, err = readLine(r)
		if err != nil {
			log.Fatal(err)
		}
		// Check for double blank line exit condition
		if prompt == "" {
			prompt, err = readLine(r)
			if err != nil {
				log.Fatal(err)
			}
			if prompt == "" {
				break // exit chat mode
			}
		}
	}

	if *tokenCountFlag {
		fmt.Fprintf(out, "\033[31m%d tokens\033[0m\n", tokenCount)
	}

	return 0
}

func main() {
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-done
		if tokenCount > 0 {
			fmt.Printf("\033[31m%d tokens\033[0m\n", tokenCount)
		}
		os.Exit(1)
	}()
	os.Exit(emitGen(os.Stdin, os.Stdout))
}
