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
	"runtime"
	"strings"
	"syscall"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

// Version information, populated by make
// Token count accumulator in case of CTRL-C
// Parameter map shared with tools
var (
	version    string
	golang     string
	githash    string
	tokenCount int32
	keyVals    ParamMap
)

const (
	siExt = ".sprompt"
	pExt  = ".prompt"
)

// Usage overrides PrintDefaults to provide custom usage information.
func emitUsage(out io.Writer) {
	fmt.Fprintf(out, "Usage: "+filepath.Base(os.Args[0])+" [options] <prompt>\n")
	fmt.Fprintf(out, "\n")
	fmt.Fprintf(out, "Command-line interface to Google Gemini large language models\n")
	fmt.Fprintf(out, "  Requires a valid GEMINI_API_KEY environment variable set.\n")
	fmt.Fprintf(out, "  Content is generated by a prompt and optional system instructions.\n")
	fmt.Fprintf(out, "  Use - to assign stdin as prompt argument or as attached file.\n")
	fmt.Fprintf(out, "\n")
	fmt.Fprintf(out, "Options:\n")
	flag.PrintDefaults()
}

func emitGen(in io.Reader, out io.Writer) int {
	var err error
	var prompts []genai.Part
	var instructions []genai.Part
	var stdinData []byte

	// Check for API key
	if val, ok := os.LookupEnv("GEMINI_API_KEY"); !ok || len(val) == 0 {
		fmt.Fprintf(out, "Environment variable GEMINI_API_KEY not set!\n")
		return 1
	}

	// Flag handling
	verboseFlag := flag.Bool("V", false, "output model details, system instructions and chat history\ndetails include model name | maxInputTokens | maxOutputTokens | temp | top_p | top_k")
	chatModeFlag := flag.Bool("c", false, "enter chat mode after content generation\ntype two consecutive blank lines to exit\nnot supported on windows when stdin used")
	filePaths := ParamArray{}
	flag.Var(&filePaths, "f", fmt.Sprintf("file to attach where value is the path to the file\nuse extensions %s and %s for user and system instructions respectively\nrepeat for each file", pExt, siExt))
	helpFlag := flag.Bool("h", false, "show this help message and exit")
	jsonFlag := flag.Bool("json", false, "response in JavaScript Object Notation")
	modelName := flag.String("m", "gemini-1.5-flash", "generative model name")
	keyVals = ParamMap{}
	flag.Var(&keyVals, "p", "prompt parameter value in format key=val\nreplaces all occurrences of {key} in prompt with val\nrepeat for each parameter")
	systemInstructionFlag := flag.Bool("s", false, "treat argument as system instruction\nunless stdin is set as file")
	tokenCountFlag := flag.Bool("t", false, "output total number of tokens")
	tempVal := flag.Float64("temp", 1.0, "changes sampling during response generation [0.0,2.0]")
	toolFlag := flag.Bool("tool", false, fmt.Sprintf("invoke one of the tools {%s}", knownTools()))
	topPVal := flag.Float64("top_p", 0.95, "changes how the model selects tokens for generation [0.0,1.0]")
	unsafeFlag := flag.Bool("unsafe", false, "force generation when gen aborts with FinishReasonSafety")
	versionFlag := flag.Bool("v", false, "show version and exit")
	flag.Parse()

	if *helpFlag {
		emitUsage(out)
		return 0
	}

	// Handle version flag
	if *versionFlag {
		fmt.Fprintf(out, "gen version %s (%s %s)\n", version, golang, githash)
		return 0
	}

	// Handle stdin data
	stdinFlag := hasInputFromStdin(in)
	if stdinFlag {
		stdinData, err = io.ReadAll(in)
		if err != nil {
			log.Fatal(err)
		}
		stdinFlag = len(stdinData) > 0
	}

	// Handle invalid argument and option combinations
	if (*topPVal < 0 || *topPVal > 1) || // temp out of range
		// no prompt as stdin, argument or file
		(!stdinFlag && len(flag.Args()) == 0 && !anyMatches(filePaths, pExt, siExt)) ||
		// lack of /dev/tty on Windows prevents this flag combination
		(runtime.GOOS == "windows" && stdinFlag && *chatModeFlag) ||
		// stdin set but neither used as file nor as argument
		(stdinFlag && !(len(flag.Args()) == 1 && flag.Args()[0] == "-") && !oneMatches(filePaths, "-")) ||
		(!*chatModeFlag && *systemInstructionFlag &&
			// no chat mode, stdin as system instruction, no prompt argument
			((stdinFlag && oneMatches(filePaths, "-") && len(flag.Args()) == 0) ||
				// no chat mode, no file as prompt, argument as system instruction
				(!stdinFlag && len(flag.Args()) > 0 && !anyMatches(filePaths, pExt)) ||
				// no chat mode, file as system instruction, no prompt
				(!stdinFlag && len(flag.Args()) == 0 && allMatch(filePaths, siExt)))) ||
		(*chatModeFlag && *systemInstructionFlag && len(filePaths) > 0 &&
			!anyMatches(filePaths, pExt, siExt) &&
			// chat mode, no file as prompt, no stdin, argument as system instruction
			((!stdinFlag && len(flag.Args()) > 0) ||
				// chat mode, no file as prompt, stdin as system instruction file, no prompt argument
				(stdinFlag && len(flag.Args()) == 0 && oneMatches(filePaths, "-")) ||
				// chat mode, no file as prompt, no file as system instruction, argument as system instruction
				(stdinFlag && !oneMatches(filePaths, "-") && len(flag.Args()) == 1 && flag.Args()[0] == "-"))) {
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

	// Set generative model
	model := client.GenerativeModel(*modelName)

	// Set temperature and top_p from args or model defaults
	model.SetTemperature(float32(*tempVal))
	model.SetTopP(float32(*topPVal))

	// Handle json flag
	if *jsonFlag {
		model.ResponseMIMEType = "application/json"
	}

	// Handle unsafe flag
	if *unsafeFlag {
		model.SafetySettings = []*genai.SafetySetting{
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

	// Handle tool flag registering tools declared in the tools.go file
	if *toolFlag {
		registerTools(model) // FunctionCallingAny
	}

	// Handle file option
	if len(filePaths) > 0 {
		for _, filePathVal := range filePaths {
			if filePathVal == "-" {
				if *systemInstructionFlag {
					instructions = append(instructions, genai.Text(searchReplace(string(stdinData), keyVals)))
				} else {
					prompts = append(prompts, genai.Text(searchReplace(string(stdinData), keyVals)))
				}
				continue
			}
			f, err := os.Open(filePathVal)
			if err != nil {
				log.Fatal(err)
			}
			defer f.Close()
			switch path.Ext(filePathVal) {
			case pExt, siExt:
				data, err := io.ReadAll(f)
				if err != nil {
					log.Fatal(err)
				}
				if path.Ext(filePathVal) == siExt {
					instructions = append(instructions, genai.Text(searchReplace(string(data), keyVals)))
				} else {
					prompts = append(prompts, genai.Text(searchReplace(string(data), keyVals)))
				}
			case ".jpg", ".jpeg", ".png", ".gif", ".webp",
				".mp3", ".wav", ".aiff", ".aac", ".ogg", ".flac", ".pdf":
				file, err := uploadFile(ctx, client, filePathVal)
				if err != nil {
					genLogFatal(err)
				}
				prompts = append(prompts, genai.FileData{MIMEType: strings.Split(file.MIMEType, ";")[0], URI: file.URI})
				defer func() {
					err := client.DeleteFile(ctx, file.Name)
					if err != nil {
						genLogFatal(err)
					}
				}()
			default:
				data, err := io.ReadAll(f)
				if err != nil {
					log.Fatal(err)
				}
				prompts = append(prompts, genai.Text(searchReplace(string(data), keyVals)))
			}
		}
	}

	// Handle argument
	if len(flag.Args()) > 0 {
		text := searchReplace(strings.Join(flag.Args(), " "), keyVals)
		if stdinFlag && text == "-" {
			text = string(stdinData)
		}
		if *chatModeFlag && *systemInstructionFlag {
			instructions = append(instructions, genai.Text(text))
		} else {
			prompts = append(prompts, genai.Text(text))
		}
	}

	// Handle model information
	if *verboseFlag {
		info, err := model.Info(ctx)
		if err != nil {
			genLogFatal(err)
		}
		fmt.Fprintf(out, "\033[36m%s | %d | %d | %.2f | %.2f | %d\033[0m\n", info.Name, info.InputTokenLimit, info.OutputTokenLimit, *tempVal, *topPVal, info.TopK)
	}

	// Start chat session
	sess := model.StartChat()
	tty := in

	// Set file descriptor for chat input
	if stdinFlag && *chatModeFlag {
		tty, err = os.Open("/dev/tty")
		if err != nil {
			log.Fatal(err)
		}
	}

	// Set system instructions
	if len(instructions) > 0 {
		model.SystemInstruction = &genai.Content{
			Parts: instructions,
			Role:  "model",
		}
		if *verboseFlag {
			fmt.Fprintf(out, "\033[36m%+v\033[0m\n", *model.SystemInstruction)
		}
	}

	// Main chat loop
	for {
		if len(prompts) > 0 {
			iter := sess.SendMessageStream(ctx, prompts...)
			if *tokenCountFlag {
				res, err := model.CountTokens(ctx, prompts...)
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
				emitGeneratedResponse(out, resp)
			}
		}
		if *verboseFlag {
			fmt.Fprintf(out, "\033[36m")
			for i, c := range sess.History {
				fmt.Fprintf(out, "%02d: %+v", i, c)
			}
			fmt.Fprintf(out, "\033[0m\n")
		}
		fmt.Fprint(out, "\n")
		if !*chatModeFlag {
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
		prompts = []genai.Part{genai.Text(input)}
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
			fmt.Printf("\n\033[31m%d tokens\033[0m\n", tokenCount)
		}
		os.Exit(1)
	}()
	os.Exit(emitGen(os.Stdin, os.Stdout))
}
