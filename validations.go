package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/jdevoo/gen/core"
	"google.golang.org/genai"
)

// isFlagSet visits the flags passed to the command at runtime.
func isFlagSet(name string) bool {
	res := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			res = true
		}
	})
	return res
}

// validPrompts checks prompts against regular interactive vs no redirect or piped content session.
func validPrompts(params *core.Parameters) error {
	if (params.Interactive &&
		// no regular prompt provided
		((len(params.Args) == 0 && !anyMatches(params.FilePaths, PExt) && !(params.NoOp && len(params.FilePaths) > 0)) ||
			// system instruction
			(params.SystemInstruction &&
				// not provided as file
				((len(params.Args) == 0 && !anyMatches(params.FilePaths, SPExt)) ||
					// provided as argument but no prompt as file and no chat mode
					(len(params.Args) > 0 && !anyMatches(params.FilePaths, PExt) && !params.ChatMode))))) ||
		(!params.Interactive &&
			// not set as file xor argument
			((!oneMatches(params.FilePaths, "-") && !(len(params.Args) == 1 && params.Args[0] == "-")) ||
				// system instruction
				(params.SystemInstruction &&
					// stdin as file, but no prompt as file or argument
					((len(params.Args) == 0 &&
						!oneMatches(params.FilePaths, "-") && !anyMatches(params.FilePaths, PExt)) ||
						// stdin as argument, no prompt as file
						(len(params.Args) == 1 &&
							params.Args[0] == "-" && !anyMatches(params.FilePaths, PExt) && !params.ChatMode))))) {
		return fmt.Errorf("invalid or missing prompt")
	}
	return nil
}

func validRanges(params *core.Parameters) error {
	// ThinkingLevel
	if strings.HasPrefix(string(genai.ThinkingLevelMinimal), string(params.ThinkingLevel)) {
		params.ThinkingLevel = genai.ThinkingLevelMinimal
	}
	if strings.HasPrefix(string(genai.ThinkingLevelLow), string(params.ThinkingLevel)) {
		params.ThinkingLevel = genai.ThinkingLevelLow
	}
	if strings.HasPrefix(string(genai.ThinkingLevelMedium), string(params.ThinkingLevel)) {
		params.ThinkingLevel = genai.ThinkingLevelMedium
	}
	if strings.HasPrefix(string(genai.ThinkingLevelHigh), string(params.ThinkingLevel)) {
		params.ThinkingLevel = genai.ThinkingLevelHigh
	}
	if
	// invalid thinking level
	(len(params.ThinkingLevel) < 3 ||
		params.ThinkingLevel != genai.ThinkingLevelUnspecified &&
			params.ThinkingLevel != genai.ThinkingLevelMinimal &&
			params.ThinkingLevel != genai.ThinkingLevelLow &&
			params.ThinkingLevel != genai.ThinkingLevelMedium &&
			params.ThinkingLevel != genai.ThinkingLevelHigh) ||
		// invalid out path
		(len(params.OutPath) > 0 && !isValidPath(params.OutPath)) ||
		// invalid k values
		(params.K < 0 || params.K > 10) ||
		// invalid lambda values
		(params.Lambda < 0 || params.Lambda > 1) ||
		// invalid temperature values
		(params.Temp <= 0 || params.Temp > 2) ||
		// invalid topP values
		(params.TopP < 0 || params.TopP > 1) ||
		// invalid frequency penalty value
		(params.FrequencyPenalty.Defined && (*params.FrequencyPenalty.Value < -2 || *params.FrequencyPenalty.Value > 2)) ||
		// invalid presence penalty value
		(params.PresencePenalty.Defined && (*params.PresencePenalty.Value < -2 || *params.PresencePenalty.Value > 2)) {
		return fmt.Errorf("invalid option values")
	}
	return nil
}

func validCombos(params *core.Parameters) error {
	if
	// at most one JSON schema
	(params.JSON && !zeroOrOneMatches(params.FilePaths, ".json")) ||
		// code execution with incompatible flags
		(params.CodeGen &&
			(params.Tool || params.Embed || params.JSON)) ||
		// tool registration with incompatible flags
		(params.Tool &&
			(params.CodeGen || params.SystemInstruction || params.Embed)) ||
		// search with incompatible flags
		(params.GoogleSearch &&
			params.Embed) ||
		// image modality with incompatible flags
		(params.ImgModality &&
			(params.GoogleSearch || params.CodeGen ||
				params.Tool || params.JSON || params.ChatMode || params.Embed || isFlagSet("stop"))) ||
		// stop with incompatible flags
		(isFlagSet("stop") &&
			(params.ImgModality || params.Embed || params.CodeGen)) ||
		// out path only with -img and no redirect
		(len(params.OutPath) > 0 &&
			(!(params.ImgModality || params.CodeGen) || params.OutRedirected)) ||
		// walk without file attached that is not some prompt
		(params.Walk &&
			(len(params.FilePaths) == 0 ||
				allMatch(params.FilePaths, PExt) || allMatch(params.FilePaths, SPExt))) ||
		// chat mode
		(params.ChatMode &&
			// with incompatible flags
			(params.JSON || params.ImgModality || params.Embed)) ||
		// no-op with incompatible flags
		(params.NoOp &&
			(params.ChatMode || params.Embed)) {
		return fmt.Errorf("invalid options combination")
	}
	return nil
}

func validEmbeddings(params *core.Parameters, keyVals core.ParamMap) error {
	if
	// embeddings
	params.Embed &&
		// incompatible flags
		(params.Unsafe || params.JSON ||
			isFlagSet("temp") || isFlagSet("top_p") || isFlagSet("k") || isFlagSet("l") ||
			// no digest set
			len(params.DigestPaths) != 1 ||
			// metadata missing
			(params.OnlyKvs && len(keyVals) == 0) ||
			// prompts set
			anyMatches(params.FilePaths, PExt) || anyMatches(params.FilePaths, SPExt) ||
			// no arguments or files to digest
			(!params.Interactive &&
				!((len(params.Args) == 1 && params.Args[0] == "-") || oneMatches(params.FilePaths, "-")))) {

		return fmt.Errorf("invalid use of -e")
	}
	return nil
}

// validArgs checks if there are still unhandled flags inside params.Args
func validArgs(params *core.Parameters) error {
	var err error
	flag.CommandLine.VisitAll(func(f *flag.Flag) {
		if err != nil {
			return
		}
		for _, arg := range params.Args {
			if arg == "-"+f.Name || strings.HasPrefix(arg, "-"+f.Name+"=") {
				err = fmt.Errorf("misplaced or unhandled flag '%s'", arg)
				return
			}

		}
	})
	return err
}

// isArgsInvalid performs a complete argument validation.
func isArgsInvalid(params *core.Parameters, keyVals core.ParamMap) error {
	if err := validArgs(params); err != nil {
		return err
	}
	if err := validPrompts(params); err != nil {
		return err
	}
	if err := validRanges(params); err != nil {
		return err
	}
	if err := validCombos(params); err != nil {
		return err
	}
	if err := validEmbeddings(params, keyVals); err != nil {
		return err
	}
	return nil
}
