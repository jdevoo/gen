package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"image/jpeg"
	_ "image/png"

	"google.golang.org/genai"
)

// isRedirected checks if stream is not a terminal.
func isRedirected(stream any) bool {
	f, ok := stream.(*os.File)
	if !ok || f == nil {
		return true // `stream` is **not** a file
	}
	fileInfo, err := f.Stat()
	if err != nil {
		return false
	}
	return fileInfo.Mode()&os.ModeCharDevice == 0
}

// readLine reads a line from standard input on Linux, Mac or Windows.
func readLine(r io.Reader) (string, error) {
	scanner := bufio.NewScanner(r)
	if scanner.Scan() {
		return scanner.Text(), nil
	}
	return "", scanner.Err()
}

// isRedirected checks if `out` supports non-printable characters.
func isEmpty(out io.Writer) bool {
	if f, ok := out.(*os.File); ok {
		if fileInfo, err := f.Stat(); err == nil {
			return fileInfo.Size() == 0
		}
	}
	return false
}

// rewind moves the file pointer back to the start if `out` is a file;
// will not work with a pipe.
func rewind(out io.Writer) bool {
	if f, ok := out.(*os.File); ok {
		if stat, err := f.Stat(); err == nil && stat.Mode().IsRegular() {
			if _, err := f.Seek(0, io.SeekStart); err == nil {
				f.Truncate(0)
				return true
			}
		}
	}
	return false
}

// emitCandidate is a wrapper to emitContent.
func emitCandidate(out io.Writer, cand *genai.Candidate, imgModality bool, verbose bool, idx *int) error {
	var finish genai.FinishReason
	if cand == nil || cand.Content == nil {
		return nil
	}
	mp := NewParser()
	if err := emitContent(out, cand.Content, imgModality, verbose, idx, mp); err != nil {
		return err
	}
	finish = cand.FinishReason
	if finish != "" {
		if verbose {
			if !isRedirected(out) {
				fmt.Fprintf(out, "\n"+infos("%s")+"\n", finish)
			} else if !imgModality {
				fmt.Fprintf(out, "\n%s\n", finish)
			}
		} else {
			fmt.Fprintln(out)
		}
	}
	return nil
}

// emitContent outputs LLM response parts.
func emitContent(out io.Writer, content *genai.Content, imgModality bool, verbose bool, idx *int, mp *MarkdownParser) error {
	for _, p := range content.Parts {
		if p.Text != "" {
			if !isRedirected(out) {
				if verbose && p.Thought {
					fmt.Fprintf(out, infos("%s"), p.Text)
				} else {
					s := p.Text
					if mp != nil {
						s = mp.Parse(p.Text)
					}
					fmt.Fprintf(out, plains("%s"), s)
				}
			} else if !imgModality || (verbose && p.Thought) {
				fmt.Fprintf(out, "%s", p.Text)
			}
			if idx != nil {
				*idx++
			}
			continue
		}
		if verbose && p.FunctionResponse != nil {
			for _, key := range []string{"output", "error"} {
				if val, ok := p.FunctionResponse.Response[key].(string); ok && val != "" {
					if !isRedirected(out) {
						fmt.Fprintf(out, infos("%s\n"), val)
					} else {
						fmt.Fprintf(out, "%s\n", val)
					}
				}
			}
			if idx != nil {
				*idx++
			}
			continue
		}
		if verbose && p.ExecutableCode != nil {
			if p.CodeExecutionResult != nil && p.CodeExecutionResult.Outcome == genai.OutcomeOK {
				if !isRedirected(out) {
					fmt.Fprintf(out, infos("```%s\n%s\n```\n"), p.ExecutableCode.Language, p.ExecutableCode.Code)
				} else {
					fmt.Fprintf(out, "```%s\n%s\n```\n", p.ExecutableCode.Language, p.ExecutableCode.Code)
				}
				if idx != nil {
					*idx++
				}
			}
		}
		if verbose && p.FileData != nil {
			if !isRedirected(out) {
				fmt.Fprintf(out, infos("[%s](%s)"), p.FileData.DisplayName, p.FileData.FileURI)
			} else {
				fmt.Fprintf(out, "[%s](%s)", p.FileData.DisplayName, p.FileData.FileURI)
			}
			if idx != nil {
				*idx++
			}
		}
		if p.InlineData != nil {
			if strings.HasPrefix(p.InlineData.MIMEType, "text") {
				if !isRedirected(out) {
					fmt.Fprintf(out, infos("%s"), p.InlineData.Data)
				} else {
					fmt.Fprint(out, p.InlineData.Data)
				}
				if idx != nil {
					*idx++
				}
			}
			if !strings.HasPrefix(p.InlineData.MIMEType, "image") {
				return fmt.Errorf("emitContent of type %s: not supported", p.InlineData.MIMEType)
			}
			reader := bytes.NewReader(p.InlineData.Data)
			img, _, err := image.Decode(reader)
			if err != nil {
				return fmt.Errorf("emitContent of type %s: %v", p.InlineData.MIMEType, err)
			}
			if isRedirected(out) {
				//if !isEmpty(out) { // output first image only
				if !rewind(out) { // rewind to keep last image only (NOTE fails for pipe)
					continue
				}
				// encode to jpeg format
				if err := jpeg.Encode(out, img, &jpeg.Options{Quality: 100}); err != nil {
					return fmt.Errorf("emitContent of type %s: %v", p.InlineData.MIMEType, err)
				}
			} else {
				if idx != nil && *idx > 0 {
					fmt.Fprintf(out, "\n")
				}
				// encode to Sixel format
				senc := SixelEncoder(out)
				senc.Dither = true
				if err := senc.Encode(img); err != nil {
					return fmt.Errorf("emitContent of type %s: %v", p.InlineData.MIMEType, err)
				}
				//if idx != nil && *idx > 0 {
				fmt.Fprint(out, "\n")
				//}
			}
			if idx != nil {
				*idx++
			}
			continue
		}
	}
	return nil
}

// emitImage renders segmentation results
func emitImage(out io.Writer, img *genai.Image) error {
	r := bytes.NewReader(img.ImageBytes)
	mask, _, err := image.Decode(r)
	if err != nil {
		return err
	}
	if isRedirected(out) {
		if !isEmpty(out) { // output first image only
			return nil
		}
		// encode to jpeg format
		if err := jpeg.Encode(out, mask, &jpeg.Options{Quality: 100}); err != nil {
			return err
		}
	} else {
		// encode to Sixel format
		senc := SixelEncoder(out)
		senc.Dither = true
		if err := senc.Encode(mask); err != nil {
			return err
		}
		fmt.Fprintf(out, "\n")
	}
	return nil
}

// emitHistory prints the chat history (verbose).
func emitHistory(out io.Writer, hist []*genai.Content) {
	var prev string
	for _, c := range hist {
		if prev != c.Role {
			if !isRedirected(out) {
				fmt.Fprintf(out, "\n"+roles("%s")+"\n", c.Role)
			} else {
				fmt.Fprintf(out, "\n***%s***\n", c.Role)
			}
			prev = c.Role
		}
		emitContent(out, c, false, true, nil, nil)
	}
	fmt.Fprint(out, "\n")
}

// uploadFile tracks state until FileStateActive reached.
// TODO timeout hardcoded
func uploadFile(ctx context.Context, client *genai.Client, path string) (*genai.File, error) {
	file, err := client.Files.UploadFromPath(ctx, path, nil)
	if err != nil {
		return nil, fmt.Errorf("uploading file '%s': %v", path, err)
	}

	pollingCtx, pollingCancel := context.WithTimeout(ctx, 120*time.Second)
	defer pollingCancel()

	for file.State == genai.FileStateProcessing {
		select {
		case <-pollingCtx.Done():
			return nil, fmt.Errorf("upload cancelled or timed out for '%s': %v", file.Name, pollingCtx.Err())
		case <-time.After(1 * time.Second):
			file, err = client.Files.Get(pollingCtx, file.Name, nil)
			if err != nil {
				return nil, fmt.Errorf("processing state for '%s': %v", file.Name, err)
			}
		}
	}
	if file.State != genai.FileStateActive {
		return nil, fmt.Errorf("uploaded file has state '%s': %v", file.State, err)
	}
	return file, nil
}

func loadImage(filePathVal string) (*genai.Image, error) {
	f, err := os.Open(filePathVal)
	if err != nil {
		return nil, fmt.Errorf("loadImage error opening file '%s': %v", filePathVal, err)
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("loadImage error reading file %s: %v", filePathVal, err)
	}
	sniffLen := len(data)
	if sniffLen > 512 {
		sniffLen = 512
	}
	sniffedType := http.DetectContentType(data[:sniffLen])
	if !strings.HasPrefix(sniffedType, "image") {
		return nil, fmt.Errorf("loadImage error reading file %s: type %s not supported", filePathVal, sniffedType)
	}
	return &genai.Image{
		ImageBytes: data,
		MIMEType:   sniffedType,
	}, nil
}

// loadPrompt reads a file and recursively replaces @subprompt with their content.
func loadPrompt(filePath string, seen map[string]bool) (string, error) {
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return "", err
	}
	if seen[absPath] {
		return "", fmt.Errorf("circular reference detected in: '%s'", absPath)
	}
	seen[absPath] = true
	data, err := os.ReadFile(absPath)
	if err != nil {
		return "", fmt.Errorf("reading prompt file '%s': %v", filePath, err)
	}
	content := string(data)
	dir := filepath.Dir(absPath)

	lines := strings.Split(content, "\n")
	for i, line := range lines {
		// find text starting with @
		words := strings.Fields(line)
		for _, word := range words {
			if strings.HasPrefix(word, "@") && (strings.HasSuffix(word, PExt) || strings.HasSuffix(word, SPExt)) {
				subFileName := strings.TrimPrefix(word, "@")
				subPath := filepath.Join(dir, subFileName)
				subContent, err := loadPrompt(subPath, seen)
				if err != nil {
					return "", err
				}
				lines[i] = strings.ReplaceAll(lines[i], word, subContent)
			}
		}
	}
	delete(seen, absPath)
	return strings.Join(lines, "\n"), nil
}

// filePathHandler processes a single file path for glob.
// parts and sysParts are extended with file content.
func filePathHandler(ctx context.Context, client *genai.Client, filePathVal string, parts *[]*genai.Part, sysParts *[]*genai.Part, jsonSchema *map[string]any) error {
	keyVals, ok := ctx.Value("keyVals").(ParamMap)
	if !ok {
		return fmt.Errorf("filePathHandler: keyVals not found in context")
	}
	f, err := os.Open(filePathVal)
	if err != nil {
		return fmt.Errorf("opening file '%s': %v", filePathVal, err)
	}
	defer f.Close()
	switch path.Ext(filePathVal) {
	case PExt, SPExt:
		data, err := loadPrompt(filePathVal, make(map[string]bool))
		if err != nil {
			return fmt.Errorf("filePathHandler '%s': %v", filePathVal, err)
		}
		if path.Ext(filePathVal) == SPExt {
			*sysParts = append(*sysParts, &genai.Part{Text: searchReplace(string(data), keyVals)})
		} else {
			*parts = append(*parts, &genai.Part{Text: searchReplace(string(data), keyVals)})
		}
	case ".jpg", ".jpeg", ".png", ".gif", ".webp",
		".mp3", ".wav", ".aiff", ".aac", ".ogg", ".flac", ".pdf":
		file, err := uploadFile(ctx, client, filePathVal)
		if err != nil {
			return fmt.Errorf("uploading file '%s': %v", filePathVal, err)
		}
		*parts = append(*parts, &genai.Part{FileData: &genai.FileData{
			FileURI:  file.URI,
			MIMEType: strings.Split(file.MIMEType, ";")[0],
		}})
	case ".json":
		data, err := io.ReadAll(f)
		if err != nil {
			return fmt.Errorf("reading file %s: %v", filePathVal, err)
		}
		if err := json.Unmarshal(data, jsonSchema); err != nil {
			return err
		}
	default:
		data, err := io.ReadAll(f)
		if err != nil {
			return fmt.Errorf("reading file %s: %v", filePathVal, err)
		}
		sniffLen := len(data)
		if sniffLen > 512 {
			sniffLen = 512
		}
		sniffedType := http.DetectContentType(data[:sniffLen])
		if !strings.HasPrefix(sniffedType, "text") {
			return fmt.Errorf("reading file %s: type %s not supported", filePathVal, sniffedType)
		}
		*parts = append(*parts, &genai.Part{Text: fmt.Sprintf("*** %s ***\n", filePathVal)})
		*parts = append(*parts, &genai.Part{Text: searchReplace(string(data), keyVals)})
	}

	return nil
}

func isHidden(name string) bool {
	return len(name) > 1 && strings.HasPrefix(name, ".")
}

// glob processes files and directories passed as argument (recursively if walk is true).
func glob(ctx context.Context, client *genai.Client, filePathVal string, parts *[]*genai.Part, sysParts *[]*genai.Part, jsonSchema *map[string]any) error {
	params, ok := ctx.Value("params").(*Parameters)
	if !ok {
		return fmt.Errorf("glob: params not found in context")
	}
	// GCS URI case
	if strings.HasPrefix(filePathVal, "gs://") {
		*parts = append(*parts, genai.NewPartFromFile(genai.File{URI: filePathVal}))
		return nil
	}
	// regular file
	fileInfo, err := os.Stat(filePathVal)
	if err == nil && fileInfo.IsDir() {
		return filepath.WalkDir(filePathVal, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if isHidden(d.Name()) {
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				if !params.Walk && path != filePathVal {
					return filepath.SkipDir
				}
				return nil
			}
			return filePathHandler(ctx, client, path, parts, sysParts, jsonSchema)
		})
	}
	matches, err := filepath.Glob(filePathVal)
	if err != nil {
		return fmt.Errorf("glob: '%s': %v", filePathVal, err)
	}
	if len(matches) == 0 {
		if _, err := os.Stat(filePathVal); os.IsNotExist(err) {
			return fmt.Errorf("directory or file not found: '%s'", filePathVal)
		}
		matches = []string{filePathVal}
	}
	// Iterate over the matches and process each file
	for _, match := range matches {
		mInfo, err := os.Stat(match)
		if err != nil {
			continue
		}
		if mInfo.IsDir() {
			err = filepath.WalkDir(match, func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if isHidden(d.Name()) {
					if d.IsDir() {
						return filepath.SkipDir
					}
					return nil
				}
				if d.IsDir() {
					if !params.Walk && path != match {
						return filepath.SkipDir
					}
					return nil
				}
				return filePathHandler(ctx, client, path, parts, sysParts, jsonSchema)
			})
			if err != nil {
				return fmt.Errorf("glob: '%s': %v", filePathVal, err)
			}
		} else {
			err := filePathHandler(ctx, client, match, parts, sysParts, jsonSchema)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// persistChat saves chat history to .gen in the current directory.
func persistChat(chat *genai.Chat) error {
	file, err := os.Create(DotGen)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	hist := chat.History(false)
	if err := encoder.Encode(hist); err != nil {
		return err
	}
	return nil
}

// retrieveHistory reads content from .gen if it exists.
func retrieveHistory(hist *[]*genai.Content) error {
	if _, err := os.Stat(DotGen); errors.Is(err, os.ErrNotExist) {
		return nil
	}
	dat, err := os.ReadFile(DotGen)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(dat, hist); err != nil {
		return err
	}
	return nil
}

// loadPrefs reads and parses .genrc from the user's home directory.
func loadPrefs(params *Parameters) error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %v", err)
	}
	prefsPath := filepath.Join(homeDir, DotGenRc)
	prefsFile, err := os.Open(prefsPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("failed to open at %s: %v", prefsPath, err)
	}
	defer prefsFile.Close()

	scanner := bufio.NewScanner(prefsFile)
	currentSection := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) == 0 || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue // skip empty line or line with comments
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			currentSection = strings.ToLower(line[1 : len(line)-1])
			continue
		}
		switch currentSection {
		case "flags":
			parts := strings.SplitN(line, "=", 2)
			if len(parts) != 2 {
				return fmt.Errorf("flag error on line: %s", line)
			}
			key := strings.TrimSpace(parts[0])
			value := strings.TrimSpace(parts[1])
			switch strings.ToLower(key) {
			case "k":
				if val, err := strconv.Atoi(value); err == nil {
					params.K = val
				}
			case "lambda":
				if val, err := strconv.ParseFloat(value, 64); err == nil {
					params.Lambda = val
				}
			case "thinkinglevel":
				val := genai.ThinkingLevel(strings.ToUpper(value))
				switch val {
				case genai.ThinkingLevelMinimal,
					genai.ThinkingLevelLow,
					genai.ThinkingLevelMedium,
					genai.ThinkingLevelHigh:
					params.ThinkingLevel = val
				}
			case "temp":
				if val, err := strconv.ParseFloat(value, 64); err == nil {
					params.Temp = val
				}
			case "timeout":
				if val, err := time.ParseDuration(value); err == nil {
					params.Timeout = val
				}
			case "topp":
				if val, err := strconv.ParseFloat(value, 64); err == nil {
					params.TopP = val
				}
			case "embmodel":
				params.EmbModel = value
			case "genmodel":
				params.GenModel = value
			default:
				return fmt.Errorf("unknown key value %s", key)
			}
		case "digestpaths":
			params.DigestPaths = append(params.DigestPaths, line)
		case "mcpservers":
			params.MCPServers = append(params.MCPServers, line)
		default:
			return fmt.Errorf("unknown section: %s", currentSection)
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}
	return nil
}

// PNGAncillaryChunkStripper wraps another io.Reader to strip ancillary chunks,
// if the data is in the PNG file format. If the data isn't PNG, it is passed
// through unmodified.
type PNGAncillaryChunkStripper struct {
	// Reader is the wrapped io.Reader.
	Reader io.Reader

	// stickyErr is the first error returned from the wrapped io.Reader.
	stickyErr error

	// buffer[rIndex:wIndex] holds data read from the wrapped io.Reader that
	// wasn't passed through yet.
	buffer [8]byte
	rIndex int
	wIndex int

	// pending and discard are the number of remaining bytes for (and whether to
	// discard or pass through) the current chunk-in-progress.
	pending int64
	discard bool

	// notPNG is set true if the data stream doesn't start with the 8-byte PNG
	// magic identifier. If true, the wrapped io.Reader's data (including the
	// first up-to-8 bytes) is passed through without modification.
	notPNG bool

	// seenMagic is whether we've seen the 8-byte PNG magic identifier.
	seenMagic bool
}

// chunkTypeAncillaryBit is whether the first byte of a big-endian uint32 chunk
// type (the first of four ASCII letters) is lower-case.
const chunkTypeAncillaryBit = 0x20000000

// Read implements io.Reader.
// Copyright 2021 The Wuffs Authors
func (r *PNGAncillaryChunkStripper) Read(p []byte) (int, error) {
	for {
		// If the wrapped io.Reader returned a non-nil error, drain r.buffer
		// (what data we have) and return that error (if fully drained).
		if r.stickyErr != nil {
			n := copy(p, r.buffer[r.rIndex:r.wIndex])
			r.rIndex += n
			if r.rIndex < r.wIndex {
				return n, nil
			}
			return n, r.stickyErr
		}

		// Handle trivial requests, including draining our buffer.
		if len(p) == 0 {
			return 0, nil
		} else if r.rIndex < r.wIndex {
			n := copy(p, r.buffer[r.rIndex:r.wIndex])
			r.rIndex += n
			return n, nil
		}

		// From here onwards, our buffer is drained: r.rIndex == r.wIndex.

		// Handle non-PNG input.
		if r.notPNG {
			return r.Reader.Read(p)
		}

		// Continue processing any PNG chunk that's in progress, whether
		// discarding it or passing it through.
		for r.pending > 0 {
			if int64(len(p)) > r.pending {
				p = p[:r.pending]
			}
			n, err := r.Reader.Read(p)
			r.pending -= int64(n)
			r.stickyErr = err
			if r.discard {
				continue
			}
			return n, err
		}

		// We're either expecting the 8-byte PNG magic identifier or the 4-byte
		// PNG chunk length + 4-byte PNG chunk type. Either way, read 8 bytes.
		r.rIndex = 0
		r.wIndex, r.stickyErr = io.ReadFull(r.Reader, r.buffer[:8])
		if r.stickyErr != nil {
			// Undo io.ReadFull converting io.EOF to io.ErrUnexpectedEOF.
			if r.stickyErr == io.ErrUnexpectedEOF {
				r.stickyErr = io.EOF
			}
			continue
		}

		// Process those 8 bytes, either:
		//  - a PNG chunk (if we've already seen the PNG magic identifier),
		//  - the PNG magic identifier itself (if the input is a PNG) or
		//  - something else (if it's not a PNG).
		if r.seenMagic {
			// The number of pending bytes is equal to (N + 4) because of the 4
			// byte trailer, a checksum.
			r.pending = int64(binary.BigEndian.Uint32(r.buffer[:4])) + 4
			chunkType := binary.BigEndian.Uint32(r.buffer[4:])
			r.discard = (chunkType & chunkTypeAncillaryBit) != 0
			if r.discard {
				r.rIndex = r.wIndex
			}
		} else if string(r.buffer[:8]) == "\x89PNG\x0D\x0A\x1A\x0A" {
			r.seenMagic = true
		} else {
			r.notPNG = true
		}
	}
}
