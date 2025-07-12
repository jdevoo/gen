package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strings"
	"time"

	"image/jpeg"
	_ "image/jpeg"
	_ "image/png"

	_ "github.com/lib/pq"

	"google.golang.org/genai"
)

// hasInputFromPipe checks if input is being piped to the program.
func hasInputFromStdin(in io.Reader) bool {
	if f, ok := in.(*os.File); ok {
		fileInfo, _ := f.Stat()
		return fileInfo.Mode()&os.ModeCharDevice == 0
	} else {
		// avoid destructive io function call
		r := reflect.ValueOf(in)
		v := reflect.Indirect(r).FieldByName("s")
		return len(v.String()) > 0
	}
}

// hasOutputRedirected checks if ok to emit non-printable characters.
func hasOutputRedirected(out io.Writer) bool {
	if f, ok := out.(*os.File); ok {
		fileInfo, _ := f.Stat()
		return fileInfo.Mode()&os.ModeCharDevice == 0
	}
	return true
}

// readLine reads a line from standard input.
func readLine(r io.Reader) (string, error) {
	scanner := bufio.NewScanner(r)
	if scanner.Scan() {
		return scanner.Text(), nil
	}
	return "", scanner.Err()
}

// emitCandidates prints LLM response candidates.
func emitCandidates(out io.Writer, resp []*genai.Candidate, imgModality bool) error {
	for _, cand := range resp {
		if cand != nil && cand.Content != nil {
			for _, p := range cand.Content.Parts {
				if p.Text != "" {
					if !hasOutputRedirected(out) {
						fmt.Fprintf(out, "\033[97m%s\033[0m", p.Text)
					} else {
						if imgModality {
							fmt.Fprintf(os.Stderr, "\033[97m%s\033[0m", p.Text)
						} else {
							fmt.Fprintf(out, "%s", p.Text)
						}
					}
					continue
				}
				if p.FunctionResponse != nil {
					if !hasOutputRedirected(out) {
						fmt.Fprintf(out, "\033[97m%+v\033[0m", p.FunctionResponse)
					} else {
						fmt.Fprintf(out, "%+v", p.FunctionResponse)
					}
					continue
				}
				if p.InlineData != nil {
					reader := bytes.NewReader(p.InlineData.Data)
					img, _, err := image.Decode(reader)
					if err != nil {
						return err
					}
					if hasOutputRedirected(out) {
						// Encode to jpeg file
						if err := jpeg.Encode(out, img, &jpeg.Options{Quality: 100}); err != nil {
							return err
						}
					}
					// encode to Sixel format
					senc := SixelEncoder(os.Stderr)
					senc.Dither = true
					if err := senc.Encode(img); err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

// emitHistory prints the chat history.
// TODO improve layout
func emitHistory(out io.Writer, hist []*genai.Content) {
	var res string
	var prev string
	for _, c := range hist {
		if prev != c.Role {
			res += fmt.Sprintf("\n%s\n", c.Role)
			prev = c.Role
		}
		for _, p := range c.Parts {
			res += p.Text
		}
	}
	fmt.Fprintf(out, "%s\n", res)
}

// uploadFile tracks state until FileStateActive reached.
func uploadFile(ctx context.Context, client *genai.Client, path string) (*genai.File, error) {
	file, err := client.Files.UploadFromPath(ctx, path, nil)
	if err != nil {
		return nil, fmt.Errorf("uploading file '%s': %w", path, err)
	}

	for file.State == genai.FileStateProcessing {
		time.Sleep(1 * time.Second)
		file, err = client.Files.Get(ctx, file.Name, nil)
		if err != nil {
			return nil, fmt.Errorf("processing state for '%s': %w", file.Name, err)
		}
	}
	if file.State != genai.FileStateActive {
		return nil, fmt.Errorf("uploaded file has state '%s': %w", file.State, err)
	}
	return file, nil
}

// filePathHandler processes a single file path.
// parts and sysParts are extended with file content.
func filePathHandler(ctx context.Context, client *genai.Client, filePathVal string, parts *[]*genai.Part, sysParts *[]*genai.Part, keyVals ParamMap) error {
	f, err := os.Open(filePathVal)
	if err != nil {
		return fmt.Errorf("opening file '%s': %w", filePathVal, err)
	}
	defer f.Close()

	switch path.Ext(filePathVal) {
	case PExt, SPExt:
		data, err := io.ReadAll(f)
		if err != nil {
			return fmt.Errorf("reading file '%s': %w", filePathVal, err)
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
			return fmt.Errorf("uploading file '%s': %w", filePathVal, err)
		}
		*parts = append(*parts, &genai.Part{FileData: &genai.FileData{
			FileURI:  file.URI,
			MIMEType: strings.Split(file.MIMEType, ";")[0],
		}})
	default:
		data, err := io.ReadAll(f)
		if err != nil {
			return fmt.Errorf("reading file %s: %w", filePathVal, err)
		}
		*parts = append(*parts, &genai.Part{Text: searchReplace(string(data), keyVals)})
	}

	return nil
}

func glob(ctx context.Context, client *genai.Client, filePathVal string, parts *[]*genai.Part, sysParts *[]*genai.Part, keyVals ParamMap) error {
	fileInfo, err := os.Stat(filePathVal)
	if err == nil && fileInfo.IsDir() {
		filePathVal = filepath.Join(filePathVal, "**/*")
	}
	matches, err := filepath.Glob(filePathVal)
	if err != nil {
		return fmt.Errorf("globbing '%s': %w", filePathVal, err)
	}
	if len(matches) == 0 {
		if _, err := os.Stat(filePathVal); os.IsNotExist(err) {
			return fmt.Errorf("directory or file not found: '%s'", filePathVal)
		}
	}
	// Iterate over the matches and process each file
	for _, match := range matches {
		err := filePathHandler(ctx, client, match, parts, sysParts, keyVals)
		if err != nil {
			return err
		}
	}
	return nil
}

// persistChat saves chat history to .gen in the current directory
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
