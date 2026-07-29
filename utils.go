package main

import (
	"bytes"
	"context"
	"regexp"
	"sort"
	"strings"

	"github.com/jdevoo/gen/core"
	_ "github.com/lib/pq"
	"google.golang.org/genai"
)

// conjoin returns a single text resulting from concatenation of all original parts.
// TODO handle other part types
func conjTexts(parts *[]*genai.Part) {
	if len(*parts) == 0 {
		return
	}
	var buf bytes.Buffer
	for _, p := range *parts {
		if p.Text != "" {
			buf.WriteString(string(p.Text))
		}
	}
	*parts = []*genai.Part{{Text: buf.String()}}
}

// searchReplace performs string replacement based on key-value pairs.
func searchReplace(prompt string, pm core.ParamMap) string {
	res := prompt
	for k, v := range pm {
		searchRegex := regexp.MustCompile("(?i){" + regexp.QuoteMeta(k) + "}")
		res = searchRegex.ReplaceAllString(res, v)
	}
	return res
}

// partWithKey searches prompt parts for occurrence of key and returns index.
func partWithKey(parts []*genai.Part, key string) int {
	for idx, part := range parts {
		if strings.Contains(string(part.Text), key) {
			return idx
		}
	}
	return -1
}

// replacePart returns new array with updated entry at idx.
func replacePart(parts *[]*genai.Part, idx int, key string, selection []QueryResult) {
	var keyVal string
	for _, s := range selection {
		keyVal += s.doc.content
	}
	text := (*parts)[idx].Text
	(*parts)[idx] = &genai.Part{Text: strings.Replace(string(text), key, keyVal, 1)}
}

// prependToParts extends prompts with digest selection.
func prependToParts(parts *[]*genai.Part, selection []QueryResult) {
	var res []*genai.Part
	for _, s := range selection {
		res = append(res, &genai.Part{Text: s.doc.content})
	}
	*parts = append(res, (*parts)...)
}

// appendToSelection extends selection with a query result in decreasing order of MMR up to k chunks.
func appendToSelection(selection []QueryResult, item QueryResult, k int) []QueryResult {
	result := selection
	result = append(result, item)
	sort.Slice(result[:], func(i, j int) bool {
		return result[i].mmr > result[j].mmr
	})
	if len(result) > k {
		return result[0:k]
	}
	return result
}

// processFunctionCalls attempts function calls, first across MCP sessions then gen tools.
func processFunctionCalls(ctx context.Context, fcMap map[string]*genai.FunctionCall) (*genai.Candidate, error) {
	var res []*genai.Part
	for _, fc := range fcMap {
		mcpRes, err := invokeMCPTool(ctx, fc)
		if err != nil {
			return nil, err
		}
		if mcpRes != nil {
			res = append(res, mcpRes)
			continue
		}
		// fc not an MCP tool, must be a native gen tool
		genRes, err := invokeGenTool(ctx, fc)
		if err != nil {
			return nil, err
		}
		res = append(res, genRes)
	}
	return &genai.Candidate{
		Content: &genai.Content{
			Parts: res,
		},
	}, nil
}

// countMatches is a helper that returns how many strings in strArray contain cand (case-insensitive).
func countMatches(strArray []string, cand string) int {
	count := 0
	for _, s := range strArray {
		if strings.Contains(strings.ToLower(s), strings.ToLower(cand)) {
			count++
		}
	}
	return count
}

// anyMatches returns true if any of the candidates match in array.
func anyMatches(strArray []string, candidates ...string) bool {
	for _, c := range candidates {
		if countMatches(strArray, c) > 0 {
			return true
		}
	}
	return false
}

// allMatch returns true if all list elements match.
func allMatch(strArray []string, cand string) bool {
	l := len(strArray)
	return l > 0 && countMatches(strArray, cand) == l
}

// oneMatches returns true if one and only one matches.
func oneMatches(strArray []string, cand string) bool {
	return countMatches(strArray, cand) == 1
}

// oneMatches returns true if one and only one matches.
func zeroOrOneMatches(strArray []string, cand string) bool {
	return countMatches(strArray, cand) <= 1
}

// alignSchema between JSON type fields in MCP vs genai SDK.
func alignSchema(m map[string]any) {
	if m == nil {
		return
	}
	// Handle nullable types
	if t, ok := m["type"].([]any); ok {
		isNullable := false
		var primaryType string
		for _, v := range t {
			if s, ok := v.(string); ok {
				if s == "null" {
					isNullable = true
				} else {
					primaryType = s
				}
			}
		}
		if primaryType != "" {
			m["type"] = primaryType
		}
		if isNullable {
			m["nullable"] = true
		}
	}
	// Recursively clean properties
	if props, ok := m["properties"].(map[string]any); ok {
		for _, v := range props {
			if sub, ok := v.(map[string]any); ok {
				alignSchema(sub)
			}
		}
	}
	// Handle array items
	if items, ok := m["items"].(map[string]any); ok {
		alignSchema(items)
	}
	// Handle additionalProperties
	if addProps, ok := m["additionalProperties"].(map[string]any); ok {
		alignSchema(addProps)
	}
	// Remove unsupported keywords
	delete(m, "$schema")
	delete(m, "definitions")
	delete(m, "$ref") // GenAI often struggles with local refs
}

// isValidPart evaluates whether a part contains non-empty generative payload.
func isValidPart(p *genai.Part) bool {
	if p == nil {
		return false
	}
	if p.Text != "" {
		return true
	}
	if p.InlineData != nil && len(p.InlineData.Data) > 0 {
		return true
	}
	if p.FileData != nil && p.FileData.FileURI != "" {
		return true
	}
	if p.FunctionCall != nil && p.FunctionCall.Name != "" {
		return true
	}
	if p.FunctionResponse != nil && p.FunctionResponse.Name != "" {
		return true
	}
	if p.CodeExecutionResult != nil && (p.CodeExecutionResult.Outcome != "" || p.CodeExecutionResult.Output != "") {
		return true
	}
	if p.ExecutableCode != nil && p.ExecutableCode.Code != "" {
		return true
	}
	return false
}

// isYouTubeURL validates the link passed via -f to glob.
func isYouTubeURL(path string) bool {
	youtubeRegex := regexp.MustCompile(`(?i)^((?:https?:)?//)?((?:www|m)\.)?((?:youtube(?:-nocookie)?\.com|youtu.be))(/(?:[\w\-]+\?v=|embed/|v/|shorts/|live/)?)([\w\-]+)(\S+)?$`)
	return youtubeRegex.MatchString(path)
}
