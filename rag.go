package main

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"

	"google.golang.org/genai"
)

type Document struct {
	embedding []float32
	content   string
	metadata  map[string]string
}

type QueryResult struct {
	doc Document
	mmr float32
}

// AppendToDigest saves embedding and content to the digest folder.
func appendToDigest(path string, embedding *genai.ContentEmbedding, keyVals ParamMap, onlyKvs bool, verbose bool, parts ...*genai.Part) error {
	d, err := Open(path, nil)
	if err != nil {
		genLogFatal(err)
	}
	defer d.Close()
	doc := Document{}
	if !onlyKvs {
		var content string
		for _, part := range parts {
			content += (*part).Text
		}
		doc.content = content
	}
	doc.embedding = embedding.Values
	doc.metadata = keyVals
	data, err := serializeDoc(doc)
	if err != nil {
		return err
	}
	if err := d.Write(data); err != nil {
		return err
	}
	if verbose {
		fmt.Fprintf(os.Stderr, "\033[36mcontent added.\033[0m\n")
	}
	return nil
}

// QueryDigest returns up to k documents from digest for a given query embedding based on MMR.
func queryDigest(path string, queryEmbedding *genai.ContentEmbedding, cand []QueryResult, k int, lambda float32, verbose bool) ([]QueryResult, error) {
	var selection []QueryResult
	d, err := Open(path, nil)
	if err != nil {
		genLogFatal(err)
	}
	defer d.Close()
	segs := d.Segments()
	if verbose {
		fmt.Fprintf(os.Stderr, "\033[36mReading %d segments from digest at %s\033[0m\n", segs, path)
	}
	for s := 1; s <= segs; s++ {
		for idx := 0; ; idx++ {
			var sim2 float64
			data, err := d.Read(uint64(s), uint64(idx))
			if err != nil {
				if err == ErrEOF {
					break
				}
				return []QueryResult{}, err
			}
			doc, err := deserializeDoc(data)
			if err != nil {
				return []QueryResult{}, err
			}
			for _, cs := range cand {
				sim2 = math.Max(sim2, float64(dotProduct(doc.embedding, cs.doc.embedding)))
			}
			mmr := lambda*dotProduct(queryEmbedding.Values, doc.embedding) - (1-lambda)*float32(sim2)
			selection = appendToSelection(selection, QueryResult{doc, mmr}, k)
		}
	}
	return selection, nil
}

// deserializeDoc deserializes []byte to Document.
func deserializeDoc(data []byte) (Document, error) {
	var doc Document
	buf := bytes.NewBuffer(data)

	// Deserialize embedding vector
	var embeddingLength uint64
	if err := binary.Read(buf, binary.LittleEndian, &embeddingLength); err != nil {
		return doc, fmt.Errorf("error reading embedding length: %v", err)
	}
	doc.embedding = make([]float32, embeddingLength)
	if err := binary.Read(buf, binary.LittleEndian, doc.embedding); err != nil {
		return doc, fmt.Errorf("error reading embedding: %v", err)
	}

	// Deserialize content
	var contentLength uint64
	if err := binary.Read(buf, binary.LittleEndian, &contentLength); err != nil {
		return doc, fmt.Errorf("error reading content length: %v", err)
	}
	if contentLength > 0 {
		//Decompress content
		contentBytes := make([]byte, contentLength)
		if _, err := buf.Read(contentBytes); err != nil {
			return doc, fmt.Errorf("error reading compressed content: %v", err)
		}
		r, err := gzip.NewReader(bytes.NewReader(contentBytes))
		if err != nil {
			return doc, fmt.Errorf("error creating gzip reader: %v", err)
		}
		defer r.Close()
		var decompressedContent bytes.Buffer
		if _, err := io.Copy(&decompressedContent, r); err != nil {
			return doc, fmt.Errorf("error decompressing content: %v", err)
		}
		doc.content = decompressedContent.String()

	}

	// Deserialize metadata
	var metadataLength uint64
	if err := binary.Read(buf, binary.LittleEndian, &metadataLength); err != nil {
		return doc, fmt.Errorf("error reading metadata length: %v", err)
	}
	doc.metadata = map[string]string{}
	for i := 0; i < int(metadataLength); i++ {
		var keySize, valueSize uint64
		if err := binary.Read(buf, binary.LittleEndian, &keySize); err != nil {
			return doc, fmt.Errorf("error reading key size: %v", err)
		}
		keyBytes := make([]byte, keySize)
		if _, err := buf.Read(keyBytes); err != nil {
			return doc, fmt.Errorf("error reading key: %v", err)
		}
		key := string(keyBytes)

		if err := binary.Read(buf, binary.LittleEndian, &valueSize); err != nil {
			return doc, fmt.Errorf("error reading value size: %v", err)
		}
		valueBytes := make([]byte, valueSize)
		if _, err := buf.Read(valueBytes); err != nil {
			return doc, fmt.Errorf("error reading value: %v", err)
		}
		value := string(valueBytes)
		doc.metadata[key] = value
	}

	return doc, nil
}

// serializeDoc serializes Document to []byte.
func serializeDoc(doc Document) ([]byte, error) {
	var data bytes.Buffer

	// Serialize embedding size
	if err := binary.Write(&data, binary.LittleEndian, uint64(len(doc.embedding))); err != nil {
		return nil, fmt.Errorf("error writing embedding length: %v", err)
	}

	// Serialize embedding
	if err := binary.Write(&data, binary.LittleEndian, doc.embedding); err != nil {
		return nil, fmt.Errorf("error writing embedding: %v", err)
	}

	// Serialize content and content length - with gzip compression if content length > 0
	if len(doc.content) > 0 {
		var buf bytes.Buffer
		zw := gzip.NewWriter(&buf)
		if _, err := zw.Write([]byte(doc.content)); err != nil {
			return nil, fmt.Errorf("error writing compressed content: %v", err)
		}
		if err := zw.Close(); err != nil {
			return nil, fmt.Errorf("error closing gzip writer: %v", err)
		}
		contentBytes := buf.Bytes()
		if err := binary.Write(&data, binary.LittleEndian, uint64(len(contentBytes))); err != nil {
			return nil, fmt.Errorf("error writing compressed content length: %v", err)
		}
		if _, err := data.Write(contentBytes); err != nil {
			return nil, fmt.Errorf("error writing compressed content: %v", err)
		}
	} else {
		if err := binary.Write(&data, binary.LittleEndian, uint64(0)); err != nil {
			return nil, fmt.Errorf("error writing content length: %v", err)
		}
	}

	// Serialize metadata
	if len(doc.metadata) > 0 {
		if err := binary.Write(&data, binary.LittleEndian, uint64(len(doc.metadata))); err != nil {
			return nil, fmt.Errorf("error writing metadata length: %v", err)
		}
		for k, v := range doc.metadata {
			if err := binary.Write(&data, binary.LittleEndian, uint64(len(k))); err != nil {
				return nil, fmt.Errorf("error writing key size: %v", err)
			}
			if _, err := data.Write([]byte(k)); err != nil {
				return nil, fmt.Errorf("error writing key: %v", err)
			}
			if err := binary.Write(&data, binary.LittleEndian, uint64(len(v))); err != nil {
				return nil, fmt.Errorf("error writing value size: %v", err)
			}
			if _, err := data.Write([]byte(v)); err != nil {
				return nil, fmt.Errorf("error writing value: %v", err)
			}
		}
	} else {
		if err := binary.Write(&data, binary.LittleEndian, uint64(0)); err != nil {
			return nil, fmt.Errorf("error writing metadata length: %v", err)
		}
	}

	return data.Bytes(), nil
}

// dotProduct calculates the distance between two vectors.
func dotProduct(a, b []float32) float32 {
	var dotProduct float32
	for i := range a {
		dotProduct += a[i] * b[i]
	}
	return dotProduct
}
