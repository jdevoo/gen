package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/google/generative-ai-go/genai"
)

type Document struct {
	Embedding []float32
	Content   string
	Metadata  map[string]string
}

// AppendToDigest saves embedding and content to the digest folder.
func AppendToDigest(path string, embedding []float32, keyVals ParamMap, verbose bool, parts ...genai.Part) error {
	d, err := Open(path, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer d.Close()
	var content string
	for _, part := range parts {
		content += fmt.Sprintf("%s", part)
	}
	doc := Document{
		Embedding: embedding,
		Content:   content,
		Metadata:  keyVals,
	}
	if verbose {
		fmt.Fprintf(os.Stderr, "%v", doc)
	}
	data, err := serializeDoc(doc)
	if err != nil {
		return err
	}
	if err := d.Write(data); err != nil {
		return err
	}
	return nil
}

// QueryDigest returns content for given query embedding.
func QueryDigest(path string, queryEmbedding []float32, verbose bool) (string, error) {
	d, err := Open(path, nil)
	if err != nil {
		log.Fatal(err)
	}
	defer d.Close()
	var result string
	segs := d.Segments()
	for s := 1; s <= segs; s++ {
		idx := 0
		minDist := float32(math.MaxFloat32)
		for {
			data, err := d.Read(uint64(s), uint64(idx))
			if err != nil {
				if err == ErrEOF {
					break
				}
				return "", err
			}
			doc, err := deserializeDoc(data)
			if err != nil {
				return "", err
			}
			// as vectors are normalized dot product is the cosine similarity
			dist := dotProduct(queryEmbedding, doc.Embedding)
			if dist < minDist {
				result = doc.Content
				minDist = dist
			}
			idx++
		}
	}
	return result, nil
}

// deserializeDoc deserializes []byte to Document
func deserializeDoc(data []byte) (Document, error) {
	var doc Document
	buf := bytes.NewBuffer(data)

	// Deserialize embedding vector
	var embeddingLength uint64
	if err := binary.Read(buf, binary.LittleEndian, &embeddingLength); err != nil {
		return doc, fmt.Errorf("error reading embedding length: %w", err)
	}
	doc.Embedding = make([]float32, embeddingLength)
	if err := binary.Read(buf, binary.LittleEndian, doc.Embedding); err != nil {
		return doc, fmt.Errorf("error reading embedding: %w", err)
	}

	// Deserialize content
	var contentLength uint64
	if err := binary.Read(buf, binary.LittleEndian, &contentLength); err != nil {
		return doc, fmt.Errorf("error reading content length: %w", err)
	}
	contentBytes := make([]byte, contentLength)
	if _, err := buf.Read(contentBytes); err != nil {
		return doc, fmt.Errorf("error reading content: %w", err)
	}
	doc.Content = string(contentBytes)

	// Deserialize metadata
	var metadataLength uint64
	if err := binary.Read(buf, binary.LittleEndian, &metadataLength); err != nil {
		return doc, fmt.Errorf("error reading metadata length: %w", err)
	}
	doc.Metadata = make(map[string]string)
	for i := 0; i < int(metadataLength); i++ {
		var keySize, valueSize uint64
		if err := binary.Read(buf, binary.LittleEndian, &keySize); err != nil {
			return doc, fmt.Errorf("error reading key size: %w", err)
		}
		keyBytes := make([]byte, keySize)
		if _, err := buf.Read(keyBytes); err != nil {
			return doc, fmt.Errorf("error reading key: %w", err)
		}
		key := string(keyBytes)

		if err := binary.Read(buf, binary.LittleEndian, &valueSize); err != nil {
			return doc, fmt.Errorf("error reading value size: %w", err)
		}
		valueBytes := make([]byte, valueSize)
		if _, err := buf.Read(valueBytes); err != nil {
			return doc, fmt.Errorf("error reading value: %w", err)
		}
		value := string(valueBytes)
		doc.Metadata[key] = value
	}

	return doc, nil
}

// serializeDoc serializes Document to []byte
func serializeDoc(doc Document) ([]byte, error) {
	var data bytes.Buffer

	// Serialize embedding size
	if err := binary.Write(&data, binary.LittleEndian, uint64(len(doc.Embedding))); err != nil {
		return nil, fmt.Errorf("error writing embedding length: %w", err)
	}

	// Serialize embedding
	if err := binary.Write(&data, binary.LittleEndian, doc.Embedding); err != nil {
		return nil, fmt.Errorf("error writing embedding: %w", err)
	}

	// Serialize content length
	if err := binary.Write(&data, binary.LittleEndian, uint64(len(doc.Content))); err != nil {
		return nil, fmt.Errorf("error writing content length: %w", err)
	}

	// Serialize content
	if _, err := data.Write([]byte(doc.Content)); err != nil {
		return nil, fmt.Errorf("error writing content: %w", err)
	}

	// Serialize metadata
	if err := binary.Write(&data, binary.LittleEndian, uint64(len(doc.Metadata))); err != nil {
		return nil, fmt.Errorf("error writing metadata length: %w", err)
	}
	for k, v := range doc.Metadata {
		if err := binary.Write(&data, binary.LittleEndian, uint64(len(k))); err != nil {
			return nil, fmt.Errorf("error writing key size: %w", err)
		}
		if _, err := data.Write([]byte(k)); err != nil {
			return nil, fmt.Errorf("error writing key: %w", err)
		}
		if err := binary.Write(&data, binary.LittleEndian, uint64(len(v))); err != nil {
			return nil, fmt.Errorf("error writing value size: %w", err)
		}
		if _, err := data.Write([]byte(v)); err != nil {
			return nil, fmt.Errorf("error writing value: %w", err)
		}
	}

	return data.Bytes(), nil
}
