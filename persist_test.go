package main

import (
	"fmt"
	"os"
	"testing"
)

func TestWriteCloseRead(t *testing.T) {
	digestPath := "tmp/"
	d, err := Open(digestPath, nil)
	if err != nil {
		t.Fatal(err)
	}

	for i := 1; i <= 10; i++ {
		rec := fmt.Sprintf("rec_%d", i)
		err = d.Write([]byte(rec))
		if err != nil {
			t.Fatalf("expected %v, got %v", nil, err)
		}
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}

	p, err := Open(digestPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	defer os.RemoveAll("tmp/")

	segs := d.Segments()
	for i := 1; i <= segs; i++ {
		for j := 0; j < 10; j++ {
			rec := fmt.Sprintf("rec_%d", j+1)
			data, err := p.Read(uint64(i), uint64(j))
			if err != nil {
				t.Fatalf("expected %v, got %v", nil, err)
			}
			if string(data) != rec {
				t.Fatalf("expected %s, got %s", rec, data)
			}
		}
	}
}

func TestCycle(t *testing.T) {
	digestPath := "tmp/"
	d, err := Open(digestPath, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	defer os.RemoveAll("tmp/")

	for i := 1; i <= 100; i++ {
		rec := fmt.Sprintf("rec_%d", i)
		if err := d.Write([]byte(rec)); err != nil {
			t.Fatal(err)
		}
	}

	if err := d.cycle(); err != nil {
		t.Fatal(err)
	}

	for i := 101; i <= 200; i++ {
		rec := fmt.Sprintf("rec_%d", i)
		if err := d.Write([]byte(rec)); err != nil {
			t.Fatal(err)
		}
	}

	segs := d.Segments()
	var lastRec string
	for i := 1; i <= segs; i++ {
		j := 0
		for {
			data, err := d.Read(uint64(i), uint64(j))
			if err != nil {
				if err == ErrEOF {
					break
				}
				t.Fatalf("expected %v, got %v", nil, err)
			}
			lastRec = string(data)
			j++
		}
	}

	if lastRec != "rec_200" {
		t.Fatalf("expected %v, got %v", "rec_200", lastRec)
	}
}
