package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"sync"
)

type Options struct {
	// NoSync disables fsync after writes. This is less durable and puts the
	// log at risk of data loss when there's a server crash.
	NoSync      bool
	SegmentSize int // SegmentSize of each segment. Default is 20 MB.
	DirPerms    os.FileMode
	FilePerms   os.FileMode
}

func (o *Options) validate() {
	if o.SegmentSize <= 0 {
		o.SegmentSize = DefaultOptions.SegmentSize
	}

	if o.DirPerms == 0 {
		o.DirPerms = DefaultOptions.DirPerms
	}

	if o.FilePerms == 0 {
		o.FilePerms = DefaultOptions.FilePerms
	}
}

var DefaultOptions = &Options{
	NoSync:      false,    // Fsync after every write
	SegmentSize: 20971520, // 20 MB log segment files.
	DirPerms:    0750,
	FilePerms:   0640,
}

var ErrEOF = errors.New("end of file reached while reading from log")

// Log represents a append only log.
type Log struct {
	mu       sync.RWMutex
	path     string     // absolute path to log directory
	segments []*segment // all known log segments
	sfile    *os.File   // tail segment file handle
	wbatch   Batch      // reusable write batch

	opts    Options
	closed  bool
	corrupt bool
}

// segment represents a single segment file.
type segment struct {
	path  string // path of segment file
	index uint64 // first index of segment
	cbuf  []byte // cached entries buffer
	cpos  []bpos // position of entries in buffer
}

type bpos struct {
	pos int // byte position
	end int // one byte past pos
}

type Batch struct {
	entries []batchEntry
	data    []byte
}

type batchEntry struct {
	size int
}

func Open(path string, opts *Options) (*Log, error) {
	if opts == nil {
		opts = DefaultOptions
	}
	opts.validate()

	var err error
	path, err = filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	l := &Log{path: path, opts: *opts}
	if err := os.MkdirAll(path, l.opts.DirPerms); err != nil {
		return nil, err
	}
	if err := l.load(); err != nil {
		return nil, err
	}
	return l, nil
}

func (l *Log) load() error {
	files, err := os.ReadDir(l.path)
	if err != nil {
		return err
	}

	for _, file := range files {
		name := file.Name()

		if file.IsDir() || len(name) < 20 {
			continue
		}

		index, err := strconv.ParseUint(name[:20], 10, 64)
		if err != nil || index == 0 {
			continue
		}

		if len(name) == 20 {
			l.segments = append(l.segments, &segment{
				index: index,
				path:  filepath.Join(l.path, name),
			})
		}
	}

	if len(l.segments) == 0 {
		// Create a new log
		l.segments = append(l.segments, &segment{
			index: 1,
			path:  filepath.Join(l.path, segmentName(1)),
		})
		l.sfile, err = os.OpenFile(l.segments[0].path, os.O_CREATE|os.O_RDWR|os.O_TRUNC, l.opts.FilePerms)
		return err
	}

	// Open the last segment for appending
	lseg := l.segments[len(l.segments)-1]
	l.sfile, err = os.OpenFile(lseg.path, os.O_WRONLY, l.opts.FilePerms)
	if err != nil {
		return err
	}

	if _, err := l.sfile.Seek(0, 2); err != nil {
		return err
	}

	// Load the last segment entries
	if err := l.loadSegmentEntries(lseg); err != nil {
		return err
	}

	return nil
}

func segmentName(index uint64) string {
	return fmt.Sprintf("%020d", index)
}

func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		if l.corrupt {
			return fmt.Errorf("Closing corrupt log")
		}
		return fmt.Errorf("Closing already closed log")
	}
	if err := l.sfile.Sync(); err != nil {
		return err
	}
	if err := l.sfile.Close(); err != nil {
		return err
	}
	l.closed = true
	if l.corrupt {
		return fmt.Errorf("Closing corrupt log")
	}
	return nil
}

func (l *Log) Write(data []byte) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.corrupt {
		return fmt.Errorf("Writing to corrupt log")
	} else if l.closed {
		return fmt.Errorf("Writing to closed log")
	}
	l.wbatch.Clear()
	l.wbatch.Write(data)
	return l.writeBatch(&l.wbatch)
}

func (l *Log) appendEntry(dst []byte, data []byte) (out []byte, cpos bpos) {
	return appendBinaryEntry(dst, data)
}

func (l *Log) cycle() error {
	if err := l.sfile.Sync(); err != nil {
		return err
	}
	if err := l.sfile.Close(); err != nil {
		return err
	}

	nidx := l.segments[len(l.segments)-1].index + 1
	s := &segment{
		index: nidx,
		path:  filepath.Join(l.path, segmentName(nidx)),
	}
	var err error
	l.sfile, err = os.OpenFile(s.path, os.O_CREATE|os.O_RDWR|os.O_TRUNC, l.opts.FilePerms)
	if err != nil {
		return err
	}
	l.segments = append(l.segments, s)
	return nil
}

func appendBinaryEntry(dst []byte, data []byte) (out []byte, cpos bpos) {
	// data_size + data
	pos := len(dst)
	dst = appendUvarint(dst, uint64(len(data)))
	dst = append(dst, data...)
	return dst, bpos{pos, len(dst)}
}

func appendUvarint(dst []byte, x uint64) []byte {
	var buf [10]byte
	n := binary.PutUvarint(buf[:], x)
	dst = append(dst, buf[:n]...)
	return dst
}

func (b *Batch) Write(data []byte) {
	b.entries = append(b.entries, batchEntry{len(data)})
	b.data = append(b.data, data...)
}

func (b *Batch) Clear() {
	b.entries = b.entries[:0]
	b.data = b.data[:0]
}

func (l *Log) WriteBatch(b *Batch) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.corrupt {
		return fmt.Errorf("Batch write to corrput log")
	} else if l.closed {
		return fmt.Errorf("Batch write to closed log")
	}
	if len(b.data) == 0 {
		return nil
	}
	return l.writeBatch(b)
}

func (l *Log) writeBatch(b *Batch) error {
	// load the tail segment
	s := l.segments[len(l.segments)-1]
	if len(s.cbuf) > l.opts.SegmentSize {
		// tail segment has reached capacity. Close it and create a new one.
		if err := l.cycle(); err != nil {
			return err
		}
		s = l.segments[len(l.segments)-1]
	}

	mark := len(s.cbuf)
	data := b.data
	for i := 0; i < len(b.entries); i++ {
		bytes := data[:b.entries[i].size]
		var cpos bpos
		s.cbuf, cpos = l.appendEntry(s.cbuf, bytes)
		s.cpos = append(s.cpos, cpos)
		if len(s.cbuf) >= l.opts.SegmentSize {
			// segment has reached capacity, cycle now
			if _, err := l.sfile.Write(s.cbuf[mark:]); err != nil {
				return err
			}
			if err := l.cycle(); err != nil {
				return err
			}
			s = l.segments[len(l.segments)-1]
			mark = 0
		}
		data = data[b.entries[i].size:]
	}
	if len(s.cbuf)-mark > 0 {
		if _, err := l.sfile.Write(s.cbuf[mark:]); err != nil {
			return err
		}
	}

	if !l.opts.NoSync {
		if err := l.sfile.Sync(); err != nil {
			return err
		}
	}

	b.Clear()
	return nil
}

// findSegment performs a bsearch on the segments
func (l *Log) findSegment(index uint64) int {
	i, j := 0, len(l.segments)
	for i < j {
		h := i + (j-i)/2
		if index >= l.segments[h].index {
			i = h + 1
		} else {
			j = h
		}
	}
	return i - 1
}

func (l *Log) loadSegmentEntries(s *segment) error {
	data, err := os.ReadFile(s.path)
	if err != nil {
		return err
	}
	ebuf := data
	var cpos []bpos
	var pos int
	for len(data) > 0 {
		var n int
		n, err = loadNextBinaryEntry(data)
		if err != nil {
			return err
		}
		data = data[n:]
		cpos = append(cpos, bpos{pos, pos + n})
		pos += n
	}
	s.cbuf = ebuf
	s.cpos = cpos
	return nil
}

func loadNextBinaryEntry(data []byte) (n int, err error) {
	// data_size + data
	size, n := binary.Uvarint(data)
	if n <= 0 {
		return 0, fmt.Errorf("Log corrupt: unable to read entry size")
	}
	if uint64(len(data)-n) < size {
		return 0, fmt.Errorf("Log corrupt: entry size exceeds available data")
	}
	return n + int(size), nil
}

func (l *Log) loadSegment(index uint64) (*segment, error) {
	// check the last segment first.
	lseg := l.segments[len(l.segments)-1]
	if index >= lseg.index {
		return lseg, nil
	}
	// find in the segment array
	idx := l.findSegment(index)
	s := l.segments[idx]
	if len(s.cpos) == 0 {
		// load the entries from cache
		if err := l.loadSegmentEntries(s); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (l *Log) Segments() int {
	return len(l.segments)
}

func (l *Log) Read(segment, index uint64) (data []byte, err error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.corrupt {
		return nil, fmt.Errorf("Reading from corrupt log")
	} else if l.closed {
		return nil, fmt.Errorf("Reading from closed log")
	}
	if segment == 0 {
		return nil, fmt.Errorf("Segment not found while reading from log")
	}
	s, err := l.loadSegment(segment)
	if err != nil {
		return nil, err
	}

	if int(index) >= len(s.cpos) {
		return nil, ErrEOF
	}
	cpos := s.cpos[index]
	edata := s.cbuf[cpos.pos:cpos.end]
	// binary read
	size, n := binary.Uvarint(edata)
	if n <= 0 {
		return nil, fmt.Errorf("Log corrupt: unable to read entry size")
	}
	if uint64(len(edata)-n) < size {
		return nil, fmt.Errorf("Log corrupt: entry size exceeds available data")
	}
	data = make([]byte, size)
	copy(data, edata[n:])
	return data, nil
}

func (l *Log) Sync() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.corrupt {
		return fmt.Errorf("Syncing corrupt log")
	} else if l.closed {
		return fmt.Errorf("Syncing closed log")
	}
	return l.sfile.Sync()
}
