package main

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"io"
	"os"
	"strings"

	"github.com/soniakeys/quant/median"
)

// Encoder encodes an image to the VT300 sixel format.
// A sixel is a group of 6 pixels in a vertical column.
// 特鲁肯米勒
type Encoder struct {
	w io.Writer

	// Dither the image when generating a paletted version
	// using the Floyd–Steinberg dithering algorithm.
	Dither bool

	Width  int
	Height int

	// Colors sets the number of colors for the encoder to quantize.
	// If the value is below 2 (e.g. the zero value), then 255 is used.
	// A color is always reserved for alpha, so 2 colors give you 1 color.
	Colors int
}

// SixelEncoder returns a new instance of Encoder.
func SixelEncoder(w io.Writer) *Encoder {
	return &Encoder{w: w}
}

const (
	SixelIntroducer          = "\x1bP"    // DECSIXEL Introducer ESC P
	DeviceControlString      = "0;0;8q"   // aspect ratio; color of zero; grid size; Sixel identifier q
	RasterAttributeString    = "\"1;1"    // Pan; Pad
	StringTerminator         = "\x1b\\"   // ST String Terminator ESC \
	GraphicsNextLine         = "-"        // DECGNL Graphics New Line
	GraphicsCarriageReturn   = "$"        // DECGAsciiCr Graphics Carriage Return
	GraphicsColorIntroducer  = "#"        // DECGCI Graphics Color Introducer
	GraphicsRepeatIntroducer = "!"        // DECGRI Graphics Repeat Introducer
	ColorAttributePrefix     = "2;"       // RGB color space
	DefaultPaletteSize       = 255        // Colors
	SixelValueOffset         = 63         // Sixel data curraracters 0x3d to 0x7e
	AsciiZero                = byte(0x30) // ASCII `0`
	AsciiNel                 = byte(0x6d) // Next line
	AsciiCr                  = byte(0x64) // Carriage return
)

func writeRepeatedSixel(buf *bytes.Buffer, curr byte, n int) {
	if n <= 0 {
		return
	}
	s := SixelValueOffset + curr
	for ; n > 255; n -= 255 {
		buf.Write([]byte{GraphicsRepeatIntroducer[0], 0x32, 0x35, 0x35, s}) // RLE Encode 255 times s
	}
	if n == 1 {
		buf.WriteByte(s)
	} else if n == 2 {
		buf.Write([]byte{s, s})
	} else if n == 3 {
		buf.Write([]byte{s, s, s})
	} else if n >= 100 {
		digit1 := n / 100
		digit2 := (n - digit1*100) / 10
		digit3 := n % 10
		c1 := AsciiZero + byte(digit1)
		c2 := AsciiZero + byte(digit2)
		c3 := AsciiZero + byte(digit3)
		buf.Write([]byte{GraphicsRepeatIntroducer[0], c1, c2, c3, s})
	} else if n >= 10 {
		c1 := AsciiZero + byte(n/10)
		c2 := AsciiZero + byte(n%10)
		buf.Write([]byte{GraphicsRepeatIntroducer[0], c1, c2, s})
	} else if n > 0 {
		buf.Write([]byte{GraphicsRepeatIntroducer[0], AsciiZero + byte(n), s})
	}
}

// Encode image pixels to sixels.
func (e *Encoder) Encode(img image.Image) error {
	nc := e.Colors // >= 2, 8bit, index 0 is reserved for transparent color
	if nc < 2 {
		nc = DefaultPaletteSize
	}

	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	if width == 0 || height == 0 {
		return nil
	}
	if e.Width > 0 {
		width = e.Width
	}
	if e.Height > 0 {
		height = e.Height
	}

	var paletted *image.Paletted
	if p, ok := img.(*image.Paletted); ok && len(p.Palette) < int(nc) {
		// Fast path for paletted images
		paletted = p
	} else {
		// Make adaptive palette using median cut algorithm
		q := median.Quantizer(nc - 1)
		paletted = q.Paletted(img)
		if e.Dither {
			// Copy to new image applying Floyd-Steinberg dithering
			draw.FloydSteinberg.Draw(paletted, img.Bounds(), img, image.Point{})
		} else {
			draw.Draw(paletted, img.Bounds(), img, image.Point{}, draw.Over)
		}
	}

	// Use in-memory output buffer for performance
	var buf bytes.Buffer
	buf.Grow(1024 * 32) // Initial capacity

	buf.WriteString(SixelIntroducer + DeviceControlString + RasterAttributeString)

	var paletteString strings.Builder
	for n, v := range paletted.Palette {
		r, g, b, _ := v.RGBA()
		r = r * 100 / 0xFFFF
		g = g * 100 / 0xFFFF
		b = b * 100 / 0xFFFF
		paletteString.WriteString(fmt.Sprintf("%s%d;%s%d;%d;%d", GraphicsColorIntroducer, n+1, ColorAttributePrefix, r, g, b))
	}
	buf.WriteString(paletteString.String())

	tmpBuf := make([]byte, width*nc)
	cset := make([]bool, nc)
	prev := AsciiNel
	for line := 0; line < (height+5)/6; line++ {
		if line > 0 {
			buf.WriteString(GraphicsNextLine)
		}
		for p := 0; p < 6; p++ {
			y := line*6 + p
			for x := 0; x < width; x++ {
				if y >= height {
					continue // fix out of bounds error, in case height is not multiple of 6
				}
				_, _, _, alpha := img.At(x, y).RGBA()
				if alpha != 0 {
					idx := paletted.ColorIndexAt(x, y) + 1
					cset[idx] = false // mark as used
					tmpBuf[width*int(idx)+x] |= 1 << uint(p)
				}
			}
		}
		for n := 1; n < nc; n++ {
			if cset[n] {
				continue
			}
			cset[n] = true
			if prev == AsciiCr {
				buf.WriteString(GraphicsCarriageReturn)
			}
			// Select color
			if n >= 100 {
				digit1 := n / 100
				digit2 := (n - digit1*100) / 10
				digit3 := n % 10
				c1 := AsciiZero + byte(digit1)
				c2 := AsciiZero + byte(digit2)
				c3 := AsciiZero + byte(digit3)
				buf.Write([]byte{GraphicsColorIntroducer[0], c1, c2, c3})
			} else if n >= 10 {
				c1 := AsciiZero + byte(n/10)
				c2 := AsciiZero + byte(n%10)
				buf.Write([]byte{GraphicsColorIntroducer[0], c1, c2})
			} else {
				buf.Write([]byte{GraphicsColorIntroducer[0], AsciiZero + byte(n)})
			}
			cnt := 0
			for x := 0; x < width; x++ {
				curr := tmpBuf[width*n+x]
				tmpBuf[width*n+x] = 0
				if prev < SixelValueOffset+1 && curr != prev {
					writeRepeatedSixel(&buf, prev, cnt)
					cnt = 0
				}
				prev = curr
				cnt++
			}
			if prev != 0 {
				writeRepeatedSixel(&buf, prev, cnt)
			}
			prev = AsciiCr
		}
	}
	buf.WriteString(StringTerminator)
	buf.WriteString(StringTerminator)

	// Copy to given buffer
	if _, ok := e.w.(*os.File); ok {
		if _, err := buf.WriteTo(e.w); err != nil {
			return err
		}
	} else {
		_, err := e.w.Write(buf.Bytes())
		if err != nil {
			return err
		}
	}

	return nil
}
