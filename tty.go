//go:build !windows

package main

import (
	"fmt"
	"io"
	"os"

	"golang.org/x/sys/unix"
)

func openConsole() (io.Reader, error) {
	consoleFile, err := os.Open("/dev/tty")
	if err != nil {
		return nil, fmt.Errorf("Failed to open /dev/tty")
	}
	return consoleFile, nil
}

// getTerminalSize returns (width, height) in pixels.
func getTerminalSize() (int, int, error) {
	fds := []uintptr{os.Stdout.Fd(), os.Stderr.Fd(), os.Stdin.Fd()}

	// stdout is redirected, we can also try to get terminal size from /dev/tty
	if tty, err := os.Open("/dev/tty"); err == nil {
		defer tty.Close()
		fds = append([]uintptr{tty.Fd()}, fds...)
	}

	var ws *unix.Winsize
	var err error

	for _, fd := range fds {
		ws, err = unix.IoctlGetWinsize(int(fd), unix.TIOCGWINSZ)
		if err == nil && ws != nil {
			break
		}
	}

	if err != nil || ws == nil {
		return 0, 0, err
	}

	widthPx := int(ws.Xpixel)
	heightPx := int(ws.Ypixel)

	// the terminal doesn't report raw pixels
	if widthPx == 0 {
		widthPx = int(ws.Col) * 8 // typical character column pixel width
	}
	if heightPx == 0 {
		heightPx = int(ws.Row) * 16 // typical character row pixel height
	}

	return widthPx, heightPx, nil
}
