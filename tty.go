//go:build !windows

package main

import (
	"fmt"
	"io"
	"os"
)

func openConsole() (io.Reader, error) {
	consoleFile, err := os.Open("/dev/tty")
	if err != nil {
		return nil, fmt.Errorf("Failed to open /dev/tty")
	}
	return consoleFile, nil
}
