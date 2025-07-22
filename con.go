//go:build windows

package main

import (
	"fmt"
	"io"
	"os"

	"golang.org/x/sys/windows"
)

func openConsole() (io.Reader, error) {
	handle, err := windows.GetStdHandle(windows.STD_INPUT_HANDLE)
	if err != nil {
		return nil, fmt.Errorf("GetStdHandle failed: %w", err)
	}
	fileType, err := windows.GetFileType(windows.Handle(handle))
	if err != nil {
		return nil, fmt.Errorf("GetFileType failed: %w", err)
	}
	var consoleFile *os.File
	// FILE_TYPE_CHAR (0x0002) indicates a character device.
	if fileType != windows.FILE_TYPE_CHAR {
		consoleFile, err = os.OpenFile("CONIN$", os.O_RDWR, 0)
		if err != nil {
			return nil, fmt.Errorf("opening CONIN$: %w", err)
		}
		return consoleFile, nil
	}
	// fileType is console
	// Ensure we don't close the original handle.
	var newHandle windows.Handle
	err = windows.DuplicateHandle(windows.CurrentProcess(), handle, windows.CurrentProcess(), &newHandle, 0, false, windows.DUPLICATE_SAME_ACCESS)
	if err != nil {
		return nil, fmt.Errorf("DuplicateHandle failed: %w", err)
	}
	consoleFile = os.NewFile(uintptr(newHandle), "stdin")
	if consoleFile == nil {
		return nil, fmt.Errorf("Failed to create stdin from handle.")
	}
	return consoleFile, nil
}
