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
		return nil, fmt.Errorf("GetStdHandle failed: %v", err)
	}
	fileType, err := windows.GetFileType(windows.Handle(handle))
	if err != nil {
		return nil, fmt.Errorf("GetFileType failed: %v", err)
	}
	var consoleFile *os.File
	// FILE_TYPE_CHAR (0x0002) indicates a character device.
	if fileType != windows.FILE_TYPE_CHAR {
		consoleFile, err = os.OpenFile("CONIN$", os.O_RDWR, 0)
		if err != nil {
			return nil, fmt.Errorf("Failed to open CONIN$: %v", err)
		}
		return consoleFile, nil
	}
	// fileType is console
	// Ensure we don't close the original handle.
	var newHandle windows.Handle
	err = windows.DuplicateHandle(windows.CurrentProcess(), handle, windows.CurrentProcess(), &newHandle, 0, false, windows.DUPLICATE_SAME_ACCESS)
	if err != nil {
		return nil, fmt.Errorf("DuplicateHandle failed: %v", err)
	}
	consoleFile = os.NewFile(uintptr(newHandle), "stdin")
	if consoleFile == nil {
		return nil, fmt.Errorf("Failed to create stdin from handle.")
	}
	return consoleFile, nil
}

func getTerminalSize() (int, int, error) {
	fd := windows.Handle(os.Stdout.Fd())
	var info windows.ConsoleScreenBufferInfo
	err := windows.GetConsoleScreenBufferInfo(fd, &info)
	if err != nil {
		return 0, 0, err
	}

	cols := int(info.Window.Right - info.Window.Left + 1)
	rows := int(info.Window.Bottom - info.Window.Top + 1)

	// estimate using standard 8x16 font dimensions
	return cols * 8, rows * 16, nil
}
