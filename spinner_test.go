package main

import (
	"testing"
	"time"
)

func TestSpin(t *testing.T) {
	frames := map[string]string{
		"Box1":    `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏`,
		"Box2":    `⠄⠆⠇⠋⠙⠸⠰⠠⠰⠸⠙⠋⠇⠆`,
		"Default": ``,
	}
	for name, spinner := range frames {
		t.Run(name, func(t *testing.T) {
			show(name, spinner)
		})
	}
}

func show(name, frames string) {
	s := NewSpinner(name + " %s ")
	if frames != "" {
		s.frames = []rune(frames)
	}
	s.Start()
	defer s.Stop()
	time.Sleep(2 * time.Second)
}
