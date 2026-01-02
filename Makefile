# Makefile for sandalone Go executable

# requires valid GITHUB_TOKEN
RELEASE_TOOL = gh
USER = jdevoo

BINARY = gen
NEWTAG := $(shell git describe --abbrev=0 --tags)
OLDTAG := $(shell $(RELEASE_TOOL) release list -L 1 --json tagName -q ."[].tagName")

NIX_BINARIES = linux-amd64-$(BINARY) darwin-amd64-$(BINARY)
WIN_BINARIES = windows-amd64-$(BINARY).exe
COMPRESSED_BINARIES = $(NIX_BINARIES:%=%.bz2) $(WIN_BINARIES:%.exe=%.zip)
COMPRESSED_TARGETS = $(COMPRESSED_BINARIES:%=target/%)

temp = $(subst /, ,$@)
target = $(subst -, ,$(word 2, $(temp)))
OS = $(word 1, $(target))
ARCH = $(word 2, $(target))
GITHASH = $(shell git log -1 --pretty=format:"%h")
GOVER = $(word 3, $(shell go version))
LDFLAGS = -ldflags '-X main.Version=$(NEWTAG) -X main.Githash=$(GITHASH)'

all: $(BINARY)

target/linux-amd64-$(BINARY) \
target/darwin-amd64-$(BINARY) \
target/windows-amd64-$(BINARY).exe:
	CGO_ENABLED=0 GOOS=$(OS) GOARCH=$(ARCH) go build $(LDFLAGS) -o "$@"

%.bz2: %
	tar -cjf $@ -C $(dir $<) $(notdir $<)

%.zip: %.exe
	zip -j $@ $<

# package main
$(BINARY):
	go build $(LDFLAGS) -o $(BINARY)

install:
	go install $(LDFLAGS)
	$(BINARY) -v

# ensure NEWTAG is set e.g. git tag v0.1
release:
ifneq ($(NEWTAG),$(OLDTAG))
	$(MAKE) $(COMPRESSED_TARGETS)
	git push && git push --tags
	git log --pretty=format:"%s" $(OLDTAG)...$(NEWTAG) | $(RELEASE_TOOL) release create --verify-tag $(NEWTAG) -F - || true
	$(foreach FILE, $(COMPRESSED_BINARIES), $(RELEASE_TOOL) release upload $(NEWTAG) target/$(FILE);)
endif

clean:
	rm -f $(BINARY)
	rm -rf target

test:
	go test -v ./...

deps:
	go get -u
	go mod tidy

.PHONY: all install release clean test deps

.DELETE_ON_ERROR:

