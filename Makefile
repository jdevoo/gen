BINARY = gen
NEWTAG := $(shell git describe --abbrev=0 --tags)
OLDTAG := $(shell git describe --abbrev=0 --tags `git rev-list --tags --skip=1 --max-count=1`)

NIX_BINARIES = linux/amd64/$(BINARY) darwin/amd64/$(BINARY)
WIN_BINARIES = windows/amd64/$(BINARY).exe
COMPRESSED_BINARIES = $(NIX_BINARIES:%=%.bz2) $(WIN_BINARIES:%.exe=%.zip)
COMPRESSED_TARGETS = $(COMPRESSED_BINARIES:%=target/%)

temp = $(subst /, ,$@)
OS = $(word 2, $(temp))
ARCH = $(word 3, $(temp))
GITHASH = $(shell git log -1 --pretty=format:"%h")
GOVER = $(word 3, $(shell go version))
LDFLAGS = -ldflags '-X main.version=$(NEWTAG) -X main.githash=$(GITHASH) -X main.golang=$(GOVER)'

# requires valid GITHUB_TOKEN
RELEASE_TOOL = gh
USER = jdevoo

all: $(BINARY)

target/linux/amd64/$(BINARY) \
target/darwin/amd64/$(BINARY) \
target/windows/amd64/$(BINARY).exe:
	CGO_ENABLED=0 GOOS=$(OS) GOARCH=$(ARCH) go build $(LDFLAGS) -o "$@"

%.bz2: %
	tar -cjf "$@" -C $(dir $<) $(BINARY)

%.zip: %.exe
	zip -j "$@" "$<"

$(BINARY):
	go build $(LDFLAGS) -o $(BINARY)

install:
	go install $(LDFLAGS)
	$(BINARY) -v

# ensure NEWTAG exists e.g. git tag v0.1
release:
	$(MAKE) $(COMPRESSED_TARGETS)
	git push && git push --tags && sleep 5
	$(foreach FILE, $(COMPRESSED_BINARIES), $(RELEASE_TOOL) release create --verify-tag --generate-notes $(NEWTAG) target/$(FILE);)

clean:
	rm -f $(BINARY)
	rm -rf target

test:
	go test -v ./...

.PHONY: all install release clean test

.DELETE_ON_ERROR:

