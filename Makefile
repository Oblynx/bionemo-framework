# Makefile for building and pushing the JupyterHub BioNeMo image

REGISTRY ?= ghcr.io
ORG      ?= oblynx
REPO     ?= bionemo-framework
IMAGE    ?= jhub
DOCKERFILE ?= Dockerfile.jhub
# Build args (space separated), e.g.: BUILD_ARGS=--build-arg INSTALL_SSH=1
BUILD_ARGS ?=
# Platform (single arch to avoid emulation overhead in local builds)
PLATFORM ?= linux/amd64
# Additional labels (space separated KEY=VALUE). The source label ensures association with repo.
EXTRA_LABELS ?=
SOURCE_LABEL = org.opencontainers.image.source="https://github.com/Oblynx/$(REPO)"
# Default tag base
BASE_TAG ?= nightly

# Computed values
DATE    := $(shell date -u +%Y%m%d)
SHORT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
VERSION_TAG := $(BASE_TAG)-$(DATE)-$(SHORT_SHA)
IMAGE_REPO := $(REGISTRY)/$(ORG)/$(REPO)/$(IMAGE)
IMAGE_VERSION := $(IMAGE_REPO):$(VERSION_TAG)
IMAGE_LATEST  := $(IMAGE_REPO):$(BASE_TAG)
LABEL_ARGS := $(foreach L,$(SOURCE_LABEL) $(EXTRA_LABELS),--label $(L))

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build          - Build image with version tag $(VERSION_TAG)"
	@echo "  tag            - Add rolling '$(BASE_TAG)' tag to built image"
	@echo "  push           - Push version + rolling tags"
	@echo "  push-version   - Push only the versioned tag"
	@echo "  push-latest    - Push only the rolling tag ($(BASE_TAG))"
	@echo "  login          - Docker login to GHCR (requires GHCR_PAT env var)"
	@echo "  inspect        - Show local image metadata"
	@echo "  run            - Run container locally (Lab on port $$TEST_PORT or 8888)"
	@echo "  clean          - Remove local image tags"
	@echo "Variables (override): REGISTRY ORG REPO IMAGE BASE_TAG PLATFORM BUILD_ARGS EXTRA_LABELS"

# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
.PHONY: build
build:
	@echo "Building $(IMAGE_VERSION) (platform=$(PLATFORM))"
	docker build \
	  --network=host \
	  --platform $(PLATFORM) \
	  -f $(DOCKERFILE) \
	  $(BUILD_ARGS) \
	  $(LABEL_ARGS) \
	  -t $(IMAGE_VERSION) .

.PHONY: tag
tag: build
	@echo "Tagging $(IMAGE_VERSION) as $(IMAGE_LATEST)"
	docker tag $(IMAGE_VERSION) $(IMAGE_LATEST)

# ------------------------------------------------------------------
# Push
# ------------------------------------------------------------------
.PHONY: push push-version push-latest
push: tag push-version push-latest

push-version:
	docker push $(IMAGE_VERSION)

push-latest:
	docker push $(IMAGE_LATEST)

# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------
.PHONY: login
login:
	@if [ -z "$$GHCR_PAT" ]; then echo "GHCR_PAT env var not set" >&2; exit 1; fi
	echo "$$GHCR_PAT" | docker login $(REGISTRY) -u $(ORG) --password-stdin

# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
.PHONY: inspect
inspect:
	docker image inspect $(IMAGE_VERSION) || docker image inspect $(IMAGE_LATEST) || echo "Image not found locally"

TEST_PORT ?= 8888
.PHONY: run
run: tag
	docker run --rm -it -p $(TEST_PORT):8888 $(IMAGE_LATEST) jupyter lab --LabApp.token=''

.PHONY: clean
clean:
	-docker rmi $(IMAGE_VERSION) 2>/dev/null || true
	-docker rmi $(IMAGE_LATEST) 2>/dev/null || true

# ------------------------------------------------------------------
# Convenience: print resolved tags
# ------------------------------------------------------------------
.PHONY: print-tags
print-tags:
	@echo VERSION_TAG=$(VERSION_TAG)
	@echo IMAGE_VERSION=$(IMAGE_VERSION)
	@echo IMAGE_LATEST=$(IMAGE_LATEST)
