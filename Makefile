SHELL := /bin/bash

IMMO_DIR := Immolate
DLL := Immolate.dll
RELEASE_DIR := release/Brainstorm_v3.1
RELEASE_ZIP := release/Brainstorm_v3.1.zip
TARGET ?= /mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm

# Update IMMO_SOURCES when adding or removing native files.
IMMO_SOURCES := brainstorm.cpp functions.cpp rng.cpp seed.cpp util.cpp
IMMO_CPP := $(addprefix $(IMMO_DIR)/,$(IMMO_SOURCES))
IMMO_HPP := $(wildcard $(IMMO_DIR)/*.hpp)
# MOD_FILES is the complete mod payload copied to deploy/release targets.
MOD_FILES := Brainstorm.lua UI.lua config.lua lovely.toml nativefs.lua steamodded_compat.lua

.DEFAULT_GOAL := help

.PHONY: help build clean format lint release deploy

help:
	@echo "Targets:"
	@echo "  build    Build $(DLL)"
	@echo "  format   Format Lua/C++ (if tools installed)"
	@echo "  lint     Check formatting (if tools installed)"
	@echo "  release  Build and package release zip"
	@echo "  deploy   Build and copy mod to TARGET=..."
	@echo "  clean    Remove build and release artifacts"

build:
	@echo "Building $(DLL)"
	@if ! command -v x86_64-w64-mingw32-g++ >/dev/null; then \
		echo "x86_64-w64-mingw32-g++ not found. Install mingw-w64."; \
		exit 1; \
	fi
	@x86_64-w64-mingw32-g++ \
		-shared \
		-O3 \
		-std=c++17 \
		-DBUILDING_DLL \
		-o $(DLL) \
		$(IMMO_CPP) \
		-I $(IMMO_DIR) \
		-static-libgcc \
		-static-libstdc++ \
		-Wl,--export-all-symbols

clean:
	@rm -rf $(IMMO_DIR)/build $(DLL) release

format:
	@if command -v stylua >/dev/null; then stylua .; else echo "stylua not found"; fi
	@if command -v clang-format >/dev/null; then \
		clang-format -i $(IMMO_CPP) $(IMMO_HPP); \
	else echo "clang-format not found"; fi

lint:
	@if command -v stylua >/dev/null; then stylua --check .; else echo "stylua not found"; fi
	@if command -v clang-format >/dev/null; then \
		clang-format --dry-run -Werror $(IMMO_CPP) $(IMMO_HPP); \
	else echo "clang-format not found"; fi

release: build
	@rm -rf release
	@mkdir -p $(RELEASE_DIR)
	@cp $(MOD_FILES) $(RELEASE_DIR)/
	@cp $(DLL) $(RELEASE_DIR)/
	@echo "3.1" > $(RELEASE_DIR)/VERSION
	@date >> $(RELEASE_DIR)/VERSION
	@if command -v zip >/dev/null; then (cd release && zip -r Brainstorm_v3.1.zip Brainstorm_v3.1 >/dev/null); else echo "zip not found"; fi

deploy: build
	@echo "Deploying to $(TARGET)"
	@mkdir -p "$(TARGET)"
	@rm -rf "$(TARGET)/Core" "$(TARGET)/UI"
	@cp $(MOD_FILES) "$(TARGET)/"
	@cp $(DLL) "$(TARGET)/$(DLL)"
