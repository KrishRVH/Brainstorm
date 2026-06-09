SHELL := /bin/bash

IMMO_DIR := Immolate
RUST_DIR := Immolate/Rust
DLL := Immolate.dll
RELEASE_DIR := release/Brainstorm_v3.1
RELEASE_ZIP := release/Brainstorm_v3.1.zip
TARGET ?= /mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm
RUST_TARGET ?= x86_64-pc-windows-gnu
TARGET_DIR := target
CPP_DLL := $(TARGET_DIR)/cpp/$(DLL)
RUST_CARGO_TARGET_DIR := $(RUST_DIR)/target
RUST_V1_CARGO_TARGET_DIR := $(TARGET_DIR)/cargo-rust-v1
RUST_V2_CARGO_TARGET_DIR := $(TARGET_DIR)/cargo-rust-v2
RUST_V3_CARGO_TARGET_DIR := $(TARGET_DIR)/cargo-rust-v3
RUST_DLL := $(RUST_CARGO_TARGET_DIR)/$(RUST_TARGET)/release/immolate.dll
RUST_V1_DLL := $(RUST_V1_CARGO_TARGET_DIR)/$(RUST_TARGET)/release/immolate.dll
RUST_V2_DLL := $(RUST_V2_CARGO_TARGET_DIR)/$(RUST_TARGET)/release/immolate.dll
RUST_V3_DLL := $(RUST_V3_CARGO_TARGET_DIR)/$(RUST_TARGET)/release/immolate.dll
RUST_ARTIFACT_DIR := $(TARGET_DIR)/rust
RUST_ARTIFACT := $(RUST_ARTIFACT_DIR)/$(DLL)
RUST_V1_ARTIFACT_DIR := $(TARGET_DIR)/rust-v1
RUST_V1_ARTIFACT := $(RUST_V1_ARTIFACT_DIR)/$(DLL)
RUST_V2_ARTIFACT_DIR := $(TARGET_DIR)/rust-v2
RUST_V2_ARTIFACT := $(RUST_V2_ARTIFACT_DIR)/$(DLL)
RUST_V3_ARTIFACT_DIR := $(TARGET_DIR)/rust-v3
RUST_V3_ARTIFACT := $(RUST_V3_ARTIFACT_DIR)/$(DLL)
RUST_BASE_DLL ?=
RUST_CANDIDATE_DLL ?=
HARNESS_EXE := $(RUST_DIR)/target/$(RUST_TARGET)/release/immolate_dll_harness.exe
BENCH_CASE ?= all
BENCH_BUDGET ?= 1000000
BENCH_THREADS ?= 1
BENCH_REPEAT ?= 5
BENCH_WARMUP ?= 1
BENCH_MIN_RATIO ?= 0.8
BENCH_CANDIDATE_MIN_RATIO ?= 0
BENCH_CANDIDATE_MIN_SCAN_PCT ?= 0.95
BENCH_FORMAT ?= pretty
BENCH_COLOR ?= auto
DOLLAR := $$
SINGLE_QUOTE := '
DOUBLE_QUOTE := "
BACKTICK := `

define validate_dll_override
$(if $(findstring $(SINGLE_QUOTE),$(value $(1))),$(error $(1) must not contain quote/backtick/dollar characters))
$(if $(findstring $(DOUBLE_QUOTE),$(value $(1))),$(error $(1) must not contain quote/backtick/dollar characters))
$(if $(findstring $(BACKTICK),$(value $(1))),$(error $(1) must not contain quote/backtick/dollar characters))
$(if $(findstring $(DOLLAR),$(value $(1))),$(error $(1) must not contain quote/backtick/dollar characters))
endef

$(eval $(call validate_dll_override,RUST_BASE_DLL))
$(eval $(call validate_dll_override,RUST_CANDIDATE_DLL))

# Update IMMO_SOURCES when adding or removing native files.
IMMO_SOURCES := brainstorm.cpp functions.cpp rng.cpp seed.cpp util.cpp
IMMO_CPP := $(addprefix $(IMMO_DIR)/,$(IMMO_SOURCES))
IMMO_HPP := $(wildcard $(IMMO_DIR)/*.hpp)
# MOD_FILES is the complete mod payload copied to deploy/release targets.
MOD_FILES := Brainstorm.lua UI.lua config.lua lovely.toml nativefs.lua steamodded_compat.lua

.DEFAULT_GOAL := help

.PHONY: help build build-cpp build-rust build-rust-v1 build-rust-v2 build-rust-v3 build-harness format-rust-check clippy-rust check-rust-dll compare bench bench-rust bench-cpp bench-compare check-rust test-rust clean format lint release deploy

help:
	@echo "Targets:"
	@echo "  build      Build $(DLL) with the Rust implementation"
	@echo "  build-cpp  Build C++ oracle DLL to $(CPP_DLL) and $(DLL)"
	@echo "  build-rust Build Rust DLL to $(DLL)"
	@echo "  build-rust-v1 Build legacy Rust V1 DLL to $(RUST_V1_ARTIFACT)"
	@echo "  build-rust-v2 Build Rust V2 baseline DLL comparison artifact to $(RUST_V2_ARTIFACT)"
	@echo "  build-rust-v3 Build Rust V3 candidate DLL comparison artifact to $(RUST_V3_ARTIFACT)"
	@echo "  compare    Compare C++ and Rust DLL ABI results under Wine"
	@echo "  bench      Benchmark the Rust DLL under Wine"
	@echo "  bench-cpp  Benchmark the C++ DLL under Wine"
	@echo "  bench-compare Benchmark C++ then Rust DLL under Wine"
	@echo "             Knobs: BENCH_CASE=all|group|case BENCH_BUDGET=... BENCH_REPEAT=... BENCH_WARMUP=... BENCH_FORMAT=pretty|tsv RUST_BASE_DLL=... RUST_CANDIDATE_DLL=... BENCH_CANDIDATE_MIN_RATIO=..."
	@echo "  check-rust Run Rust format, clippy, tests, DLL validation, compare, and bench smoke"
	@echo "  test-rust  Run Rust unit tests"
	@echo "  format     Format Lua/C++/Rust (if tools installed)"
	@echo "  lint       Check formatting (if tools installed)"
	@echo "  release    Build and package release zip"
	@echo "  deploy     Build and copy mod to TARGET=..."
	@echo "  clean      Remove build and release artifacts"

build: build-rust

build-cpp:
	@echo "Building $(DLL)"
	@if ! command -v x86_64-w64-mingw32-g++ >/dev/null; then \
		echo "x86_64-w64-mingw32-g++ not found. Install mingw-w64."; \
		exit 1; \
	fi
	@mkdir -p "$(TARGET_DIR)/cpp"
	@x86_64-w64-mingw32-g++ \
		-shared \
		-O3 \
		-std=c++17 \
		-DBUILDING_DLL \
		-o $(CPP_DLL) \
		$(IMMO_CPP) \
		-I $(IMMO_DIR) \
		-static-libgcc \
		-static-libstdc++ \
		-Wl,--export-all-symbols
	@cp "$(CPP_DLL)" "$(DLL)"

build-rust:
	@echo "Building Rust $(DLL)"
	@if ! rustup target list --installed | grep -qx "$(RUST_TARGET)"; then \
		echo "Rust target $(RUST_TARGET) not installed. Run: rustup target add $(RUST_TARGET)"; \
		exit 1; \
	fi
	@cargo build --manifest-path $(RUST_DIR)/Cargo.toml --release --target $(RUST_TARGET)
	@mkdir -p "$(RUST_ARTIFACT_DIR)"
	@cp "$(RUST_DLL)" "$(RUST_ARTIFACT)"
	@cp "$(RUST_DLL)" "$(DLL)"

build-rust-v1:
	@echo "Building legacy Rust V1 $(DLL)"
	@if ! rustup target list --installed | grep -qx "$(RUST_TARGET)"; then \
		echo "Rust target $(RUST_TARGET) not installed. Run: rustup target add $(RUST_TARGET)"; \
		exit 1; \
	fi
	@CARGO_TARGET_DIR="$(RUST_V1_CARGO_TARGET_DIR)" cargo build --manifest-path $(RUST_DIR)/Cargo.toml --release --target $(RUST_TARGET) --features v1-legacy
	@mkdir -p "$(RUST_V1_ARTIFACT_DIR)"
	@cp "$(RUST_V1_DLL)" "$(RUST_V1_ARTIFACT)"

build-rust-v2:
	@echo "Building Rust V2 baseline $(DLL)"
	@if ! rustup target list --installed | grep -qx "$(RUST_TARGET)"; then \
		echo "Rust target $(RUST_TARGET) not installed. Run: rustup target add $(RUST_TARGET)"; \
		exit 1; \
	fi
	@CARGO_TARGET_DIR="$(RUST_V2_CARGO_TARGET_DIR)" cargo build --manifest-path $(RUST_DIR)/Cargo.toml --release --target $(RUST_TARGET) --features v2-baseline
	@mkdir -p "$(RUST_V2_ARTIFACT_DIR)"
	@cp "$(RUST_V2_DLL)" "$(RUST_V2_ARTIFACT)"

build-rust-v3:
	@echo "Building Rust V3 candidate $(DLL)"
	@if ! rustup target list --installed | grep -qx "$(RUST_TARGET)"; then \
		echo "Rust target $(RUST_TARGET) not installed. Run: rustup target add $(RUST_TARGET)"; \
		exit 1; \
	fi
	@CARGO_TARGET_DIR="$(RUST_V3_CARGO_TARGET_DIR)" cargo build --manifest-path $(RUST_DIR)/Cargo.toml --release --target $(RUST_TARGET)
	@mkdir -p "$(RUST_V3_ARTIFACT_DIR)"
	@cp "$(RUST_V3_DLL)" "$(RUST_V3_ARTIFACT)"

build-harness:
	@echo "Building Rust DLL harness"
	@if ! rustup target list --installed | grep -qx "$(RUST_TARGET)"; then \
		echo "Rust target $(RUST_TARGET) not installed. Run: rustup target add $(RUST_TARGET)"; \
		exit 1; \
	fi
	@cargo build --manifest-path $(RUST_DIR)/Cargo.toml --release --target $(RUST_TARGET) --bin immolate_dll_harness

format-rust-check:
	@cargo fmt --manifest-path $(RUST_DIR)/Cargo.toml --check

clippy-rust:
	@cargo clippy --manifest-path $(RUST_DIR)/Cargo.toml --all-targets -- -D warnings

check-rust-dll: build-rust
	@file "$(RUST_ARTIFACT)" | grep -Eq 'PE32\+.*x86-64' || { file "$(RUST_ARTIFACT)"; exit 1; }
	@for sym in brainstorm_search free_result immolate_set_log_path; do \
		x86_64-w64-mingw32-objdump -p "$(RUST_ARTIFACT)" | grep -Eq "[[:space:]]$$sym$$" || { \
			echo "missing Rust DLL export: $$sym"; \
			exit 1; \
		}; \
	done
	@exports="$$(x86_64-w64-mingw32-objdump -p "$(RUST_ARTIFACT)" | \
		sed -n '/^\[Ordinal\/Name Pointer\] Table$$/,/^$$/p' | \
		awk '/\[[[:space:]]*[0-9]+\]/{print $$NF}')"; \
	expected="$$(printf '%s\n' brainstorm_search free_result immolate_set_log_path | sort)"; \
	actual="$$(printf '%s\n' "$$exports" | sort)"; \
	if [ "$$actual" != "$$expected" ]; then \
		echo "unexpected Rust DLL exports:"; \
		printf '%s\n' "$$exports"; \
		exit 1; \
	fi
	@for bad in libgcc_s_seh-1.dll libstdc++-6.dll libwinpthread-1.dll; do \
		if x86_64-w64-mingw32-objdump -p "$(RUST_ARTIFACT)" | grep -qi "DLL Name: $$bad"; then \
			echo "Rust DLL has unshipped runtime import: $$bad"; \
			exit 1; \
		fi; \
	done

compare: build-cpp build-rust build-harness
	@base_dll="$(RUST_ARTIFACT)"; \
	if [ -n "$(value RUST_BASE_DLL)" ]; then base_dll="$(value RUST_BASE_DLL)"; fi; \
	candidate_args=(); \
	if [ -n "$(value RUST_CANDIDATE_DLL)" ]; then \
		candidate_args=(--rust-candidate "$$(winepath -w "$$(readlink -f "$(value RUST_CANDIDATE_DLL)")")"); \
	fi; \
	wine "$(HARNESS_EXE)" compare \
		--cpp "$$(winepath -w "$$(readlink -f "$(CPP_DLL)")")" \
		--rust-base "$$(winepath -w "$$(readlink -f "$$base_dll")")" \
		"$${candidate_args[@]}" \
		--threads 1

bench: bench-rust

bench-rust: build-rust build-harness
	@wine "$(HARNESS_EXE)" bench \
		--dll "$$(winepath -w "$$(readlink -f "$(RUST_ARTIFACT)")")" \
		--case "$(BENCH_CASE)" \
		--budget "$(BENCH_BUDGET)" \
		--threads "$(BENCH_THREADS)" \
		--repeat "$(BENCH_REPEAT)" \
		--warmup "$(BENCH_WARMUP)" \
		--format "$(BENCH_FORMAT)" \
		--color "$(BENCH_COLOR)"

bench-cpp: build-cpp build-harness
	@wine "$(HARNESS_EXE)" bench \
		--dll "$$(winepath -w "$$(readlink -f "$(CPP_DLL)")")" \
		--case "$(BENCH_CASE)" \
		--budget "$(BENCH_BUDGET)" \
		--threads "$(BENCH_THREADS)" \
		--repeat "$(BENCH_REPEAT)" \
		--warmup "$(BENCH_WARMUP)" \
		--format "$(BENCH_FORMAT)" \
		--color "$(BENCH_COLOR)"

bench-compare: build-cpp build-rust build-harness
	@base_dll="$(RUST_ARTIFACT)"; \
	if [ -n "$(value RUST_BASE_DLL)" ]; then base_dll="$(value RUST_BASE_DLL)"; fi; \
	candidate_args=(); \
	if [ -n "$(value RUST_CANDIDATE_DLL)" ]; then \
		candidate_args=(--rust-candidate "$$(winepath -w "$$(readlink -f "$(value RUST_CANDIDATE_DLL)")")"); \
	fi; \
	wine "$(HARNESS_EXE)" bench-compare \
		--cpp "$$(winepath -w "$$(readlink -f "$(CPP_DLL)")")" \
		--rust-base "$$(winepath -w "$$(readlink -f "$$base_dll")")" \
		"$${candidate_args[@]}" \
		--case "$(BENCH_CASE)" \
		--budget "$(BENCH_BUDGET)" \
		--threads "$(BENCH_THREADS)" \
		--repeat "$(BENCH_REPEAT)" \
		--warmup "$(BENCH_WARMUP)" \
		--min-ratio "$(BENCH_MIN_RATIO)" \
		--candidate-min-ratio "$(BENCH_CANDIDATE_MIN_RATIO)" \
		--candidate-min-scan-pct "$(BENCH_CANDIDATE_MIN_SCAN_PCT)" \
		--format "$(BENCH_FORMAT)" \
		--color "$(BENCH_COLOR)"

check-rust: format-rust-check clippy-rust test-rust check-rust-dll compare
	@$(MAKE) bench-compare BENCH_BUDGET=1000 BENCH_REPEAT=1 BENCH_CASE=pack-miss BENCH_MIN_RATIO=0.8

test-rust:
	@cargo test --manifest-path $(RUST_DIR)/Cargo.toml

clean:
	@rm -rf $(IMMO_DIR)/build $(DLL) release $(TARGET_DIR)/cpp $(RUST_ARTIFACT_DIR) $(RUST_V1_ARTIFACT_DIR) $(RUST_V2_ARTIFACT_DIR) $(RUST_V3_ARTIFACT_DIR) $(RUST_V1_CARGO_TARGET_DIR) $(RUST_V2_CARGO_TARGET_DIR) $(RUST_V3_CARGO_TARGET_DIR)

format:
	@if command -v stylua >/dev/null; then stylua .; else echo "stylua not found"; fi
	@if command -v clang-format >/dev/null; then \
		clang-format -i $(IMMO_CPP) $(IMMO_HPP); \
	else echo "clang-format not found"; fi
	@if command -v cargo >/dev/null; then cargo fmt --manifest-path $(RUST_DIR)/Cargo.toml; else echo "cargo not found"; fi

lint:
	@if command -v stylua >/dev/null; then stylua --check .; else echo "stylua not found"; fi
	@if command -v clang-format >/dev/null; then \
		clang-format --dry-run -Werror $(IMMO_CPP) $(IMMO_HPP); \
	else echo "clang-format not found"; fi
	@if command -v cargo >/dev/null; then cargo fmt --manifest-path $(RUST_DIR)/Cargo.toml --check; else echo "cargo not found"; fi
	@if command -v cargo >/dev/null; then cargo clippy --manifest-path $(RUST_DIR)/Cargo.toml --all-targets -- -D warnings; else echo "cargo not found"; fi

release: check-rust
	@rm -rf release
	@mkdir -p $(RELEASE_DIR)
	@cp $(MOD_FILES) $(RELEASE_DIR)/
	@cp $(DLL) $(RELEASE_DIR)/
	@echo "3.1" > $(RELEASE_DIR)/VERSION
	@date >> $(RELEASE_DIR)/VERSION
	@if command -v zip >/dev/null; then (cd release && zip -r Brainstorm_v3.1.zip Brainstorm_v3.1 >/dev/null); else echo "zip not found"; fi

deploy: check-rust
	@echo "Deploying to $(TARGET)"
	@mkdir -p "$(TARGET)"
	@rm -rf "$(TARGET)/Core" "$(TARGET)/UI"
	@cp $(MOD_FILES) "$(TARGET)/"
	@cp $(DLL) "$(TARGET)/$(DLL)"
