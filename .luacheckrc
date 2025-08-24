-- Luacheck configuration for Brainstorm mod
-- Production-ready settings with strict standards

-- Balatro globals
globals = {
    "G",
    "SMODS",
    "Brainstorm",
    "STR_PACK",
    "STR_UNPACK",
    "pseudoseed",
    "pseudorandom",
    "pseudorandom_element",
    "compress_and_save",
    "get_compressed",
    "sendDebugMessage",
    "nfs",
    "lovely"
}

-- Standard library extensions and LuaJIT FFI
read_globals = {
    "love",
    "ffi",
    "bit",
    "jit"
}

-- Performance: cache all globals
cache = true

-- Allow trailing whitespace (stylua handles this)
ignore = {
    "611", -- trailing whitespace
    "612", -- trailing whitespace in string
    "613", -- trailing whitespace in comment
    "614", -- trailing whitespace in empty line
}

-- Max line length (matching stylua config)
max_line_length = 120

-- Cyclomatic complexity threshold
max_cyclomatic_complexity = 15

-- Allow unused args with underscore prefix
unused_args = false
unused_secondaries = false
self = false

-- Exclude external libraries
exclude_files = {
    "BalatroSource/**",
    "tests/**",
    "*.min.lua"
}

-- Module-specific overrides
files["Core/logger.lua"] = {
    -- Logger can have longer lines for formatted output
    max_line_length = 150
}

files["UI/ui.lua"] = {
    -- UI code often has deeply nested callbacks
    max_cyclomatic_complexity = 20
}

-- Allow certain patterns
allow_defined_top = true