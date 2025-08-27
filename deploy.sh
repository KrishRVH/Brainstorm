#!/bin/bash

# Deploy Brainstorm mod to Balatro folder
# Target: C:\Users\Krish\AppData\Roaming\Balatro\Mods\Brainstorm

# Convert Windows path to WSL path
TARGET="/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Brainstorm v3.0 Deployment Script${NC}"
echo -e "${YELLOW}========================================${NC}"

# Check if we're in the right directory
if [ ! -f "lovely.toml" ]; then
    echo -e "${RED}✗${NC} Error: Not in Brainstorm directory"
    echo "Please run from the root of the Brainstorm repository"
    exit 1
fi

# Create target directory structure
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p "$TARGET"
mkdir -p "$TARGET/Core"
mkdir -p "$TARGET/UI"

# Check and deploy DLL
echo -e "\n${YELLOW}Deploying DLL...${NC}"
if [ -f "Immolate.dll" ]; then
    DLL_SIZE=$(du -h "Immolate.dll" | cut -f1)
    cp "Immolate.dll" "$TARGET/Immolate.dll"
    
    # Check DLL size to determine if GPU support is included
    # Always deploy GPU files if they exist (regardless of size)
    if [ -f "seed_filter.ptx" ] || [ -f "gpu_worker.exe" ]; then
        echo -e "${GREEN}✓${NC} Deployed enhanced DLL with GPU support (${DLL_SIZE})"
        
        # Also deploy PTX files if they exist
        if [ -f "seed_filter.ptx" ]; then
            cp "seed_filter.ptx" "$TARGET/seed_filter.ptx"
            echo -e "${GREEN}✓${NC} Deployed CUDA kernel (PTX)"
        fi
        if [ -f "seed_filter.fatbin" ]; then
            cp "seed_filter.fatbin" "$TARGET/seed_filter.fatbin"
            echo -e "${GREEN}✓${NC} Deployed CUDA kernel (fatbin)"
        fi
        
        # Deploy GPU worker process if it exists
        if [ -f "gpu_worker.exe" ]; then
            cp "gpu_worker.exe" "$TARGET/gpu_worker.exe"
            echo -e "${GREEN}✓${NC} Deployed GPU worker process"
        fi
    else
        echo -e "${GREEN}✓${NC} Deployed CPU-only DLL (${DLL_SIZE})"
    fi
else
    echo -e "${RED}✗${NC} Error: Immolate.dll not found!"
    echo "Build it with: cd ImmolateCPP && ./build_simple.sh"
    echo "Or for GPU: cd ImmolateCPP && ./build_gpu.sh"
    exit 1
fi

# Core mod files
echo -e "\n${YELLOW}Deploying core files...${NC}"
CORE_FILES=(
    "Core/Brainstorm.lua"
    "Core/logger.lua"
    "UI/ui.lua"
    "config.lua"
    "lovely.toml"
    "nativefs.lua"
    "steamodded_compat.lua"
)

for file in "${CORE_FILES[@]}"; do
    if [ -f "$file" ]; then
        dir=$(dirname "$file")
        if [ "$dir" != "." ]; then
            mkdir -p "$TARGET/$dir"
        fi
        cp "$file" "$TARGET/$file"
        echo -e "${GREEN}✓${NC} Deployed: $file"
    else
        echo -e "${RED}✗${NC} Missing required file: $file"
        exit 1
    fi
done

# Documentation files
echo -e "\n${YELLOW}Deploying documentation...${NC}"
DOC_FILES=(
    "README.md"
    "CLAUDE.md"
)

for file in "${DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$TARGET/$file"
        echo -e "${GREEN}✓${NC} Deployed: $file"
    else
        echo -e "${YELLOW}⚠${NC} Documentation missing: $file"
    fi
done

# Optional utility files (not required for runtime)
echo -e "\n${YELLOW}Checking optional utilities...${NC}"
OPTIONAL_FILES=(
    "analyze_logs.lua"
    "run_tests.lua"
)

for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${YELLOW}ℹ${NC} Available but not deployed: $file"
    fi
done

# Check if tests directory exists
if [ -d "tests" ]; then
    echo -e "${YELLOW}ℹ${NC} Test suite available in tests/ (not deployed)"
fi

# Verify deployment
echo -e "\n${YELLOW}Verifying deployment...${NC}"
REQUIRED_COUNT=0
DEPLOYED_COUNT=0

for file in "${CORE_FILES[@]}"; do
    REQUIRED_COUNT=$((REQUIRED_COUNT + 1))
    if [ -f "$TARGET/$file" ]; then
        DEPLOYED_COUNT=$((DEPLOYED_COUNT + 1))
    fi
done

if [ $DEPLOYED_COUNT -eq $REQUIRED_COUNT ]; then
    echo -e "${GREEN}✓${NC} All core files deployed successfully ($DEPLOYED_COUNT/$REQUIRED_COUNT)"
else
    echo -e "${RED}✗${NC} Deployment incomplete ($DEPLOYED_COUNT/$REQUIRED_COUNT)"
    exit 1
fi

# Configuration check
echo -e "\n${YELLOW}Configuration status:${NC}"
if [ -f "$TARGET/config.lua" ]; then
    # Check if debug is enabled
    if grep -q "debug_enabled.*true" "$TARGET/config.lua"; then
        echo -e "${YELLOW}ℹ${NC} Debug mode is ENABLED (logs will be written to brainstorm.log)"
    else
        echo -e "${GREEN}✓${NC} Debug mode is disabled (production mode)"
    fi
    
    # Check if CUDA is enabled
    if grep -q "use_cuda.*true" "$TARGET/config.lua"; then
        echo -e "${YELLOW}ℹ${NC} GPU/CUDA is ENABLED (will attempt to use GPU acceleration)"
    else
        echo -e "${YELLOW}ℹ${NC} GPU/CUDA is disabled (CPU-only mode)"
    fi
fi

# Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Mod installed at: ${YELLOW}C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm${NC}"
echo ""
echo "In-game controls:"
echo "  Ctrl+T - Open Brainstorm settings"
echo "  Ctrl+R - Manual reroll"
echo "  Ctrl+A - Toggle auto-reroll"
echo "  Z+1-5  - Save state to slot"
echo "  X+1-5  - Load state from slot"
echo ""

# Check for updates that might be needed
if [ -f "ImmolateCPP/src/brainstorm_unified.cpp" ]; then
    if [ "ImmolateCPP/src/brainstorm_unified.cpp" -nt "Immolate.dll" ]; then
        echo -e "${YELLOW}⚠ Note:${NC} DLL source is newer than compiled DLL"
        echo "  Consider rebuilding: cd ImmolateCPP && ./build_gpu.sh"
    fi
fi

# Development tips
if [ -f "$TARGET/Core/logger.lua" ]; then
    echo "Debug tips:"
    echo "  - Enable debug mode in settings for detailed logging"
    echo "  - Logs are saved to: %AppData%\\Roaming\\Balatro\\Mods\\Brainstorm\\brainstorm.log"
    echo "  - Analyze logs with: lua analyze_logs.lua brainstorm.log"
fi

echo ""
echo -e "${GREEN}Happy seed hunting!${NC} 🎰"