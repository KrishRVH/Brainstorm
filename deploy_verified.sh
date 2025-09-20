#!/bin/bash

# Brainstorm Deployment Script with Verification
# This script ensures the deployment actually happens and verifies the files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Target directory - Windows path via WSL
TARGET_DIR="/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Brainstorm VERIFIED Deployment${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""

# Function to calculate checksum
get_checksum() {
    if [ -f "$1" ]; then
        md5sum "$1" | cut -d' ' -f1
    else
        echo "FILE_NOT_FOUND"
    fi
}

# Function to get file size
get_size() {
    if [ -f "$1" ]; then
        stat -c%s "$1"
    else
        echo "0"
    fi
}

# First, check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Creating target directory...${NC}"
    mkdir -p "$TARGET_DIR"
fi

# Store checksums before deployment
echo -e "${YELLOW}Checking current deployment state...${NC}"
OLD_DLL_CHECKSUM=$(get_checksum "$TARGET_DIR/Immolate.dll")
OLD_DLL_SIZE=$(get_size "$TARGET_DIR/Immolate.dll")

# Store new file checksums
NEW_DLL_CHECKSUM=$(get_checksum "ImmolateCPP/Immolate.dll")
NEW_DLL_SIZE=$(get_size "ImmolateCPP/Immolate.dll")

echo "Current DLL in game folder:"
echo "  Checksum: $OLD_DLL_CHECKSUM"
echo "  Size: $OLD_DLL_SIZE bytes"
echo ""
echo "New DLL to deploy:"
echo "  Checksum: $NEW_DLL_CHECKSUM"
echo "  Size: $NEW_DLL_SIZE bytes"
echo ""

if [ "$OLD_DLL_CHECKSUM" = "$NEW_DLL_CHECKSUM" ]; then
    echo -e "${YELLOW}⚠ WARNING: DLL hasn't changed! Did you forget to rebuild?${NC}"
    echo "Press Ctrl+C to cancel, or Enter to continue anyway..."
    read
fi

# Back up existing DLL if it exists
if [ -f "$TARGET_DIR/Immolate.dll" ]; then
    BACKUP_NAME="Immolate.dll.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}Backing up existing DLL to $BACKUP_NAME${NC}"
    cp "$TARGET_DIR/Immolate.dll" "$TARGET_DIR/$BACKUP_NAME"
fi

# Deploy files with verification
echo -e "${YELLOW}Deploying files...${NC}"

# Critical files that MUST deploy successfully
CRITICAL_FILES=(
    "ImmolateCPP/Immolate.dll:Immolate.dll"
    "Core/Brainstorm.lua:Core/Brainstorm.lua"
    "config.lua:config.lua"
)

# Deploy and verify each critical file
for file_mapping in "${CRITICAL_FILES[@]}"; do
    IFS=':' read -r source dest <<< "$file_mapping"
    
    if [ ! -f "$source" ]; then
        echo -e "${RED}✗ ERROR: Source file missing: $source${NC}"
        exit 1
    fi
    
    # Ensure target directory exists
    target_dir=$(dirname "$TARGET_DIR/$dest")
    mkdir -p "$target_dir"
    
    # Copy file
    cp -f "$source" "$TARGET_DIR/$dest"
    
    # Verify copy succeeded
    if [ ! -f "$TARGET_DIR/$dest" ]; then
        echo -e "${RED}✗ ERROR: Failed to deploy $dest${NC}"
        exit 1
    fi
    
    # Verify checksums match
    src_checksum=$(get_checksum "$source")
    dst_checksum=$(get_checksum "$TARGET_DIR/$dest")
    
    if [ "$src_checksum" != "$dst_checksum" ]; then
        echo -e "${RED}✗ ERROR: Checksum mismatch for $dest${NC}"
        echo "  Source: $src_checksum"
        echo "  Deployed: $dst_checksum"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} Verified: $dest ($(get_size "$TARGET_DIR/$dest") bytes)"
done

# Deploy other files (less critical but still verify)
OTHER_FILES=(
    "ImmolateCPP/seed_filter.ptx:seed_filter.ptx"
    "ImmolateCPP/seed_filter.fatbin:seed_filter.fatbin"
    "ImmolateCPP/gpu_worker.exe:gpu_worker.exe"
    "Core/logger.lua:Core/logger.lua"
    "UI/ui.lua:UI/ui.lua"
    "lovely.toml:lovely.toml"
    "nativefs.lua:nativefs.lua"
    "steamodded_compat.lua:steamodded_compat.lua"
)

for file_mapping in "${OTHER_FILES[@]}"; do
    IFS=':' read -r source dest <<< "$file_mapping"
    
    if [ -f "$source" ]; then
        target_dir=$(dirname "$TARGET_DIR/$dest")
        mkdir -p "$target_dir"
        cp -f "$source" "$TARGET_DIR/$dest"
        
        if [ -f "$TARGET_DIR/$dest" ]; then
            echo -e "${GREEN}✓${NC} Deployed: $dest"
        else
            echo -e "${YELLOW}⚠${NC} Warning: Failed to deploy $dest"
        fi
    else
        echo -e "${YELLOW}⚠${NC} Skipped (not found): $source"
    fi
done

echo ""
echo -e "${YELLOW}Final verification...${NC}"

# Verify DLL was actually updated
FINAL_DLL_CHECKSUM=$(get_checksum "$TARGET_DIR/Immolate.dll")
FINAL_DLL_SIZE=$(get_size "$TARGET_DIR/Immolate.dll")

echo "Deployed DLL:"
echo "  Checksum: $FINAL_DLL_CHECKSUM"
echo "  Size: $FINAL_DLL_SIZE bytes"

if [ "$NEW_DLL_CHECKSUM" != "$FINAL_DLL_CHECKSUM" ]; then
    echo -e "${RED}✗ ERROR: DLL deployment failed! Checksums don't match!${NC}"
    echo "  Expected: $NEW_DLL_CHECKSUM"
    echo "  Got: $FINAL_DLL_CHECKSUM"
    exit 1
fi

# Check if game might be running (could be locking DLL)
if pgrep -x "Balatro.exe" > /dev/null 2>&1; then
    echo ""
    echo -e "${YELLOW}⚠ WARNING: Balatro appears to be running!${NC}"
    echo "The game may be using the old DLL from memory."
    echo "Please restart the game for changes to take effect."
fi

# Create deployment log
DEPLOY_LOG="$TARGET_DIR/deployment.log"
echo "$(date): Deployment completed" >> "$DEPLOY_LOG"
echo "  DLL checksum: $FINAL_DLL_CHECKSUM" >> "$DEPLOY_LOG"
echo "  DLL size: $FINAL_DLL_SIZE" >> "$DEPLOY_LOG"
echo "" >> "$DEPLOY_LOG"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment VERIFIED!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "DLL deployed to: ${TARGET_DIR}/Immolate.dll"
echo "Checksum: $FINAL_DLL_CHECKSUM"
echo "Size: $FINAL_DLL_SIZE bytes"
echo ""
echo -e "${YELLOW}IMPORTANT: Restart Balatro for changes to take effect!${NC}"
echo ""

# Show recent GPU driver log entries to verify new version is running
if [ -f "$TARGET_DIR/gpu_driver.log" ]; then
    echo -e "${YELLOW}Recent GPU driver log entries:${NC}"
    tail -n 20 "$TARGET_DIR/gpu_driver.log" | grep -E "\[DLL\] Parameters:|voucher=|pack=|tag1=|tag2=" || true
fi