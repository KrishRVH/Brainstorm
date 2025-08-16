#!/bin/bash

# Deploy Brainstorm mod to Balatro folder
# Target: C:\Users\Krish\AppData\Roaming\Balatro\Mods\Brainstorm

# Convert Windows path to WSL path
TARGET="/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Deploying Brainstorm mod...${NC}"

# Create target directory if it doesn't exist
mkdir -p "$TARGET"

# Check for DLL and use the new one if available
DLL_TO_DEPLOY="Immolate.dll"
if [ -f "Immolate_new.dll" ]; then
    echo -e "${GREEN}Found enhanced DLL (Immolate_new.dll), using it${NC}"
    # Backup original if it exists
    if [ -f "$TARGET/Immolate.dll" ] && [ ! -f "$TARGET/Immolate_original.dll" ]; then
        cp "$TARGET/Immolate.dll" "$TARGET/Immolate_original.dll"
        echo -e "${YELLOW}Backed up original DLL to Immolate_original.dll${NC}"
    fi
    cp "Immolate_new.dll" "$TARGET/Immolate.dll"
    echo -e "${GREEN}✓${NC} Deployed enhanced DLL as Immolate.dll"
fi

# Files to deploy
FILES=(
    "Core/Brainstorm.lua"
    "UI/ui.lua"
    "config.lua"
    "lovely.toml"
    "nativefs.lua"
    "steamodded_compat.lua"
    "README.md"
    "CLAUDE.md"
)

# Copy each file
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$TARGET/$file" 2>/dev/null || {
            # If subdirectory doesn't exist, create it
            dir=$(dirname "$file")
            mkdir -p "$TARGET/$dir"
            cp "$file" "$TARGET/$file"
        }
        echo -e "${GREEN}✓${NC} Deployed: $file"
    else
        echo -e "${YELLOW}⚠${NC} Skipped (not found): $file"
    fi
done

# Handle DLL separately if not using enhanced version
if [ ! -f "Immolate_new.dll" ] && [ -f "Immolate.dll" ]; then
    cp "Immolate.dll" "$TARGET/Immolate.dll"
    echo -e "${GREEN}✓${NC} Deployed: Immolate.dll (original)"
fi

echo -e "${GREEN}Deployment complete!${NC}"
echo "Mod installed at: C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm"