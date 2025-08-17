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

# Deploy the enhanced DLL (now the only version)
if [ -f "Immolate.dll" ]; then
    cp "Immolate.dll" "$TARGET/Immolate.dll"
    echo -e "${GREEN}✓${NC} Deployed enhanced DLL"
else
    echo -e "${YELLOW}⚠${NC} Warning: Immolate.dll not found"
    echo "Build it with: cd ImmolateCPP && ./build_simple.sh"
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


echo -e "${GREEN}Deployment complete!${NC}"
echo "Mod installed at: C:\\Users\\Krish\\AppData\\Roaming\\Balatro\\Mods\\Brainstorm"