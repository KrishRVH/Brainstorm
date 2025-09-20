#!/bin/bash

# Simple deployment script
echo "Deploying Brainstorm DLL..."

DEST="/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm"

# Copy DLL
cp Immolate.dll "$DEST/Immolate.dll"

echo "✓ Deployed. Remember to restart Balatro!"