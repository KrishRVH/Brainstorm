#!/bin/bash
DEST="/mnt/c/Users/Krish/AppData/Roaming/Balatro/Mods/Brainstorm"

# Only update the DLL
cp -f Immolate.dll "$DEST/"
echo "✓ DLL deployed"
