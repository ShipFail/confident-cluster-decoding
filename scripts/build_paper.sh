#!/bin/bash

# Build script for CCD Paper
# Usage: ./scripts/build_paper.sh

# Ensure we are in the project root
cd "$(dirname "$0")/.."

# Output directory
mkdir -p output

# Check for pandoc
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed."
    exit 1
fi

echo "Building PDF..."

export TEXINPUTS=".:./templates:"

pandoc ccd-paper.md \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+citations \
  --template=templates/neurips_2024.tex \
  --bibliography=references.bib \
  --pdf-engine=xelatex \
  --output=output/ccd-paper.pdf

if [ $? -eq 0 ]; then
    echo "Success! PDF generated at output/ccd-paper.pdf"
else
    echo "Build failed."
    exit 1
fi
