# Title: Update YARP Documentation'
# This script updates the YARP documentation by regenerating the .rst files
# and rebuilding the HTML documentation. It also converts the README.md and
# LICENSE.md files to .rst format if they exist.
#
# Note: This script must be run from the 'docs' directory. YARP must be installed in the same environment as the dependencies.
#
# Usage: ./update_docs.sh
#
# Dependencies:
# - sphinx (pip install sphinx)
# - sphinx-apidoc (pip install sphinx-autodoc)
# - sphinx-rtd-theme (pip install sphinx-rtd-theme)
# - pandoc (brew install pandoc)
# - (Optional) myst-parser (pip install myst-parser) for MyST markdown support
#
# Files that have to be manually updated as developmend proceeds:
# - docs/source/conf.py
# - docs/source/index.rst
# - docs/README.md
# - docs/LICENSE.md

# Files that automatically update:
# - docs/source/README.rst
# - docs/source/LICENSE.rst
# - docs/source/yarp.rst
# - docs/source/yarp.constants.rst
# - docs/source/yarp.enums.rst
# - docs/source/yarp.find_lewis.rst
# - docs/source/yarp.hashes.rst
# - docs/source/yarp.input_parsers.rst
# - docs/source/yarp.misc.rst
# - docs/source/yarp.properties.rst
# - docs/source/yarp.sieve.rst
# - docs/source/yarp.smiles.rst
# - docs/source/yarp.taffi_functions.rst
# - docs/source/yarp.yarpecule.rst

#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Define paths
SOURCE_DIR="source"
BUILD_DIR="build"
YARP_DIR="../../yarp"

# Ensure we're in the docs directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: This script must be run from the 'docs' directory."
    exit 1
fi

# Remove old .rst files (excluding index.rst)
echo "Removing old autodoc .rst files..."
find "$SOURCE_DIR" -type f -name "*.rst" ! -name "index.rst" -delete

# Convert README and LICENSE to .rst if they exist
if [ -f "../README.md" ]; then
    echo "Converting README.md to README.rst..."
    pandoc ../README.md -o "$SOURCE_DIR/README.rst"
elif [ -f "../README" ]; then
    echo "Converting README to README.rst..."
    pandoc ../README -o "$SOURCE_DIR/README.rst"
fi
if [ -f "../LICENSE.md" ]; then
    echo "Converting LICENSE.md to LICENSE.rst..."
    pandoc ../LICENSE.md -o "$SOURCE_DIR/LICENSE.rst"
elif [ -f "../LICENSE" ]; then
    echo "Converting LICENSE to LICENSE.rst..."
    pandoc ../LICENSE -o "$SOURCE_DIR/LICENSE.rst"
fi

# Re-run sphinx-apidoc to regenerate .rst files
#Run for YARP
echo "Generating new .rst files..."
sphinx-apidoc --force --separate -o "$SOURCE_DIR" "$YARP_DIR"


# Build the HTML documentation
echo "Building HTML documentation..."
make html > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "Documentation successfully updated."
else
    echo "Error occurred while building the documentation."
    exit 1
fi

# Open the documentation in the default web browser
DOC_PATH="$BUILD_DIR/html/index.html"

if [ -f "$DOC_PATH" ]; then
    echo "Opening documentation..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open "$DOC_PATH"  # macOS
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open "$DOC_PATH"  # Linux
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        start "$DOC_PATH"  # Windows Git Bash
    else
        echo "Documentation is available at: $DOC_PATH"
    fi
else
    echo "Error: Documentation HTML file not found."
fi