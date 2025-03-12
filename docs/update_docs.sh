# Title: Update YARP Documentation'
#This script updates the YARP documentation by regenerating the .rst files
#and rebuilding the HTML documentation. It also converts the README.md and
#LICENSE.md files to .rst format if they exist.
#
#Dependencies:
#- sphinx (pip install sphinx)
#- sphinx-apidoc (pip install sphinx-autodoc)
#- sphinx-rtd-theme (pip install sphinx-rtd-theme)
#- pandoc (brew install pandoc)
#- (Optional) myst-parser (pip install myst-parser) for MyST markdown support
#

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