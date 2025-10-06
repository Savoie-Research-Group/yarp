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
#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Define paths
SOURCE_DIR="source"
BUILD_DIR="build"
YARP_DIR="../yarp"
#TEST_DIR="../test"

# Ensure we're in the docs directory
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: This script must be run from the 'docs' directory."
    exit 1
fi

# Remove old .rst files (excluding index.rst)
echo "Removing old autodoc .rst files..."
find "$SOURCE_DIR" -type f -name "*.rst" ! -name "index.rst" -delete

# Clean out auto-generated API docs
# Clean out auto-generated API docs
echo "Removing old autoapi files..."
rm -rf "$SOURCE_DIR/autoapi"
rm -rf "$BUILD_DIR/doctrees"  # Prevent stale reference warnings



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
#echo "Generating new .rst files..."
sphinx-apidoc --force --module-first --no-toc -o "$SOURCE_DIR/autoapi" "$YARP_DIR" --separate

#sphinx-apidoc --force --separate --no-toc -o "$SOURCE_DIR" "../test"

# Build the HTML documentation
echo "Building HTML documentation..."

#Silent Mode:
make html > /dev/null 2>&1

#Debugging Mode:
#make html

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

echo "Welcome to
                __   __ _    ____  ____  AGAIN!
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """
