#!/bin/bash

# open_docs.sh — Open the YARP documentation
# Usage: ./open_docs.sh

cd "$(dirname "$0")"

# Path to built HTML
DOC_PATH="build/html/index.html"

# Check if it exists
if [ ! -f "$DOC_PATH" ]; then
    echo "❌ Documentation not found. Build may have failed."
    exit 1
fi

# Determine OS and open accordingly
echo "📘 Opening documentation at $DOC_PATH ..."
case "$OSTYPE" in
  darwin*)  open "$DOC_PATH" ;;                # macOS
  linux*)   xdg-open "$DOC_PATH" ;;            # Linux
  msys*|cygwin*|win32*) start "$DOC_PATH" ;;   # Git Bash or WSL on Windows
  *)        echo "📝 Please open $DOC_PATH manually." ;;
esac

echo "Welcome to
                __   __ _    ____  ____  AGAIN!
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """
