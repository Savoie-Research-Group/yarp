# Open the built documentation in the default web browser.
#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Define the path to the built documentation
DOC_PATH="docs/build/html/index.html"

# Check if the documentation exists
if [ -f "$DOC_PATH" ]; then
    echo "Opening documentation...Welcome to YARP!"
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
    echo "Error: Documentation HTML file not found. Run 'yarp/yarp/docs/update_docs.sh' first."
    exit 1
fi

