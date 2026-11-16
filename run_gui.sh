#!/bin/bash
# Convenient launcher for Document Scanner GUI
# Automatically picks up the Conda environment or standard python3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "/opt/homebrew/Caskroom/miniconda/base/bin/python3" ]; then
    exec "/opt/homebrew/Caskroom/miniconda/base/bin/python3" "$SCRIPT_DIR/gui.py" "$@"
elif command -v python3 &>/dev/null; then
    exec python3 "$SCRIPT_DIR/gui.py" "$@"
else
    exec python "$SCRIPT_DIR/gui.py" "$@"
fi

