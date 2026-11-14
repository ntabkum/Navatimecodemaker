#!/usr/bin/env bash
# Builds the NavaTimecodeMaker executable for the OS you run this on.
# Run from the folder containing tc_app_v3.py:  bash build_mac_linux.sh
set -e

python3 -m pip install -r requirements.txt
python3 -m PyInstaller --onefile --clean --name NavaTimecodeMaker tc_app_v3.py

echo
echo "Build finished: dist/NavaTimecodeMaker"
if [ "$(uname)" = "Linux" ]; then
  echo "Note: Linux users need PortAudio installed (e.g. sudo apt install libportaudio2)."
fi
