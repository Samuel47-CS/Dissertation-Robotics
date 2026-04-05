#!/bin/bash
set -e

git submodule add https://github.com/huggingface/lerobot.git 

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

cd lerobot
uv pip install -e ".[async,feetech,intelrealsense,smolvla]"
uv pip install lerobot

cd ..
uv pip install -r training/requirements.txt

echo "Setup complete. Run 'source .venv/bin/activate' to activate!"