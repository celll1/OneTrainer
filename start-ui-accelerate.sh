#!/usr/bin/env bash

set -e

# 環境変数の設定
if [ -z "$PYTHON" ]; then
    PYTHON="python"
fi

if [ -z "$VENV_DIR" ]; then
    VENV_DIR="$(dirname "$(readlink -f "$0")")/venv"
fi

# venvの確認
if [ ! -d "$VENV_DIR" ]; then
    echo "venv not found, please run install.sh first"
    exit 1
fi

# venvのアクティベート
echo "activating venv $VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "venv activated: $VENV_DIR"

# Pythonコマンドの設定
if [ -n "$PROFILE" ]; then
    PYTHON="$PYTHON -m scalene --off --cpu --gpu --profile-all --no-browser"
fi
echo "Using Python $PYTHON"

# accelerateでの起動を試みる
accelerate launch scripts/train_ui.py "$@"

# accelerateでの起動が失敗した場合は通常のPythonで起動
if [ $? -ne 0 ]; then
    echo "Failed to launch with accelerate. Launching with regular Python."
    $PYTHON scripts/train_ui.py "$@"
fi
