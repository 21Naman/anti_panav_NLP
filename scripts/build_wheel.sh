#!/usr/bin/env bash
# Fallback install for Kaggle: build a wheel + a source snapshot of the committed code.
#   bash scripts/build_wheel.sh            -> dist/liars-*.whl and build/kaggle_code/liars-src.tar.gz
# Upload with:  python scripts/kaggle_run.py code   (private dataset "liars-code")
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist build/kaggle_code
python -m pip wheel --no-deps -w dist . -q
mkdir -p build/kaggle_code
git archive --format=tar.gz -o build/kaggle_code/liars-src.tar.gz HEAD
ls -la dist build/kaggle_code
