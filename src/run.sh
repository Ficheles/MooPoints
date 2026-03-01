#!/usr/bin/env sh
set -eu

python /app/src/prepare_dataset.py
python /app/src/validate_annotations.py
