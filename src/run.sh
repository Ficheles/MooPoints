#!/usr/bin/env sh
set -eu

python prepare_dataset.py
python validate_annotations.py
