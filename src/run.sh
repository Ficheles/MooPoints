#!/usr/bin/env sh
set -eu

python src/keypoints/prepare_dataset.py
python src/keypoints/validate_annotations.py
