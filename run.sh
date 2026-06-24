#!/bin/bash

# Allows passing custom paths or defaulting to candidates.jsonl and submission.csv
CANDIDATES=${1:-"./candidates.jsonl"}
OUT=${2:-"./submission.csv"}

python rank.py --candidates "$CANDIDATES" --out "$OUT"
