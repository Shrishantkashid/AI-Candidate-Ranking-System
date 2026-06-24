# Candidate Feature Engineering Script (Brick 1)

This repository contains the `preprocess_features.py` script, which processes raw candidate data into a rich set of numeric features used by the downstream ranking engine (Brick 2).

## Overview
The script extracts key information from a provided `candidates.jsonl` file. It evaluates candidates against an array of features:
- **Title and Skill Relevance:** Scores based on keywords matching standard search, ML, and retrieval roles.
- **Production ML Experience:** Extraction of specific keywords representing real-world ML usage.
- **Consulting / Research Background penalties:** Penalty scores evaluated through heuristics identifying heavily consulting-focused or research-only backgrounds.
- **Title Hopping Penalty:** Calculates average job tenure and penalizes lower averages.
- **Behavioural Multiplier:** Integrates `redrob_signals` like response rates, notice periods, and recent activity into a multiplier for a final raw score.
- **Honeypot Detection:** Identifies potentially falsified or exaggerated profiles based on high expert proficiency relative to low overall experience.

## Installation

Ensure you have Python 3.9+ installed.

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the script directly from the command line.

```bash
python preprocess_features.py --input candidates.jsonl --output candidates_features.parquet
```

### Arguments:
- `--input`: Path to the input JSONL file. Note: The script also natively supports compressed gzipped files (`.jsonl.gz`).
- `--output`: Path to the output Parquet file. Parquet format is used for high-speed I/O and data compression.

### Outputs:
1. `candidates_features.parquet`: The processed dataset with rows corresponding to candidates, and columns including the evaluated features, raw initial score, and debugging components (`title_match_keywords`, `prod_ml_years`, `skill_set_summary`).
2. `metadata.json`: Auto-generated summary of the processing run, identifying row counts, time of processing, and exact schema of the output parquet file.

## Performance
The script is optimized to process the provided batch (100k records) efficiently on standard hardware without external API constraints. Expected completion time is under 10 minutes locally.
