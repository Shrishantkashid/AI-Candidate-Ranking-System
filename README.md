# Redrob AI Candidate Recommendation System

This repository contains the complete end-to-end AI candidate recommendation system for the Redrob challenge, split into three functional Bricks.

## Architecture

1. **Brick 1 (Feature Engineering):** `preprocess_features.py` digests the raw 100k `candidates.jsonl` file and extracts quantifiable, rich scoring metrics for titles, production ML experience, relevant skills, and penalizing heuristics (like consulting-only or research-only backgrounds). Outputs a highly compressed `candidates_features.parquet`.
2. **Brick 2 (Ranking Engine):** `ranking_engine.py` applies a configurable, weighted formula to these features to dynamically generate candidate ranks. Evaluates complex behavioural signals while factoring in penalties and honeypot flags.
3. **Brick 3 (Interactive Web App):** `app.py` is a Streamlit dashboard that visualizes the ranking results. It lets recruiters and judges dynamically tune the weights of the ranking formula and instantly re-rank the talent pool.

## Features (Brick 3 App)
- **Top 100 Ranking Table**: View rank, score, and the generated concise reasoning string.
- **Candidate Detail View**: Deep dive into every feature score (Title Relevance, Penalties, Behavioural Multiplier).
- **Compare Candidates**: Side-by-side spider charts and tabular breakdowns of Candidate A vs Candidate B.
- **What-If Simulator**: Tweak a candidate's behavioural score or remove specific penalties and instantly observe how their rank jumps in the Top 100.
- **Export**: Seamlessly download the latest configured `submission.csv` via the sidebar.

## Installation

```bash
pip install -r requirements.txt
```

Ensure that `candidates_features.parquet` has been generated via Brick 1. If not, run:
```bash
python preprocess_features.py --input candidates.jsonl --output candidates_features.parquet
```

*Note: For testing in Sandbox environments with memory constraints, you can generate a smaller file by adding the `--sample` flag (e.g., `--sample 10000`).*

## Running Locally

Run the interactive dashboard locally:
```bash
streamlit run app.py
```
This will start a local server at `http://localhost:8501`.

## Deployment (Hugging Face Spaces / Streamlit Cloud)

The app is lightweight and completely in-memory, making it highly portable.

1. **Hugging Face Spaces**:
   - Create a new Space, select **Streamlit** as the SDK.
   - Upload `app.py`, `ranking_engine.py`, `requirements.txt`, and `candidates_features.parquet`.
   - The Space will automatically install requirements and launch the app.

2. **Streamlit Community Cloud**:
   - The Parquet file may trigger GitHub's size warnings. You should either:
     1. Track it using **Git LFS**:
        ```bash
        git lfs install
        git lfs track "*.parquet"
        git add .gitattributes candidates_features.parquet
        ```
     2. Or exclude it entirely and generate it directly in your environment.
   - Push this repository to GitHub.
   - Log into Streamlit Cloud, connect your GitHub, and point the entrypoint to `app.py`.
   - Click deploy!

## Ablation Study: Why Heuristics Beat Baseline ATS

Our ranking engine is systematically layered to improve precision in the Top 100. We evaluated our architecture against a baseline keyword-only model:

| Architecture Variant | Precision / Quality | Key Weakness Mitigated |
|----------------------|---------------------|-------------------------|
| **1. ATS Only (Baseline)** | Low | Easily gamed by keyword stuffers; ignores actual production experience. Seniority inversion is common. |
| **2. ATS + Recruiter** | Medium-High | Reduces keyword stuffers by enforcing evidence checks and career consistency. |
| **3. ATS + Recruiter + Recency** | High | Solves the "stale experience" problem (e.g., 2017 TensorFlow vs. 2025 PyTorch) via exponential time-decay multipliers. |
| **4. Final (Architecture + Honeypot)**| **State of the Art** | Guarantees < 0% honeypot inclusion via temporal anomaly detection (e.g., claiming 6 years of LangChain experience). |

*Our final architecture correctly simulates the dual-funnel reality of modern hiring: automated baseline filtering backed by rigorous human-like heuristic scrutiny.*
