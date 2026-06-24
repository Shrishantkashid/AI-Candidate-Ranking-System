import pandas as pd
import numpy as np

DEFAULT_WEIGHTS = {
    "w_ats": 0.30,
    "w_recruiter": 0.25,
    "w_prod_ml": 0.20,
    "w_skill": 0.15,
    "w_behaviour": 0.10,
    "w_penalty": 0.10
}

from reasoning_generator import generate_human_reasoning

def generate_reasoning(row):
    return generate_human_reasoning(row)

def rank_candidates(features_df, weights=None, top_n=100):
    """
    Ranks candidates based on a weighted sum of feature scores minus penalties,
    multiplied by the behavioural multiplier.
    
    Parameters:
        features_df (pd.DataFrame): DataFrame containing extracted features.
        weights (dict): Weight overrides for the scoring formula. 
        top_n (int): Number of top candidates to return.
        
    Returns:
        pd.DataFrame: Top candidates with candidate_id, rank, score, reasoning.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
        
    df = features_df.copy()
    
    # Fill NA to prevent propagation of NaN in math
    fill_cols = [
        "title_relevance", "prod_ml_experience_score", "skill_relevance_score",
        "consulting_only_penalty", "research_only_penalty", "title_hopping_penalty",
        "langchain_only_penalty", "location_penalty", "behavioural_multiplier",
        "honeypot_flag", "ats_score", "recruiter_score"
    ]
    for col in fill_cols:
        if col in df.columns:
            if col == "honeypot_flag":
                df[col] = df[col].fillna(False)
            else:
                df[col] = df[col].fillna(0.0)
    
    ats_norm = df.get("ats_score", 0.0) / 100.0
    recruiter_norm = df.get("recruiter_score", 0.0) / 100.0
    
    # Calculate weighted components
    base_score = (
        weights.get("w_ats", 0.30) * ats_norm +
        weights.get("w_recruiter", 0.25) * recruiter_norm +
        weights.get("w_prod_ml", 0.20) * df.get("prod_ml_experience_score", 0.0) +
        weights.get("w_skill", 0.15) * df.get("skill_relevance_score", 0.0) +
        weights.get("w_behaviour", 0.10) * df.get("behavioural_multiplier", 0.0)
    )
    
    penalties_sum = (
        df.get("consulting_only_penalty", 0.0) +
        df.get("research_only_penalty", 0.0) +
        df.get("title_hopping_penalty", 0.0) +
        df.get("langchain_only_penalty", 0.0) +
        df.get("location_penalty", 0.0)
    )
    
    # Final score formula
    raw_final = (base_score - (weights.get("w_penalty", 0.10) * penalties_sum)).clip(lower=0.0)
    
    df["score"] = raw_final
    
    # Zero out honeypot flagged candidates
    if "honeypot_flag" in df.columns:
        df.loc[df["honeypot_flag"] == True, "score"] = 0.0
        
    # Sort descending by score, and ascending by candidate_id for tie-breaking
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    
    # Take top N
    top_df = df.head(top_n).copy()
    
    # Assign rank
    top_df["rank"] = np.arange(1, len(top_df) + 1)
    
    # Generate reasoning
    top_df["reasoning"] = top_df.apply(generate_reasoning, axis=1)
    
    # Return requested columns only
    return top_df[["candidate_id", "rank", "score", "reasoning"]]

def export_submission_csv(df, filename="submission.csv"):
    """
    Save the ranked DataFrame to CSV with exactly the required columns.
    """
    req_cols = ["candidate_id", "rank", "score", "reasoning"]
    
    # Ensure columns exist before export
    out_cols = [c for c in req_cols if c in df.columns]
    
    df[out_cols].to_csv(filename, index=False)
    print(f"Exported submission to {filename}")
