import pandas as pd
import os
import time
from ranking_engine import rank_candidates, export_submission_csv

def main():
    input_file = "candidates_features.parquet"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please run preprocess_features.py first.")
        return
        
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    print(f"Ranking {len(df)} candidates...")
    
    start_time = time.time()
    top_candidates = rank_candidates(df, top_n=100)
    end_time = time.time()
    
    print(f"Ranking completed in {(end_time - start_time):.4f} seconds.\n")
    
    print("Top 10 Candidates:")
    print("-" * 80)
    # Print formatted top 10
    for idx, row in top_candidates.head(10).iterrows():
        print(f"Rank {row['rank']}: {row['candidate_id']} | Score: {row['score']:.4f}")
        print(f"Reasoning: {row['reasoning']}")
        print("-" * 80)
        
    output_csv = "submission.csv"
    export_submission_csv(top_candidates, output_csv)
    
if __name__ == "__main__":
    main()
