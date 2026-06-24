import argparse
import gzip
import pandas as pd
import time
from preprocess_features import process_candidate
from ranking_engine import rank_candidates, export_submission_csv

def main():
    parser = argparse.ArgumentParser(description="End-to-End Candidate Ranking Pipeline")
    parser.add_argument("--candidates", required=True, help="Path to input candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()
    
    start_time = time.time()
    input_file = args.candidates
    output_file = args.out
    
    features = []
    open_func = gzip.open if input_file.endswith('.gz') else open
    
    print(f"Reading and extracting features from {input_file}...")
    with open_func(input_file, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            res = process_candidate(line)
            if res:
                features.append(res)
                
            if (i + 1) % 20000 == 0:
                print(f"  ...processed {i + 1} candidates")
                
    extract_time = time.time()
    print(f"Feature extraction completed in {extract_time - start_time:.2f} seconds.")
    
    print("Running ranking engine...")
    df = pd.DataFrame(features)
    
    # rank_candidates expects features_df
    top_candidates = rank_candidates(df, top_n=100)
    
    print(f"Exporting top 100 candidates to {output_file}...")
    export_submission_csv(top_candidates, filename=output_file)
    
    end_time = time.time()
    print(f"Pipeline finished successfully in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
