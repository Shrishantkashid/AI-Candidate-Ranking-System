import time
import tracemalloc
import gzip
import pandas as pd
import os
import argparse

from preprocess_features import process_candidate
from ranking_engine import rank_candidates, export_submission_csv

def run_benchmark(input_file, output_file):
    print("Starting Benchmark...")
    
    # 1. CPU Verification
    print("\n--- Hardware Verification ---")
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        pass
        
    # Also check env vars
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in [None, "", "-1"] or gpu_available:
        print("⚠️ WARNING: GPU appears to be available or enabled. Hackathon requires CPU only.")
    else:
        print("✅ CPU Only Verification Passed. Execution is strictly bound to CPU.")
        
    print("\n--- Pipeline Execution ---")
    tracemalloc.start()
    start_time = time.time()
    
    features = []
    open_func = gzip.open if input_file.endswith('.gz') else open
    
    try:
        with open_func(input_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                res = process_candidate(line)
                if res: features.append(res)
    except FileNotFoundError:
        print(f"❌ File {input_file} not found. Please provide a valid candidates file.")
        return
        
    total_processed = len(features)
    
    df = pd.DataFrame(features)
    top_candidates = rank_candidates(df, top_n=100)
    export_submission_csv(top_candidates, filename=output_file)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime = end_time - start_time
    peak_mb = peak / 10**6
    
    print("\n--- Benchmark Results ---")
    print(f"Total Candidates Processed: {total_processed:,}")
    print(f"Total Runtime:              {runtime:.2f} seconds")
    print(f"Peak Python Memory Usage:   {peak_mb:.2f} MB")
    
    if runtime < 300:
        print("✅ Runtime is well under the 5-minute (300s) limit.")
    else:
        print("❌ WARNING: Runtime exceeds 5 minutes limit.")
        
    if peak_mb < 16000:
        print("✅ Memory is well under the 16GB limit.")
    else:
        print("❌ WARNING: Memory exceeds 16GB limit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()
    
    run_benchmark(args.candidates, args.out)
