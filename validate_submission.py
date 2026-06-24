import pandas as pd
import sys

def validate(csv_path):
    print(f"Validating {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ FAIL: Could not read CSV. Error: {e}")
        sys.exit(1)
        
    expected_columns = ["candidate_id", "rank", "score", "reasoning"]
    
    # 1. Correct CSV Schema
    if list(df.columns) != expected_columns:
        print(f"❌ FAIL: Invalid schema. Expected {expected_columns}, got {list(df.columns)}")
        sys.exit(1)
    else:
        print("✅ Correct CSV schema")
        
    # 2. Exactly 100 rows
    if len(df) != 100:
        print(f"❌ FAIL: Expected exactly 100 rows, got {len(df)}")
        sys.exit(1)
    else:
        print("✅ Exactly 100 rows")
        
    # 3. No null values
    if df.isnull().values.any():
        print("❌ FAIL: Null values detected in the submission")
        sys.exit(1)
    else:
        print("✅ No null values")
        
    # 4. candidate_id unique
    if df["candidate_id"].nunique() != len(df):
        print("❌ FAIL: Duplicate candidate_ids found")
        sys.exit(1)
    else:
        print("✅ candidate_id unique")
        
    # 5. rank 1 to 100 and unique
    ranks = df["rank"].tolist()
    expected_ranks = list(range(1, 101))
    if sorted(ranks) != expected_ranks:
        print("❌ FAIL: Rank column must be exactly 1 to 100")
        sys.exit(1)
    else:
        print("✅ rank 1 to 100 and unique")
        
    # 6. Scores strictly descending (or equal, but monotonically decreasing)
    scores = df["score"].tolist()
    if not all(scores[i] >= scores[i+1] for i in range(len(scores)-1)):
        print("❌ FAIL: Scores are not monotonically decreasing")
        sys.exit(1)
    else:
        print("✅ Scores are monotonically decreasing")
        
    print("\n🎉 All checks passed! The submission is valid.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="submission.csv")
    args = parser.parse_args()
    validate(args.file)
