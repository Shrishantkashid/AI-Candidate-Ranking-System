@echo off
echo ==============================================
echo   AI Candidate Recommendation System Startup
echo ==============================================

REM Create the virtual environment if it doesn't exist
if not exist myenv\ (
    echo [1/4] Creating virtual environment myenv...
    python -m venv myenv
    
    echo [2/4] Activating virtual environment...
    call myenv\Scripts\activate.bat
    
    echo [3/4] Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [1/4] Virtual environment already exists. Skipping installation.
    echo [2/4] Activating virtual environment...
    call myenv\Scripts\activate.bat
)

REM Generate the parquet file if it doesn't exist
if not exist candidates_features.parquet (
    echo [4/4] Generating sample Parquet features file - 1000 candidates...
    python preprocess_features.py --sample 1000 --input "[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
) else (
    echo [4/4] Parquet features file already exists. Skipping generation.
)

REM Launch the dashboard
echo.
echo ==============================================
echo   Launching Streamlit Dashboard...
echo ==============================================
streamlit run app.py
pause
