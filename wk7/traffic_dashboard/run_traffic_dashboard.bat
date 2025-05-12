
@echo off
REM Activate conda environment if available
call conda activate traffic-env || echo Skipping conda environment activation

REM Navigate to the project directory
cd /d "%~dp0"

REM Install dependencies
echo Installing required Python packages...
pip install flask torch scikit-learn pandas requests joblib

REM Run historical weather fetcher
echo Running historical weather script...
python get_historical_weather_okaloosa.py

REM Run predictor to train models
echo Running predictor and training models...
python predictor_with_estimated_aadt.py

REM Start Flask server
echo Starting Flask dashboard...
python app.py
