@echo off
setlocal enabledelayedexpansion

:: Navigate to the script's directory
cd /d "%~dp0"

:: Check if .env exists, if so, parse and set environment variables
if exist .env (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        :: Remove any spaces or surrounding quotes if necessary (basic parsing)
        set "%%A=%%B"
    )
)

:: If GEMINI_API_KEY is still not set, ask the user
if "%GEMINI_API_KEY%"=="" (
    echo =======================================================
    echo   GEMINI API KEY REQUIRED
    echo =======================================================
    echo It looks like your Gemini API Key is not set.
    echo.
    echo 1. Go to: https://aistudio.google.com/app/apikey
    echo 2. Sign in with your Google account.
    echo 3. Click "Create API Key".
    echo 4. Copy the key and paste it below.
    echo.
    set /p "NEW_KEY=Paste your GEMINI_API_KEY here: "
    
    :: Save it to the environment and the .env file
    set "GEMINI_API_KEY=!NEW_KEY!"
    echo GEMINI_API_KEY=!NEW_KEY!>> .env
    echo.
    echo ✅ Saved API key to .env file!
    echo =======================================================
)

:: Set provider to gemini explicitly
set GENERATION_PROVIDER=gemini
set PYTHONIOENCODING=utf-8

echo.
echo 🚀 Starting the NCERT Doubt Solver application...
echo ⏳ Please wait for the browser window to open...
echo.

:: Run the python app
python app.py

:: Pause in case of errors
pause
