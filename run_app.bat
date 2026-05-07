@echo off
echo Starting AntyDezinformator Services...

:: Check if Ollama is running
ollama list >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ollama is NOT reachable!
    echo Starting Ollama...
    start cmd /k "ollama serve"
    timeout /t 5 >nul
) else (
    echo [OK] Ollama is running.
)

if not exist "backend\.venv\Scripts\activate.bat" (
    echo Setting up virtual environment...
    python -m venv backend\.venv
    call backend\.venv\Scripts\activate.bat && pip install -r backend\requirements.txt
)
start cmd /k "call backend\.venv\Scripts\activate.bat && cd backend && python -m app.main"
start cmd /k "cd frontend && npm run dev"

echo Services are starting in separate windows.
