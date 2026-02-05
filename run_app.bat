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

start cmd /k "cd backend && python -m app.main"
start cmd /k "cd frontend && npm run dev"

echo Services are starting in separate windows.
