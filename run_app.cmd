@echo off
echo Starting AntyDezinformator Services...

:: Check if Ollama is running
ollama list >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is NOT reachable!
    echo Starting Ollama...
    start cmd /k "ollama serve"
    timeout /t 5 >nul
) else (
    echo [OK] Ollama is running.
)

if not exist "backend\.venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.cmd first.
    pause
    exit /b 1
)

netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start cmd /k "call backend\.venv\Scripts\activate.bat && cd backend && python -m app.main"
    echo [STARTING] Backend on :8000.
) else (
    echo [OK] Backend is already running on :8000.
)

if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies not found. Run setup.cmd first.
    pause
    exit /b 1
)

netstat -ano | findstr ":5173 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start cmd /k "cd frontend && npm run dev"
    echo [STARTING] Frontend on :5173.
) else (
    echo [OK] Frontend is already running on :5173.
)

echo All services launched. You can close this window.
pause