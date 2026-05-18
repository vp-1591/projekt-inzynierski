@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Starting System detekcji manipulacji Services...

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

if not exist "%PROJECT_ROOT%\backend\.venv\Scripts\activate.bat" (
    echo [ERROR] Python virtual environment not found. Run setup.cmd first.
    pause
    exit /b 1
)

netstat -ano | findstr ":8000 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "Backend" /D "%PROJECT_ROOT%\backend" cmd /k ""%PROJECT_ROOT%\backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws websockets"
    echo [STARTING] Backend on :8000.
) else (
    echo [OK] Backend is already running on :8000.
)

if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo [ERROR] Frontend dependencies not found. Run setup.cmd first.
    pause
    exit /b 1
)

netstat -ano | findstr ":5173 " | findstr "LISTENING" >nul 2>&1
if errorlevel 1 (
    start "Frontend" /D "%PROJECT_ROOT%\frontend" cmd /k "npm run dev"
    echo [STARTING] Frontend on :5173.
) else (
    echo [OK] Frontend is already running on :5173.
)

echo All services launched. You can close this window.
pause
