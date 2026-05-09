@echo off
setlocal enabledelayedexpansion

:: Resolve project root (directory of this script)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Setting up AntyDezinformator...
echo.

:: ── Prerequisite checks ──────────────────────────────────────────────
set "MISSING=0"

where python >nul 2>&1
if errorlevel 1 (
    echo [MISSING] Python is not on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and add it to PATH.
    set "MISSING=1"
)

where npm >nul 2>&1
if errorlevel 1 (
    echo [MISSING] npm is not on PATH. Install Node.js LTS from https://nodejs.org/ and add it to PATH.
    set "MISSING=1"
)

where ollama >nul 2>&1
if errorlevel 1 (
    echo [MISSING] Ollama is not on PATH. Install from https://ollama.com and add it to PATH.
    set "MISSING=1"
)

if "%MISSING%"=="1" (
    echo.
    echo [ABORT] Install the missing prerequisites above and re-run setup.
    pause
    exit /b 1
)

:: ── Git submodules ────────────────────────────────────────────────────
echo Initializing git submodules...
git submodule update --init --recursive
if errorlevel 1 (
    echo [ERROR] git submodule update failed.
    pause
    exit /b 1
)

:: ── Python virtual environment ────────────────────────────────────────
if not exist "%PROJECT_ROOT%\backend\.venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv "%PROJECT_ROOT%\backend\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create Python virtual environment.
        pause
        exit /b 1
    )
    call "%PROJECT_ROOT%\backend\.venv\Scripts\activate.bat" && pip install -r "%PROJECT_ROOT%\backend\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install Python dependencies.
        pause
        exit /b 1
    )
) else (
    echo [OK] Python virtual environment already exists.
)

:: ── Frontend dependencies ─────────────────────────────────────────────
if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd "%PROJECT_ROOT%\frontend"
    npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd
        pause
        exit /b 1
    )
    popd
) else (
    echo [OK] Frontend dependencies already installed.
)

:: ── Ollama model ──────────────────────────────────────────────────────
set "ADAPTER_PATH=%PROJECT_ROOT%\model\xai-adapter\checkpoint-2475\checkpoint-2475-F32-LoRA.gguf"

if not exist "%ADAPTER_PATH%" (
    echo.
    echo [WARNING] LoRA adapter not found at:
    echo   %ADAPTER_PATH%
    echo Ollama model creation will be skipped.
    echo Place model files in model\ and re-run setup.
    echo.
) else (
    :: Update Modelfile with absolute ADAPTER path (Ollama requires absolute paths)
    echo Configuring Ollama model...
    powershell -Command "(Get-Content '%PROJECT_ROOT%\model\Modelfile') -replace '^ADAPTER .*$', ('ADAPTER ' + '%ADAPTER_PATH%'.Replace('\','/')) | Set-Content '%PROJECT_ROOT%\model\Modelfile'"

    echo Creating Ollama model 'bielik-lora-mipd'...
    ollama create bielik-lora-mipd -f "%PROJECT_ROOT%\model\Modelfile"
    if errorlevel 1 (
        echo [ERROR] Failed to create Ollama model. Make sure Ollama is running.
        pause
        exit /b 1
    ) else (
        echo [OK] Ollama model 'bielik-lora-mipd' created.
    )
)

echo.
echo Setup complete. Run run_app.cmd to start the services.
pause