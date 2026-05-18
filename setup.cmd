@echo off
setlocal enabledelayedexpansion

:: Resolve project root (directory of this script)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo Setting up System detekcji manipulacji...
echo.

:: ── Prerequisite checks ──────────────────────────────────────────────
set "MISSING=0"

where python >nul 2>&1 && (
    echo [OK] python found
) || (
    echo [MISSING] Python is not on PATH. Install Python 3.11+ from https://www.python.org/downloads/ and add it to PATH.
    set "MISSING=1"
)

where npm >nul 2>&1 && (
    echo [OK] npm found
) || (
    echo [MISSING] npm is not on PATH. Install Node.js LTS from https://nodejs.org/ and add it to PATH.
    set "MISSING=1"
)

where ollama >nul 2>&1 && (
    echo [OK] ollama found
) || (
    echo [MISSING] Ollama is not on PATH. Install from https://ollama.com and add it to PATH.
    set "MISSING=1"
)

if "%MISSING%"=="1" (
    echo.
    echo [ABORT] Install the missing prerequisites above and re-run setup.
    goto :done
)

:: ── Git submodules ────────────────────────────────────────────────────
echo Initializing git submodules...
git submodule update --init --recursive
if errorlevel 1 (
    echo [ERROR] git submodule update failed.
    goto :done
)

:: ── Python virtual environment ────────────────────────────────────────
if not exist "%PROJECT_ROOT%\backend\.venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv "%PROJECT_ROOT%\backend\.venv"
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to create Python virtual environment.
        echo.
        echo Troubleshooting:
        echo   1. Re-run the Python installer from https://www.python.org/downloads/
        echo   2. Check "Add Python to PATH" and "Install pip" options.
        echo   3. If using a corporate/managed machine, the venv module may be
        echo      blocked by policy. Try: python -m venv test_env
        echo.
        goto :done
    )
)

echo Installing/updating backend dependencies...
call "%PROJECT_ROOT%\backend\.venv\Scripts\activate.bat" && python -m pip install -r "%PROJECT_ROOT%\backend\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies.
    goto :done
)

:: ── Frontend dependencies ─────────────────────────────────────────────
if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo Installing frontend dependencies...
    pushd "%PROJECT_ROOT%\frontend"
    npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        popd
        goto :done
    )
    popd
) else (
    echo [OK] Frontend dependencies already installed.
)

:: ── WSL2 training dependencies ──────────────────────────────────────────
where wsl >nul 2>&1 || goto :skip_wsl

echo Checking WSL2 training environment...
wsl echo ok >nul 2>&1 || goto :no_wsl_distro

echo [OK] WSL2 distro found. Checking prerequisites...

:: Check python3 exists in WSL
wsl bash -lc "python3 --version" >nul 2>&1 || goto :no_wsl_python

:: Check python3-venv module is available
wsl bash -lc "python3 -c 'import venv'" >nul 2>&1 || goto :no_wsl_venv

echo [OK] WSL2 Python prerequisites met. Setting up training environment...
wsl bash -lc "cd $(wslpath -u '%PROJECT_ROOT%/backend') && if ! test -x .venv-wsl/bin/python || ! .venv-wsl/bin/python -m pip --version >/dev/null 2>&1; then rm -rf .venv-wsl && python3 -m venv .venv-wsl; fi"
if errorlevel 1 (
    echo [WARNING] WSL training environment setup failed. Training will not work.
    echo   Try re-running setup.cmd or install manually:
    echo     wsl bash -lc "sudo apt install python3-venv python3-pip"
) else (
    echo Checking installed WSL training dependencies...
    wsl bash -lc "cd $(wslpath -u '%PROJECT_ROOT%/backend') && .venv-wsl/bin/python check_wsl_dependencies.py requirements-wsl.txt"
    if errorlevel 1 (
        echo Installing/updating missing WSL training dependencies...
        wsl bash -lc "cd $(wslpath -u '%PROJECT_ROOT%/backend') && .venv-wsl/bin/python -m pip install -r requirements-wsl.txt"
        if errorlevel 1 (
            echo [WARNING] WSL dependency install failed. Training will not work.
            echo   Try re-running setup.cmd or install manually:
            echo     wsl bash -lc "sudo apt install python3-venv python3-pip"
        ) else (
            echo [OK] WSL training dependencies installed.
        )
    )
)
goto :after_wsl

:skip_wsl
echo [SKIP] WSL not available. Training is unavailable on Windows.
goto :after_wsl

:no_wsl_distro
echo [SKIP] No WSL2 distro installed. Training is unavailable on Windows.
echo   Install a WSL2 distro with: wsl --install -d Ubuntu
goto :after_wsl

:no_wsl_python
echo [SKIP] Python3 not found in WSL. Training is unavailable.
echo   Install Python inside WSL:
echo     wsl bash -lc "sudo apt update && sudo apt install -y python3 python3-pip"
goto :after_wsl

:no_wsl_venv
echo [SKIP] python3-venv not available in WSL. Training is unavailable.
echo   Install the venv module inside WSL:
echo     wsl bash -lc "sudo apt install -y python3-venv"
goto :after_wsl

:after_wsl

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
    ollama list 2>nul | findstr /i "bielik-lora-mipd" >nul 2>&1 && (
        echo [OK] Ollama model 'bielik-lora-mipd' already exists.
    ) || (
        :: Update Modelfile with absolute ADAPTER path (Ollama requires absolute paths)
        echo Configuring Ollama model...
        powershell -Command "(Get-Content '%PROJECT_ROOT%\model\Modelfile') -replace '^ADAPTER .*$', ('ADAPTER ' + '%ADAPTER_PATH%'.Replace('\','/')) | Set-Content '%PROJECT_ROOT%\model\Modelfile'"

        echo Creating Ollama model 'bielik-lora-mipd'...
        ollama create bielik-lora-mipd -f "%PROJECT_ROOT%\model\Modelfile"
        if errorlevel 1 (
            echo [ERROR] Failed to create Ollama model. Make sure Ollama is running.
            goto :done
        ) else (
            echo [OK] Ollama model 'bielik-lora-mipd' created.
        )
    )
)

echo.
echo Setup complete. Run run_app.cmd to start the services.
goto :done

:done
pause
