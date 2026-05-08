@echo off
echo Setting up AntyDezinformator...

echo Initializing git submodules...
git submodule update --init --recursive

if not exist "backend\.venv\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv backend\.venv
    call backend\.venv\Scripts\activate.bat && pip install -r backend\requirements.txt
) else (
    echo [OK] Python virtual environment already exists.
)

if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    cd frontend && npm install && cd ..
) else (
    echo [OK] Frontend dependencies already installed.
)

:: Resolve absolute path for Modelfile ADAPTER directive (Ollama requires absolute paths)
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "ADAPTER_PATH=%PROJECT_ROOT%\model\xai-adapter\checkpoint-2475\checkpoint-2475-F32-LoRA.gguf"

if not exist "%ADAPTER_PATH%" (
    echo [WARNING] LoRA adapter file not found at %ADAPTER_PATH%
    echo Ollama model creation will be skipped. Place model files in model\ and re-run setup.
) else (
    :: Update Modelfile with absolute ADAPTER path
    powershell -Command "(Get-Content 'model\Modelfile') -replace '^ADAPTER .*$', ('ADAPTER ' + $env:ADAPTER_PATH.Replace('\','/')) | Set-Content 'model\Modelfile'"

    :: Create Ollama model
    echo Creating Ollama model 'bielik-lora-mipd'...
    ollama create bielik-lora-mipd -f model\Modelfile
    if errorlevel 1 (
        echo [ERROR] Failed to create Ollama model. Make sure Ollama is running and model files are in place.
    ) else (
        echo [OK] Ollama model 'bielik-lora-mipd' created.
    )
)

echo Setup complete. Run run_app.cmd to start the services.
pause