# AntyDezinformator — System detekcji dezinformacji z XAI

System do wykrywania 11 technik manipulacji w tekstach medialnych w języku polskim, oparty na modelu **Bielik-4.5B-Instruct** z adapterami LoRA. Projekt integruje inferencję, automatyczny trening (SFT) oraz ewaluację w jeden cykl MLOps.

## Wymagania

- **OS**: Windows 10/11 z [WSL2](https://learn.microsoft.com/pl-pl/windows/wsl/install) (Ubuntu) — trening wymaga środowiska Linux
- **GPU**: NVIDIA z min. 8 GB VRAM (dla treningu 4-bit)
- **Ollama**: [ollama.com](https://ollama.com)
- **Python 3.11+**, **Node.js 18+**

## Instalacja

### 1. Klonowanie i submoduły

```bash
git clone <repo-url>
cd projekt-inzynierski
git submodule update --init --recursive
```

### 2. Pliki modelu (~7.7 GB)

Katalog `model/` jest gitignore'd — należy go pobrać oddzielnie i umieścić w katalogu projektu.

**Pobierz pliki modelu:** [Google Drive](https://drive.google.com/file/d/1b17N4rKeinj1ahqTz-nhMtxZT8k02OYD/view?usp=sharing)

Powinien zawierać:

```
model/
├── bielik-4.5b-base/
│   ├── Bielik-4.5B-v3.0-Instruct.Q8_0.gguf   (~4.7 GB)
│   ├── model.safetensors                        (~2.5 GB)
│   ├── config.json, tokenizer.json, ...
│   └── ...
├── xai-adapter/checkpoint-2475/
│   ├── checkpoint-2475-F32-LoRA.gguf            (~190 MB)
│   ├── adapter_model.safetensors                 (~190 MB)
│   └── ...
├── dataset/
│   ├── mipd_train_cot_clean.jsonl                (~55 MB)
│   ├── mipd_test.jsonl                           (~8.5 MB)
│   └── mipd_val.jsonl                            (~18 MB)
├── benchmark-reports/   (już w repozytorium)
└── Modelfile            (już w repozytorium)
```

### 3. Automatyczna konfiguracja

Uruchom `setup.cmd` — skrypt:

1. Inicjalizuje submoduły git (`llama.cpp`)
2. Tworzy środowisko wirtualne Python i instaluje zależności
3. Instaluje zależności frontend (npm)
4. Rozwiązuje ścieżkę ADAPTER w `Modelfile` na absolutną
5. Rejestruje model w Ollama (`bielik-lora-mipd`)

```cmd
setup.cmd
```

Wymagane: Ollama musi być uruchomiona przed `setup.cmd`.

### 4. Ręczna konfiguracja (alternatywa)

Jeśli wolisz konfigurować ręcznie:

```bash
# Backend
python -m venv backend/.venv
backend/.venv/Scripts/activate && pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# Ollama — zaktualizuj ścieżkę ADAPTER w model/Modelfile na absolutną, potem:
ollama create bielik-lora-mipd -f model/Modelfile
```

### 5. Środowisko treningowe (WSL2)

Trening działa wyłącznie w WSL2. W terminalu Ubuntu:

```bash
pip install unsloth bitsandbytes accelerate torch trl datasets gguf
nvidia-smi   # sprawdź dostęp do GPU
```

Trening jest uruchamiany przez Panel Ekspercki w UI, nie z CLI.

## Uruchomienie

```cmd
run_app.cmd
```

Uruchamia Ollama, backend (FastAPI :8000) i frontend (Vite :5173).

## Cykl MLOps

1. **Analiza** — wpisz tekst, zobacz wykryte techniki manipulacji
2. **Tryb Ekspercki** — przełącznik w prawym górnym rogu
3. **Trening** — prześlij plik `.jsonl`, system uruchamia SFT w WSL2, postęp w czasie rzeczywistym
4. **Ewaluacja** — automatyczny benchmark na `mipd_test.jsonl`
5. **Hot-swap** — zatwierdź nowy model, Ollama aktualizuje się bez restartu

## Reset do stanu początkowego

```cmd
reset_state.cmd
```

Przywraca adapter `xai-adapter` jako aktywny model, resetuje raport benchmark i usuwa artefakty treningowe.

## Architektura

```
Frontend (React/Vite :5173)
  └── WebSocket + REST ──→ Backend (FastAPI :8000)
                               ├── /analyze → Ollama (:11434) → Bielik LLM
                               ├── /ws/training/status → real-time pipeline updates
                               ├── SQLite (disinfo_system.db)
                               └── /upload, /train, /promote → MLOps pipeline
```

## Metryki

- **PSR** — Parsing Success Rate (% poprawnych odpowiedzi JSON)
- **F1 Score (Strict)** — średnia harmoniczna precyzji i czułości dla technik manipulacji
- **Exact Match** — % dokumentów z idealnie odtworzonym zbiorem technik

---

*Projekt zrealizowany w ramach pracy inżynierskiej.*