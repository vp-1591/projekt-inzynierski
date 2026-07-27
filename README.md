# System detekcji manipulacji — wykrywanie dezinformacji z XAI

System do wykrywania 11 technik manipulacji w tekstach medialnych w języku polskim, oparty na modelu **Bielik-4.5B-Instruct** z adapterami LoRA. Projekt integruje inferencję, automatyczny trening (SFT) oraz ewaluację w jeden cykl MLOps.

## Wymagania

- **Docker Desktop** z WSL2 backend (Windows) lub Docker Engine (Linux)
- **GPU**: NVIDIA z min. 8 GB VRAM (dla treningu 4-bit)
- **NVIDIA Container Toolkit** — [instrukcja instalacji](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Do uruchomienia bez Dockera: Python 3.11+, Node.js 18+, Ollama

## Instalacja

### 1. Klonowanie i submoduły

```bash
git clone <repo-url>
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
├── Modelfile            (już w repozytorium)
└── Modelfile.docker     (już w repozytorium)
```

### 3. Uruchomienie przez Docker (zalecane)

```bash
docker compose up
```

To uruchamia trzy serwisy:
- **Ollama** na porcie :11435 — automatycznie tworzy model `bielik-lora-mipd` przy pierwszym starcie
- **Backend API** na porcie :8000 (dokumentacja Swagger: `/docs`)
- **Frontend** na porcie :5173

Szczegóły w [DOCKER.md](DOCKER.md).

### 4. Ręczna konfiguracja (alternatywa, bez Dockera)

Jeśli wolisz uruchomić bez Dockera:

```bash
# Backend
python -m venv backend/.venv
backend/.venv/Scripts/activate && pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install && cd ..

# WSL2 — środowisko treningowe (opcjonalnie, jeśli WSL dostępne)
wsl bash -lc "cd $(wslpath -u 'C:\...\projekt-inzynierski\backend') && \
  python3 -m venv .venv-wsl && .venv-wsl/bin/pip install -r requirements-wsl.txt"

# Ollama — zaktualizuj ścieżkę ADAPTER w model/Modelfile na absolutną, potem:
ollama create bielik-lora-mipd -f model/Modelfile
```

### 5. Środowisko treningowe (WSL2)

W Docker trening działa wewnątrz kontenera backend z GPU. Poza Dockerem wymaga WSL2:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
cd $(wslpath -u 'C:\...\projekt-inzynierski\backend')
python3 -m venv .venv-wsl
.venv-wsl/bin/pip install -r requirements-wsl.txt
nvidia-smi   # sprawdź dostęp do GPU
```

Trening jest uruchamiany przez Panel Ekspercki w UI, nie z CLI.

### Rozwiązywanie problemów

**Błąd `.venv-wsl/bin/pip: No such file or directory`** — środowisko wirtualne WSL jest uszkodzone. Usuń je i utwórz ponownie:

```bash
wsl bash -lc "cd /mnt/d/Documents/Vadym/GitRep/projekt-inzynierski/backend && rm -rf .venv-wsl"
```

Następnie utwórz venv ręcznie (patrz sekcja wyżej).

## Uruchomienie

```bash
docker compose up
```

Uruchamia Ollama (:11435), backend (FastAPI :8000) i frontend (Vite :5173) w kontenerach.

## Cykl MLOps

1. **Analiza** — wpisz tekst, zobacz wykryte techniki manipulacji
2. **Tryb Ekspercki** — przełącznik w prawym górnym rogu
3. **Trening** — prześlij plik `.jsonl`, system uruchamia SFT, postęp w czasie rzeczywistym
4. **Ewaluacja** — automatyczny benchmark na `mipd_test.jsonl`
5. **Hot-swap** — zatwierdź nowy model, Ollama aktualizuje się bez restartu

## Reset do stanu początkowego

Aby przywrócić adapter `xai-adapter` jako aktywny model i usunąć artefakty treningowe:

```bash
# Zatrzymaj kontenery
docker compose down

# Usuń wolumeny z logami i uploadami
docker compose down -v

# Zrestartuj — model zostanie odtworzony z xai-adapter
docker compose up
```

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

_Projekt zrealizowany w ramach pracy inżynierskiej._
