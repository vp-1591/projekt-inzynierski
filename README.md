# AntyDezinformator (Antigravity MLOps Edition)

System do wykrywania technik manipulacji w tekstach w języku polskim, oparty na modelu **Bielik-4.5B-Instruct**. Projekt integruje inferencję, automatyczny trening (SFT) oraz ewaluację w jeden pełny cykl MLOps.

## 🚀 Architektura Systemu

- **Frontend**: React (Vite) - Nowoczesny interfejs z "Panelem Eksperckim" do zarządzania cyklem życia modelu.
- **Backend Orchestrator**: FastAPI - Serce systemu zarządzające inferencją, bazą danych (SQLite) i procesami MLOps.
- **Inference**: Ollama - Lokalny serwer LLM obsługujący model Bielik z adapterami LoRA.
- **Training (WSL2)**: Unsloth + Hugging Face - Optymalizowany pod kątem VRAM potok treningowy działający w środowisku Linux (WSL2).

## 🛠️ Instalacja i Konfiguracja

### 1. Wymagania Sprzętowe (Wersja Deweloperska)
- **GPU**: NVIDIA (min. 8GB VRAM dla treningu 4-bit).
- **OS**: Windows 10/11 z zainstalowanym **WSL2** (Ubuntu).

### 2. Przygotowanie Ollama
1. Zainstaluj [Ollama](https://ollama.ai/).
2. Pobierz bazowy model Bielik (lub zaimportuj z Modelfile):
   ```bash
   ollama create bielik-4.5b -f ./model/Modelfile
   ```

### 3. Konfiguracja Backend (Windows)
1. Przejdź do folderu `backend`.
2. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt
   ```
3. Uruchom serwer:
   ```bash
   python -m app.main
   ```

### 4. Konfiguracja Training Environment (WSL2)
1. Otwórz terminal WSL2 (Ubuntu).
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install unsloth bitsandbytes accelerate torch trl datasets
   ```
3. Upewnij się, że masz dostęp do GPU (`nvidia-smi` wewnątrz WSL).

### 5. Konfiguracja Frontend (Windows)
1. Przejdź do folderu `frontend`.
2. Zainstaluj zależności:
   ```bash
   npm install
   ```
3. Uruchom aplikację:
   ```bash
   npm run dev
   ```

## 🧠 Cykl MLOps (Human-in-the-Loop)

1. **Analiza**: Wprowadź tekst w głównym oknie, aby zobaczyć wykryte techniki przez aktualny model.
2. **Tryb Ekspercki**: Aktywuj przełącznik w prawym górnym rogu.
3. **Trening**: 
   - Prześlij plik `.jsonl` z nowymi przykładami (format Alpaca/ChatML).
   - System automatycznie uruchomi proces `trainer.py` wewnątrz WSL2.
   - Postęp treningu jest raportowany w czasie rzeczywistym na pasku bocznym.
4. **Ewaluacja**: Po treningu system automatycznie uruchamia benchmark na zbiorze testowym (`model/datasets/mipd_test.jsonl`).
5. **Wdrożenie (Hot-Swap)**: Jeśli nowy wynik F1 jest satysfakcjonujący, kliknij "Potwierdź Zmianę Modelu". System zaktualizuje Ollama bez restartu usług.

## 📊 Metryki
System mierzy:
- **PSR (Parsing Success Rate)**: Czy model generuje poprawny JSON?
- **F1 Score**: Skuteczność klasyfikacji technik manipulacji względem zbioru złotego.

---
*Projekt zrealizowany w ramach pracy inżynierskiej.*
