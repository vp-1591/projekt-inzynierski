# System Engineering Architecture (MLOps Loop) - Implementation Guide

This document summarizes the integration of point 2.5 into the project.

## 1. System Components
- **Frontend (React)**: Minimalist UI for analysis + "Expert Mode" for data verification.
- **Backend (FastAPI)**: Central orchestrator managing inference and data collection.
- **Database (SQLite)**: Stores "Golden Samples" (expert-verified data) and training history.
- **Inference (Ollama)**: Local GGUF model runner with LoRA adapters.
- **Training (Unsloth/Python)**: Automated SFT script (optimized for WSL2/Linux).

## 2. Key MLOps Workflows

### A. Human-in-the-Loop Feedback (2.5.1)
1. User enters text in the Frontend.
2. Model analyzes the text via the Backend.
3. In **Expert Mode**, the analyst can "Confirm" (add to training) or "Correct" (manually edit tags).
4. Feedback is saved to `golden_samples` table in SQLite.

### B. Automated Retraining Trigger (2.5.3)
- The Backend monitors the count of new samples.
- Once 50 new samples are reached, it generates `training_data_latest.jsonl`.
- If on a supported GPU (WSL2), it triggers `trainer.py`.
- If not, the file is ready for manual upload to **Google Colab**.

### C. Auto-Benchmark & Hot-Swap (2.5.4)
- After training, `AutoBenchmarker` calculates:
    - **PSR**: Parsing Success Rate
    - **FCR**: Format Correction Rate
    - **F1-Score**: Overall accuracy
- If the new score is better than the baseline, the Backend updates the Ollama `Modelfile` and reloads the model instantly.

## 3. Developer vs. Production Versions

### Developer Version (Current Settings)
- **max_seq_length**: 2048 (balanced for local GPU VRAM)
- **max_steps**: 60 (quick verification of the training pipeline)
- **Hardware**: Optimized for local execution with limited resources.

### Production Environment (Recommended)
- **max_seq_length**: 16384 (full context analysis)
- **max_steps**: Determined by full dataset (e.g., 2 epochs)
- **Hardware**: High-VRAM GPUs (e.g., Tesla T4 16GB, A100) or Google Colab environments.

## 4. Local Hardware Notes
The system handles the "Local Laptop vs GPU" problem by:
1. Using **8-bit Quantized Inference** (Ollama) which is very lightweight.
2. Decoupling **Training Data Preparation** (Local) from **Heavy SFT Training** (WSL or Colab).
3. Allowing "Developer" training runs with reduced parameters to verify the MLOps pipeline logic locally.
