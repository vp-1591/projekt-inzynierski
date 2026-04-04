from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import httpx
import json
from .db import database
from .llm_processor import normalize_llm_response
from pydantic import BaseModel
from typing import Any

app = FastAPI(title="Disinformation Detector Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "bielik-lora-mipd:latest"


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        import asyncio
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()


class AnalysisRequest(BaseModel):
    text: str


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/analyze")
async def analyze_text(request: AnalysisRequest):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": request.text}],
        "stream": False,
        "format": "json"
    }
    print("\n--- DEBUG: POŁĄCZENIE Z LLM ---")
    print(f"MODEL: {MODEL_NAME}")
    print(f"PROMPT: {request.text[:100]}...") # Print first 100 chars
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            ollama_data = response.json()
            
            # Physical response from Ollama
            content = ollama_data.get('message', {}).get('content', '')
            print(f"RAW CONTENT FROM OLLAMA: {content}")
            
            # Normalize and heal the response using the dedicated helper module
            parsed_content = normalize_llm_response(content)

            print(f"PARSED CONTENT: {json.dumps(parsed_content, indent=2)}")
            print("-------------------------------\n")
            
            return parsed_content
        except Exception as e:
            print(f"ERROR DURING LLM CALL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")


# --- Training Orchestration ---
from .training.orchestrator import MLOpsOrchestrator

orchestrator_instance = None

async def get_orchestrator(db: Session = Depends(get_db)):
    """Dependency provider for a singleton MLOpsOrchestrator instance."""
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = MLOpsOrchestrator(db)
        # Register a bridge between Orchestrator and WebSocket broadcast
        import asyncio
        main_loop = asyncio.get_running_loop()
        def ws_notify_bridge(status):
            if main_loop.is_running():
                # Use the captured main_loop to securely schedule the coroutine from any thread
                asyncio.run_coroutine_threadsafe(manager.broadcast(status), main_loop)
        orchestrator_instance.on_status_change.append(ws_notify_bridge)
    orchestrator_instance.db = db # Ensure current DB session is used
    return orchestrator_instance

@app.post("/training/upload")
async def upload_training_data(
    file: UploadFile = File(...), 
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)
):
    """Uploads a training file and triggers the manual training process."""
    import shutil
    import os
    
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    if orchestrator.start_manual_training(file_path):
        return {"status": "started", "file": file.filename}
    else:
        raise HTTPException(status_code=400, detail="Training already in progress")

@app.get("/training/status")
async def get_training_status(orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):
    """Returns the current status and progress of the MLOps pipeline."""
    return orchestrator.get_status()

@app.websocket("/ws/training/status")
async def websocket_training_status(websocket: WebSocket, orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):
    """WebSocket endpoint for real-time status updates."""
    await manager.connect(websocket)
    # Send initial status
    await websocket.send_json(orchestrator.get_status())
    try:
        while True:
            # We don't expect messages from client for now, but keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/training/promote")
async def promote_model(orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):
    """Deploys the latest successfully trained adapter to the production inference path."""
    if orchestrator.status != "ready_to_promote":
        raise HTTPException(status_code=400, detail="Not ready to promote")
    
    await orchestrator.deploy_new_adapter(orchestrator.latest_adapter_path)
    return {"status": "promoted"}

@app.post("/training/progress")
async def report_progress(
    progress_data: dict, 
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)
):
    """Updates training or evaluation progress (called by external workers)."""
    orchestrator.update_progress(progress_data['stage'], progress_data['value'])
    return {"status": "ok"}

@app.post("/training/complete")
async def training_complete(
    adapter_path: str, 
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)
):
    """Signals that training is done and transitions the pipeline to evaluation."""
    orchestrator.finish_training_and_evaluate(adapter_path)
    return {"status": "evaluation_started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
