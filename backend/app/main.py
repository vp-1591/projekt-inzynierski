import asyncio
import json
import logging
import os

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .db import database
from .llm_processor import normalize_llm_response

app = FastAPI(title="Disinformation Detector Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/api/chat")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama:11434")
MODEL_NAME = "bielik-lora-mipd:latest"


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._heartbeat_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Start heartbeat if not already running
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for conn in dead_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

    async def _heartbeat_loop(self):
        """Send periodic pings to keep WebSocket connections alive during long deployments."""
        try:
            while self.active_connections:
                await asyncio.sleep(25)
                await self.broadcast({"type": "heartbeat"})
        except asyncio.CancelledError:
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
        "format": "json",
        "keep_alive": 0,
    }
    print("\n--- DEBUG: POŁĄCZENIE Z LLM ---")
    print(f"MODEL: {MODEL_NAME}")
    print(f"PROMPT: {request.text[:100]}...")  # Print first 100 chars

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload, timeout=120.0)
            response.raise_for_status()
            ollama_data = response.json()

            # Physical response from Ollama
            content = ollama_data.get("message", {}).get("content", "")
            print(f"RAW CONTENT FROM OLLAMA: {content}")

            # Normalize and heal the response using the dedicated helper module
            parsed_content = normalize_llm_response(content)

            print(f"PARSED CONTENT: {json.dumps(parsed_content, indent=2)}")
            print("-------------------------------\n")

            return parsed_content
        except Exception as e:
            print(f"ERROR DURING LLM CALL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}") from None


# --- Training Orchestration ---
from .training.orchestrator import MLOpsOrchestrator  # noqa: E402

orchestrator_instance = None


async def get_orchestrator(db: Session = Depends(get_db)):  # noqa: B008
    """Dependency provider for a singleton MLOpsOrchestrator instance."""
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = MLOpsOrchestrator(db)
        # Register a bridge between Orchestrator and WebSocket broadcast
        main_loop = asyncio.get_running_loop()

        def ws_notify_bridge(status):
            try:
                if main_loop.is_running():
                    # Use the captured main_loop to securely schedule the coroutine from any thread
                    asyncio.run_coroutine_threadsafe(manager.broadcast(status), main_loop)
            except RuntimeError as e:
                logging.warning("Failed to schedule WS broadcast: %s", e)

        orchestrator_instance.on_status_change.append(ws_notify_bridge)
    orchestrator_instance.db = db  # Ensure current DB session is used
    return orchestrator_instance


@app.post("/training/upload")
async def upload_training_data(
    file: UploadFile = File(...),  # noqa: B008
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator),  # noqa: B008
):
    """Uploads a training file and triggers the manual training process."""
    import os
    import shutil

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Validate JSONL format
    try:
        with open(file_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                os.remove(file_path)
                raise HTTPException(status_code=400, detail="Empty file")
            json.loads(first_line)
            line_count = 0
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
                    line_count += 1
                if line_count >= 100:
                    break
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Invalid JSONL format: {str(e)}") from None

    if orchestrator.start_manual_training(file_path):
        return {"status": "started", "file": file.filename}
    else:
        raise HTTPException(status_code=400, detail="Training already in progress")


@app.get("/training/status")
async def get_training_status(orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):  # noqa: B008
    """Returns the current status and progress of the MLOps pipeline."""
    return orchestrator.get_status()


@app.websocket("/ws/training/status")
async def websocket_training_status(websocket: WebSocket, orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):  # noqa: B008
    """WebSocket endpoint for real-time status updates."""
    await manager.connect(websocket)
    # Send initial status
    await websocket.send_json(orchestrator.get_status())
    try:
        while True:
            # We don't expect messages from client for now, but keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)


@app.post("/training/promote")
async def promote_model(orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):  # noqa: B008
    """Deploys the latest successfully trained adapter to the production inference path.

    Returns 202 Accepted immediately; deployment proceeds in the background.
    The client receives completion/failure status via the WebSocket.
    """
    if orchestrator.status != "ready_to_promote":
        raise HTTPException(status_code=400, detail="Not ready to promote")

    adapter_path = orchestrator.latest_adapter_path

    async def _run_deployment():
        await orchestrator.deploy_new_adapter(adapter_path)

    asyncio.create_task(_run_deployment())
    return JSONResponse(status_code=202, content={"status": "deployment_started"})


@app.post("/training/progress")
async def report_progress(
    progress_data: dict,
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator),  # noqa: B008
):
    """Updates training or evaluation progress (called by external workers)."""
    orchestrator.update_progress(progress_data["stage"], progress_data["value"])
    return {"status": "ok"}


@app.post("/training/complete")
async def training_complete(
    adapter_path: str,
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator),  # noqa: B008
):
    """Signals that training is done and transitions the pipeline to evaluation."""
    orchestrator.finish_training_and_evaluate(adapter_path)
    return {"status": "evaluation_started"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
