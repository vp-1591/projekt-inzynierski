from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import httpx
from .db import database
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
    
    import json
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
            
            # Print physical response from Ollama
            content = ollama_data.get('message', {}).get('content', '')
            print(f"RAW CONTENT FROM OLLAMA: {content}")
            
            # Try to parse content as JSON if it's a string (fastapi will do it anyway, but we want to log it)
            parsed_content = json.loads(content) if isinstance(content, str) else content
            print(f"PARSED CONTENT: {json.dumps(parsed_content, indent=2)}")
            print("-------------------------------\n")
            
            # Note: the frontend expects discovered_techniques field.
            # If the model returns it inside content, we should return that.
            return parsed_content
        except Exception as e:
            print(f"ERROR DURING LLM CALL: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")


# Global orchestrator instance (singleton-ish for this simpliciy level)
from .training.orchestrator import MLOpsOrchestrator
# Initialize later to avoid circular imports during startup if needed, 
# or use a dependency injection pattern.
# For now, we will instantiate it per request but share state via singleton pattern or database
# BUT Orchestrator stores state in memory (self.training_progress).
# So we need a global instance.
orchestrator_instance = None

def get_orchestrator(db: Session = Depends(get_db)):
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = MLOpsOrchestrator(db)
    # Update db session reference
    orchestrator_instance.db = db
    return orchestrator_instance

@app.post("/training/upload")
async def upload_training_data(
    file: UploadFile = File(...), 
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)
):
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
    return orchestrator.get_status()

@app.post("/training/promote")
async def promote_model(orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)):
    if orchestrator.status != "ready_to_promote":
        raise HTTPException(status_code=400, detail="Not ready to promote")
    
    await orchestrator.deploy_new_adapter(orchestrator.latest_adapter_path)
    orchestrator.status = "idle"
    return {"status": "promoted"}

@app.post("/training/progress")
async def report_progress(
    progress_data: dict, 
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)
):
    # { "stage": "training"|"evaluation", "value": 50 }
    orchestrator.update_progress(progress_data['stage'], progress_data['value'])
    return {"status": "ok"}

@app.post("/training/complete")
async def training_complete(
    adapter_path: str, 
    orchestrator: MLOpsOrchestrator = Depends(get_orchestrator)
):
    orchestrator.finish_training_and_evaluate(adapter_path)
    return {"status": "evaluation_started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
