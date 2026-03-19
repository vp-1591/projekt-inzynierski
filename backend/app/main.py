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
            
            # Try to parse content as JSON if it's a string
            parsed_content = None
            try:
                parsed_content = json.loads(content) if isinstance(content, str) else content
            except json.JSONDecodeError:
                print("DEBUG: JSON decode failed, attempting heuristic recovery")
                pass

            # Heuristic Recovery System (consistent with benchmark.py)
            if not parsed_content or not isinstance(parsed_content, dict):
                import re
                
                # 1. Recover Reasoning
                # Matches "reasoning": "..." OR "reasonng": "..." (and other typos)
                reasoning_match = re.search(r'"reason\w*"\s*:\s*"(.*?)"', content, re.DOTALL)
                reasoning = reasoning_match.group(1) if reasoning_match else "Nie udało się wygenerować uzasadnienia."
                
                # 2. Recover Tags
                # Matches list looking like ["TAG1", "TAG2"] anywhere in text
                tags = []
                list_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                if list_match:
                    raw_list = list_match.group(1)
                    # Extract anything that looks like a tag (UPPERCASE_WITH_UNDERSCORES usually)
                    # But cleaning up quotes first
                    candidates = [t.strip().strip('"\'') for t in raw_list.split(',')]
                    tags = [t for t in candidates if t]
                
                parsed_content = {
                    "reasoning": reasoning,
                    "discovered_techniques": tags
                }
                print("DEBUG: Recovered content via heuristics")

            # Validate/Fix Keys if JSON was valid but keys were typoed
            # e.g. "discoverted_techniques" -> "discovered_techniques"
            if isinstance(parsed_content, dict):
                # Normalize reasoning key
                if "reasoning" not in parsed_content:
                    # Look for fuzzy match
                    for k in parsed_content.keys():
                        if k.startswith("reason"):
                            parsed_content["reasoning"] = parsed_content[k]
                            break
                    if "reasoning" not in parsed_content:
                         parsed_content["reasoning"] = "Brak uzasadnienia."

                # Normalize techniques key
                if "discovered_techniques" not in parsed_content:
                    found_techs = []
                    # Look for fuzzy match
                    for k in parsed_content.keys():
                        if "technique" in k or "discovered" in k:
                             found_techs = parsed_content[k]
                             break
                    parsed_content["discovered_techniques"] = found_techs

            # Final Cleanup: Fix tag typos (e.g. EMOTIORAL_COUNTENT -> EMOTIONAL_CONTENT)
            # This is a basic mapping based on Levenshtein distance concept or manual mapping of known hallucinations
            VALID_TAGS = {
                'REFERENCE_ERROR', 'WHATABOUTISM', 'STRAWMAN', 'EMOTIONAL_CONTENT', 
                'CHERRY_PICKING', 'FALSE_CAUSE', 'MISLEADING_CLICKBAIT', 'ANECDOTE', 
                'LEADING_QUESTIONS', 'EXAGGERATION', 'QUOTE_MINING'
            }
            
            cleaned_tags = []
            raw_tags = parsed_content.get("discovered_techniques", [])
            
            if isinstance(raw_tags, list):
                for tag in raw_tags:
                    tag_upper = tag.upper().strip()
                    if tag_upper in VALID_TAGS:
                        cleaned_tags.append(tag_upper)
                    else:
                        # Simple corrections for known issues
                        if "EMOTIO" in tag_upper: cleaned_tags.append("EMOTIONAL_CONTENT")
                        elif "CHERRY" in tag_upper: cleaned_tags.append("CHERRY_PICKING")
                        elif "CLICKBAIT" in tag_upper: cleaned_tags.append("MISLEADING_CLICKBAIT")
                        elif "QUOTE" in tag_upper: cleaned_tags.append("QUOTE_MINING")
                        elif "ANECDOT" in tag_upper: cleaned_tags.append("ANECDOTE")
                        else:
                            # Keep original if we can't map it (frontend will show "Unknown")
                            cleaned_tags.append(tag)
            
            parsed_content["discovered_techniques"] = cleaned_tags

            print(f"PARSED CONTENT: {json.dumps(parsed_content, indent=2)}")
            print("-------------------------------\n")
            
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
    # orchestrator.status = "idle"  <-- Removed to persist success state for UI
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
