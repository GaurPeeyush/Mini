from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import AskRequest, AskResponse, ClearHistoryResponse
from bot.kb_chatbot import KbChatbot
from kb.storage import append_entry, load_recent, clear_history
import uvicorn
import os
import subprocess
from typing import Optional
from config.settings import Settings

app = FastAPI(title="Mini AI Chatbot API")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize KB chatbot
kb_bot = KbChatbot()

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        answer, source, matched_q, score, trace = kb_bot.ask(req.question)
        try:
            append_entry(req.question, answer, source)
        except Exception:
            pass
        return AskResponse(answer=answer, source=source, matchedQuestion=matched_q, score=score, trace=trace)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def history(limit: int = 10):
    try:
        return {"items": load_recent(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/history/clear", response_model=ClearHistoryResponse)
async def history_clear():
    try:
        clear_history()
        return ClearHistoryResponse(success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex")
async def reindex(index: Optional[str] = None, embed_model: Optional[str] = None):
    try:
        settings = Settings()
        idx = index or settings.PINECONE_INDEX_KB
        em = embed_model or settings.OPENAI_EMBED_MODEL
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
        env["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
        cmd = [
            "python", str((os.path.dirname(__file__))) + "/reindex_kb.py",
            "--index", idx,
            "--kb", str((os.path.dirname(__file__)) + "/../kb/kb.json"),
            "--embed-model", em,
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        ok = proc.returncode == 0
        if not ok:
            raise RuntimeError(proc.stderr or proc.stdout)
        return {"success": True, "stdout": proc.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)