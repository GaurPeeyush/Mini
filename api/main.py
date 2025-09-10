from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import AskRequest, AskResponse, ClearHistoryResponse
from bot.kb_chatbot import KbChatbot
from kb.storage import append_entry, load_recent, clear_history
import uvicorn

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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)