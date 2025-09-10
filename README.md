# Mini AI Chatbot (Pinecone + Claude)

A simple web-based AI chatbot API that answers professional questions using a small knowledge base stored in Pinecone. If a question does not sufficiently match the KB, the API falls back to Anthropic Claude to generate a concise answer.

## Features
- Knowledge base: 8–10 Q&A pairs in `kb/kb.json`
- Vector search: Pinecone (384-d cosine) with SentenceTransformers `all-MiniLM-L6-v2`
- Fallback LLM: Anthropic Claude (optional)
- REST API: FastAPI endpoints `/ask`, `/history`, `/health`
- Simple persistence: appends Q&A turns to `kb/history.json`

## Setup
1. Create and activate a Python 3.10+ venv
2. Install deps
```bash
pip install -r requirements.txt
```
3. Create `.env` at the repo root with:
```
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_KB=kb-st-384
ANTHROPIC_API_KEY=your_anthropic_key   # optional, for fallback
KB_MATCH_THRESHOLD=0.7
```

## Create/Update the KB index
- Index settings: dimension 384, metric cosine, serverless (e.g., AWS/us-east-1)
- Stored metadata per vector: `{ q, a, topic, tags }`

## Run the API
```bash
uvicorn api.main:app --reload
```

### Test
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Tips for remote work?"}'

curl http://localhost:8000/history?limit=10
```

## Project layout (key files)
- `kb/kb.json` – knowledge base pairs
- `api/reindex_kb.py` – build/refresh KB index in Pinecone
- `bot/kb_chatbot.py` – retrieval + optional Claude fallback
- `api/main.py` – FastAPI app exposing `/ask` and `/history`
- `kb/history.json` – simple JSON persistence for last turns

## Notes
- The API returns:
```json
{
  "answer": "...",
  "source": "kb" | "llm",
  "matchedQuestion": "..." | null,
  "score": 0.82 | null
}
```
- If `ANTHROPIC_API_KEY` is not set, low-confidence queries return a polite default.



