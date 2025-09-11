# Mini AI Chatbot (Pinecone + OpenAI)

A simple web-based AI chatbot API that answers professional questions using a small knowledge base stored in Pinecone. If a question does not sufficiently match the KB, the API falls back to an OpenAI chat model to generate a concise answer.

## Features
- Knowledge base: Q&A pairs in `kb/kb.json`
- Vector search: Pinecone (cosine) with OpenAI embeddings (`text-embedding-3-small`, 1536-d by default)
- Fallback LLM: OpenAI chat (`gpt-4o-mini` by default)
- REST API: FastAPI endpoints `/ask`, `/history`, `/health`, `/reindex`
- Simple persistence: appends Q&A turns to `kb/history.json`

## Setup
1. Create and activate a Python 3.10+ venv
2. Install deps
```bash
pip install -r requirements.txt
```
3. Create `.env` at the repo root with:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

# Optional overrides (these defaults are used if not set)
PINECONE_INDEX_KB=kb-openai-1536
OPENAI_EMBED_MODEL=text-embedding-3-small   # 1536-d
OPENAI_CHAT_MODEL=gpt-4o-mini
KB_MATCH_THRESHOLD=0.7
```

## Create/Update the KB index
- The reindex script uses OpenAI embeddings and will auto-detect the embedding dimension and create the Pinecone index if it does not exist (serverless).
- Stored metadata per vector: `{ q, a, topic, tags }`

Example:
```bash
export OPENAI_API_KEY=... PINECONE_API_KEY=...
python api/reindex_kb.py \
  --index kb-openai-1536 \
  --kb kb/kb.json \
  --embed-model text-embedding-3-small
```
- If you switch to `text-embedding-3-large` (3072-d), use a new index name and re-run the script.

## Run the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
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
- `api/reindex_kb.py` – build/refresh KB index in Pinecone using OpenAI embeddings
- `bot/kb_chatbot.py` – retrieval + OpenAI chat fallback
- `api/main.py` – FastAPI app exposing `/ask`, `/history`, `/reindex`, `/health`
- `kb/history.json` – simple JSON persistence for last turns

## API response shape
```json
{
  "answer": "...",
  "source": "kb" | "llm",
  "matchedQuestion": "..." | null,
  "score": 0.82 | null,
  "trace": ["..."]
}
```

## Notes
- Set `PINECONE_INDEX_KB` to the index you reindexed.
- The `/reindex` endpoint runs the reindex script as a subprocess; for serverless deployments, prefer running reindex locally.
- No local Torch/SentenceTransformers models are used; all embeddings and LLM are via OpenAI APIs.



