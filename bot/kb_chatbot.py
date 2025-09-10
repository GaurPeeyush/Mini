from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config.settings import Settings
from anthropic import Anthropic
from typing import Tuple, Optional, List

settings = Settings()

class KbChatbot:
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_KB)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = settings.KB_MATCH_THRESHOLD
        self.anthropic = Anthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None

    def ask(self, question: str) -> Tuple[str, str, Optional[str], Optional[float], List[str]]:
        # returns (answer, source, matchedQuestion, score, trace)
        trace: List[str] = []
        trace.append("Checking knowledge base...")
        vec = self.model.encode(question).tolist()
        results = self.index.query(vector=vec, top_k=3, include_metadata=True)
        matches = results.get("matches", [])
        trace.append(f"Found {len(matches)} candidate(s) from KB")

        if matches:
            top = matches[0]
            score = float(top.get("score", 0.0))
            meta = top.get("metadata", {})
            trace.append(f"Top similarity score: {score:.4f}")
            if score >= self.threshold and "a" in meta:
                trace.append(f"Score >= threshold ({self.threshold}); generating polished answer from KB")
                if self.anthropic:
                    kb_q = meta.get("q", "")
                    kb_a = meta.get("a", "")
                    prompt = (
                        "Your name is Mini, a friendly, professional assistant. Rewrite an answer for the user using ONLY the provided knowledge base entry. "
                        "Be concise, clear, and helpful. Do not invent facts.\n\n"
                        f"Knowledge entry (from the KB):\nQ: {kb_q}\nA: {kb_a}\n\n"
                        f"User question: {question}\n\n"
                        "Write a human-friendly answer grounded in the KB entry above."
                    )
                    completion = self.anthropic.messages.create(
                        model="claude-3-5-sonnet-latest",
                        max_tokens=512,
                        temperature=0.4,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    polished = "".join([b.text for b in completion.content if getattr(b, "type", None) == "text"]) or str(completion.content)
                    trace.append("Polished KB answer with Claude")
                    return polished, "kb", meta.get("q", None), score, trace
                else:
                    trace.append("Anthropic not configured; returning raw KB answer")
                    return meta.get("a", ""), "kb", meta.get("q", None), score, trace
            else:
                trace.append(f"Score < threshold ({self.threshold}); will fallback to LLM if available")
        else:
            trace.append("No KB matches; will fallback to LLM if available")

        # Fallback to LLM if configured or return a polite default
        if self.anthropic:
            prompt = f"You are a concise professional assistant. Answer the question clearly. Question: {question}"
            completion = self.anthropic.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=512,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            text = "".join([b.text for b in completion.content if getattr(b, "type", None) == "text"]) or str(completion.content)
            trace.append("Used Claude fallback to generate answer")
            return text, "llm", None, None, trace

        trace.append("LLM not configured; returning default message")
        return "I'm not sure about that yet, but I can look it up!", "llm", None, None, trace 