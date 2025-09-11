from pinecone import Pinecone
from config.settings import Settings
from openai import OpenAI
from typing import Tuple, Optional, List

settings = Settings()

class KbChatbot:
    def __init__(self):
        if not settings.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is missing")
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing")
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_KB)
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.threshold = settings.KB_MATCH_THRESHOLD
        self.chat_model = settings.OPENAI_CHAT_MODEL
        self.embed_model = settings.OPENAI_EMBED_MODEL

    def _embed(self, text: str) -> List[float]:
        resp = self.openai.embeddings.create(model=self.embed_model, input=text)
        return resp.data[0].embedding

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> str:
        completion = self.openai.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content

    def ask(self, question: str) -> Tuple[str, str, Optional[str], Optional[float], List[str]]:
        # returns (answer, source, matchedQuestion, score, trace)
        trace: List[str] = []
        trace.append("Checking knowledge base...")
        vec = self._embed(question)
        results = self.index.query(vector=vec, top_k=3, include_metadata=True)
        matches = results.get("matches", [])
        trace.append(f"Found {len(matches)} candidate(s) from KB")

        if matches:
            top = matches[0]
            score = float(top.get("score", 0.0))
            meta = top.get("metadata", {}) or {}
            trace.append(f"Top similarity score: {score:.4f}")
            if score >= self.threshold and "a" in meta:
                trace.append(f"Score >= threshold ({self.threshold}); generating polished answer from KB")
                kb_q = meta.get("q", "")
                kb_a = meta.get("a", "")
                system_prompt = (
                    "You are Mini, a friendly, professional assistant. Answer using ONLY the provided knowledge base entry. "
                    "Be concise, clear, and helpful. Do not invent facts."
                )
                user_prompt = (
                    f"Knowledge entry:\nQ: {kb_q}\nA: {kb_a}\n\n"
                    f"User question: {question}\n\n"
                    "Write a human-friendly answer grounded in the knowledge entry."
                )
                polished = self._chat(system_prompt, user_prompt, temperature=0.2, max_tokens=512)
                trace.append("Polished KB answer with OpenAI")
                return polished, "kb", meta.get("q", None), score, trace
            else:
                trace.append(f"Score < threshold ({self.threshold}); will fallback to LLM if available")
        else:
            trace.append("No KB matches; will fallback to LLM if available")

        system_prompt = "You are Mini, a concise professional and friendly assistant"
        user_prompt = f"Answer the question clearly. Question: {question}"
        text = self._chat(system_prompt, user_prompt, temperature=0.3, max_tokens=512)
        trace.append("Used OpenAI fallback to generate answer")
        return text, "llm", None, None, trace 