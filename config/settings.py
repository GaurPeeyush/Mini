import os
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# Load .env from both repo root and package directory to be robust under reloaders
ROOT_DIR = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
load_dotenv(PKG_DIR / ".env")

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_INDEX_KB: str = os.environ.get("PINECONE_INDEX_KB", "kb-openai-1536")

    KB_MATCH_THRESHOLD: float = float(os.environ.get("KB_MATCH_THRESHOLD", 0.7))
    MAX_CONTEXT_LENGTH: int = 5

    class Config:
        env_file = ".env"