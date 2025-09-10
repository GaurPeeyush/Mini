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
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
    PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
    PINECONE_INDEX_KB: str = os.environ.get("PINECONE_INDEX_KB", "kb-st-384")
    KB_MATCH_THRESHOLD: float = float(os.environ.get("KB_MATCH_THRESHOLD", 0.7))
    MAX_CONTEXT_LENGTH: int = 5

    class Config:
        env_file = ".env"