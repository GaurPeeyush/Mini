import json
import time
import threading
from pathlib import Path

HISTORY_PATH = Path(__file__).resolve().parent / "history.json"
_lock = threading.Lock()

# Ensure file exists
if not HISTORY_PATH.exists():
    HISTORY_PATH.write_text("[]")


def append_entry(question: str, answer: str, source: str) -> None:
    with _lock:
        try:
            data = json.loads(HISTORY_PATH.read_text())
        except Exception:
            data = []
        data.append({
            "ts": int(time.time()),
            "question": question,
            "answer": answer,
            "source": source
        })
        HISTORY_PATH.write_text(json.dumps(data, indent=2))


def load_recent(limit: int = 10):
    try:
        data = json.loads(HISTORY_PATH.read_text())
    except Exception:
        return []
    if limit <= 0:
        return []
    return list(reversed(data[-limit:]))


def clear_history() -> None:
    with _lock:
        HISTORY_PATH.write_text("[]") 