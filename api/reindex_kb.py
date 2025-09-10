import os
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from config.settings import Settings
from dotenv import load_dotenv

# Load env
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

import argparse

def build_parser():
    p = argparse.ArgumentParser(description="Reindex KB into Pinecone from kb.json")
    p.add_argument("--index", required=True, help="KB index name (e.g., kb-st-384)")
    p.add_argument("--kb", default=str(ROOT_DIR / "kb" / "kb.json"), help="Path to kb.json")
    p.add_argument("--cloud", default="aws")
    p.add_argument("--region", default="us-east-1")
    return p


def ensure_index(pc: Pinecone, name: str, cloud: str, region: str):
    existing = [getattr(x, "name", str(x)) for x in pc.list_indexes()]
    if name in existing:
        print(f"Index '{name}' already exists")
        return
    print(f"Creating index '{name}' (dim=384, cosine, {cloud}/{region})...")
    pc.create_index(name=name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud=cloud, region=region))
    # wait until ready
    while True:
        desc = pc.describe_index(name)
        status = getattr(desc, "status", {})
        ready = getattr(status, "ready", False) if status else False
        if isinstance(status, dict):
            ready = status.get("ready", False)
        if ready:
            break


def main():
    args = build_parser().parse_args()

    settings = Settings()
    api_key = os.environ.get("PINECONE_API_KEY") or settings.PINECONE_API_KEY
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY missing")

    pc = Pinecone(api_key=api_key)
    ensure_index(pc, args.index, args.cloud, args.region)
    index = pc.Index(args.index)

    kb_path = Path(args.kb)
    kb = json.loads(kb_path.read_text())

    model = SentenceTransformer("all-MiniLM-L6-v2")

    vectors = []
    for i, item in tqdm(list(enumerate(kb))):
        q = item["q"].strip()
        a = item["a"].strip()
        topic = item.get("topic")
        tags = item.get("tags", [])
        emb = model.encode(q).tolist()
        vectors.append({
            "id": f"kb-{i:03d}",
            "values": emb,
            "metadata": {"q": q, "a": a, "topic": topic, "tags": tags}
        })
        if len(vectors) >= 100:
            index.upsert(vectors=vectors)
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)

    print("KB upsert complete.")

if __name__ == "__main__":
    main() 