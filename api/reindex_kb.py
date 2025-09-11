import os
import json
from pathlib import Path
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from config.settings import Settings
from dotenv import load_dotenv
from openai import OpenAI

# Load env
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

import argparse


def build_parser():
    p = argparse.ArgumentParser(description="Reindex KB into Pinecone from kb.json using OpenAI embeddings")
    p.add_argument("--index", required=True, help="KB index name (e.g., kb-openai-1536)")
    p.add_argument("--kb", default=str(ROOT_DIR / "kb" / "kb.json"), help="Path to kb.json")
    p.add_argument("--embed-model", default=os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"), help="OpenAI embedding model")
    p.add_argument("--cloud", default="aws")
    p.add_argument("--region", default="us-east-1")
    return p


def ensure_index(pc: Pinecone, name: str, dimension: int, cloud: str, region: str):
    existing = [getattr(x, "name", str(x)) for x in pc.list_indexes()]
    if name in existing:
        print(f"Index '{name}' already exists")
        return
    print(f"Creating index '{name}' (dim={dimension}, cosine, {cloud}/{region})...")
    pc.create_index(name=name, dimension=dimension, metric="cosine", spec=ServerlessSpec(cloud=cloud, region=region))
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
    pinecone_api_key = os.environ.get("PINECONE_API_KEY") or settings.PINECONE_API_KEY
    openai_api_key = os.environ.get("OPENAI_API_KEY") or getattr(settings, "OPENAI_API_KEY", "")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY missing")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    pc = Pinecone(api_key=pinecone_api_key)
    openai = OpenAI(api_key=openai_api_key)

    # Determine embedding dimension from a probe
    probe = openai.embeddings.create(model=args.embed_model, input="dimension probe")
    dimension = len(probe.data[0].embedding)

    ensure_index(pc, args.index, dimension, args.cloud, args.region)
    index = pc.Index(args.index)

    kb_path = Path(args.kb)
    kb = json.loads(kb_path.read_text())

    vectors = []
    for i, item in tqdm(list(enumerate(kb))):
        q = item["q"].strip()
        a = item["a"].strip()
        topic = item.get("topic")
        tags = item.get("tags", [])
        emb = openai.embeddings.create(model=args.embed_model, input=q).data[0].embedding
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