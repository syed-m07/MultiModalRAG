"""
============================================================
Video-RAG Retriever
============================================================
PURPOSE:
    Connects to the local LanceDB vector database and performs
    semantic search using CLIP text embeddings. Given a natural
    language query, returns the top-K most relevant video
    segments with their timestamps and metadata.

USAGE:
    As a module:
        from retriever import retrieve
        results = retrieve("show me the suit up scene", top_k=5)

    Standalone test:
        python retriever.py
============================================================
"""

import os
import torch
from transformers import CLIPProcessor, CLIPModel
import lancedb
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Configuration
# ============================================================

DB_PATH = os.getenv("DB_PATH", "./VideoRAG_DB")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_TOP_K = 5

# ============================================================
# Lazy-loaded globals (initialized once on first call)
# ============================================================

_clip_model = None
_clip_processor = None
_db_table = None


def _load_clip():
    """Load CLIP model and processor (CPU-only for local queries)."""
    global _clip_model, _clip_processor

    if _clip_model is not None:
        return

    print(f"Loading CLIP text encoder: {CLIP_MODEL_NAME}...")
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _clip_model.eval()
    print("CLIP text encoder loaded.")


def _load_db():
    """Connect to the local LanceDB database."""
    global _db_table

    if _db_table is not None:
        return

    print(f"Connecting to LanceDB at: {DB_PATH}...")
    db = lancedb.connect(DB_PATH)
    _db_table = db.open_table("video_chunks")
    print(f"Table 'video_chunks' opened. ({_db_table.count_rows()} rows)")


def embed_query(query_text: str) -> list:
    """
    Convert a natural language query into a CLIP embedding vector.

    Args:
        query_text: The user's search query string.

    Returns:
        A list of floats representing the 512-D normalized embedding.
    """
    _load_clip()

    with torch.no_grad():
        inputs = _clip_processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        text_features = _clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Handle potential output object (Colab compatibility fix)
        if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
            text_features = text_features.pooler_output

        # Normalize to unit vector
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze().tolist()


def retrieve(query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Perform semantic search against the VideoRAG database.

    Args:
        query_text: Natural language query (e.g., "show me the suit up scene").
        top_k: Number of top results to return.

    Returns:
        A list of dicts, each containing:
            - start_time (float): Start timestamp in seconds
            - end_time (float): End timestamp in seconds
            - text (str): The transcript or visual description
            - modality (str): "audio" or "visual"
            - scene_id (int): Scene identifier (-1 if N/A)
            - _distance (float): Similarity distance (lower = more relevant)
    """
    _load_db()

    query_vector = embed_query(query_text)
    results = _db_table.search(query_vector).limit(top_k).to_list()

    # Clean up the results for downstream consumption
    cleaned = []
    for r in results:
        cleaned.append({
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "timestamp": r["timestamp"],
            "text": r["text"],
            "modality": r["modality"],
            "scene_id": r["scene_id"],
            "_distance": r.get("_distance", None),
        })

    return cleaned


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    import json

    test_queries = [
        "show me all suit-up scenes of Iron Man",
        "a person talking",
        "explosion or action scene",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: \"{query}\"")
        print(f"{'='*60}")

        results = retrieve(query, top_k=3)

        for i, r in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Time:     [{r['start_time']:.2f}s - {r['end_time']:.2f}s]")
            print(f"    Modality: {r['modality']}")
            print(f"    Text:     {r['text'][:80]}...")
            print(f"    Distance: {r['_distance']:.4f}")
        print()
