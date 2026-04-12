"""
============================================================
Video-RAG Retriever (v3 - Temporal Clustering + RRF)
============================================================
PURPOSE:
    Connects to the local LanceDB vector database and performs
    HYBRID semantic search using CLIP text embeddings. Searches
    visual and audio modalities SEPARATELY, applies temporal
    clustering to visual results (grouping nearby frames into
    scene-level events), then merges everything using Reciprocal
    Rank Fusion (RRF).

    This solves the single-frame accuracy problem: instead of
    returning one random frame, we find dense CLUSTERS of
    matching frames that indicate a sustained visual event.

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
from collections import defaultdict
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

# RRF smoothing constant (standard value from literature)
RRF_K = 60

# Modality weights for RRF fusion
VISUAL_WEIGHT = 1.5   # Most important: strict visual confirmation
CAPTION_WEIGHT = 1.3  # BLIP-2 Scene captions (second most important)
AUDIO_WEIGHT = 1.0    # Audio transcripts (lowest weight)

# How many candidates to fetch per modality before fusion
CANDIDATES_PER_MODALITY = 50

# Temporal clustering: frames within this many seconds of each
# other are grouped into the same "event cluster"
CLUSTER_GAP_SECONDS = 15.0

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


def _search_by_modality(query_vector: list, modality: str, limit: int) -> list[dict]:
    """
    Search the database filtered to a specific modality.

    Args:
        query_vector: The 512-D CLIP embedding of the query.
        modality: Either "visual" or "audio".
        limit: Maximum results to return.

    Returns:
        List of result dicts from LanceDB.
    """
    results = (
        _db_table
        .search(query_vector)
        .where(f"modality = '{modality}'")
        .limit(limit)
        .to_list()
    )
    return results


def _cluster_visual_results(
    visual_results: list[dict],
    gap_seconds: float = CLUSTER_GAP_SECONDS,
) -> list[dict]:
    """
    Group nearby visual frame results into temporal event clusters.

    Instead of treating each frame independently, this groups
    consecutive frames that are close in time. A cluster of 10
    frames scoring moderately is much more meaningful than 1
    frame scoring slightly better at a random timestamp.

    Each cluster gets:
        - start_time: earliest frame timestamp in the cluster
        - end_time: latest frame timestamp in the cluster
        - frame_count: how many frames matched in this region
        - avg_distance: average cosine distance (lower = better)
        - cluster_score: composite score = frame_count / avg_distance
        - best_distance: the best single-frame distance in the cluster

    Args:
        visual_results: Raw visual search results sorted by distance.
        gap_seconds: Max gap between frames to be in the same cluster.

    Returns:
        List of cluster dicts sorted by cluster_score (best first).
    """
    if not visual_results:
        return []

    # Sort by timestamp for clustering
    sorted_by_time = sorted(visual_results, key=lambda r: r["timestamp"])

    clusters = []
    current_cluster = [sorted_by_time[0]]

    for i in range(1, len(sorted_by_time)):
        frame = sorted_by_time[i]
        prev_frame = current_cluster[-1]

        # If this frame is within gap_seconds of the previous, same cluster
        if frame["timestamp"] - prev_frame["timestamp"] <= gap_seconds:
            current_cluster.append(frame)
        else:
            # Finalize the current cluster and start a new one
            clusters.append(current_cluster)
            current_cluster = [frame]

    # Don't forget the last cluster
    clusters.append(current_cluster)

    # Score each cluster
    scored_clusters = []
    for cluster_frames in clusters:
        timestamps = [f["timestamp"] for f in cluster_frames]
        distances = [f.get("_distance", 999) for f in cluster_frames]

        frame_count = len(cluster_frames)
        avg_distance = sum(distances) / frame_count
        best_distance = min(distances)

        # Cluster score: more frames + lower distance = better
        # We use frame_count / avg_distance so dense, low-distance clusters win
        cluster_score = frame_count / (avg_distance + 0.001)

        # Find the scene_id that appears most often in this cluster
        scene_ids = [f["scene_id"] for f in cluster_frames]
        dominant_scene = max(set(scene_ids), key=scene_ids.count)

        scored_clusters.append({
            "start_time": min(timestamps),
            "end_time": max(timestamps),
            "timestamp": min(timestamps),  # for RRF keying
            "frame_count": frame_count,
            "avg_distance": round(avg_distance, 4),
            "best_distance": round(best_distance, 4),
            "cluster_score": round(cluster_score, 4),
            "scene_id": dominant_scene,
            "modality": "visual",
            "text": (
                f"[Visual cluster: {frame_count} frames, "
                f"scene {dominant_scene}, "
                f"{min(timestamps):.1f}s - {max(timestamps):.1f}s]"
            ),
            "_distance": best_distance,  # Use best frame for RRF comparison
        })

    # Sort by cluster_score (highest first)
    scored_clusters.sort(key=lambda c: c["cluster_score"], reverse=True)

    return scored_clusters


def _reciprocal_rank_fusion(
    visual_clusters: list[dict],
    audio_results: list[dict],
    caption_results: list[dict] = None,
    visual_weight: float = VISUAL_WEIGHT,
    audio_weight: float = AUDIO_WEIGHT,
    caption_weight: float = CAPTION_WEIGHT,
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge visual clusters, audio, and caption results using RRF.

    Args:
        visual_clusters: Ranked visual clusters from temporal clustering.
        audio_results: Ranked results from audio-only search.
        caption_results: Ranked results from caption-only search (BLIP-2).
        visual_weight: Weight multiplier for visual scores.
        audio_weight: Weight multiplier for audio scores.
        caption_weight: Weight multiplier for caption scores.
        k: Smoothing constant (standard: 60).

    Returns:
        Combined results sorted by fused RRF score (descending).
    """
    fused = []

    # Score visual clusters
    for rank, cluster in enumerate(visual_clusters):
        rrf_score = visual_weight / (k + rank + 1)
        fused.append({
            "data": cluster,
            "rrf_score": rrf_score,
            "sources": ["visual"],
        })

    # Score audio results
    for rank, r in enumerate(audio_results):
        rrf_score = audio_weight / (k + rank + 1)
        fused.append({
            "data": r,
            "rrf_score": rrf_score,
            "sources": ["audio"],
        })

    # Score caption results (BLIP-2 scene descriptions)
    if caption_results:
        for rank, r in enumerate(caption_results):
            rrf_score = caption_weight / (k + rank + 1)
            fused.append({
                "data": r,
                "rrf_score": rrf_score,
                "sources": ["caption"],
            })

    # Sort by fused score (highest first)
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return fused


def retrieve(
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    visual_weight: float = VISUAL_WEIGHT,
    audio_weight: float = AUDIO_WEIGHT,
) -> list[dict]:
    """
    Perform hybrid semantic search with temporal clustering + RRF.

    Pipeline:
        1. Embed the user's text query with CLIP
        2. Search visual frames -> cluster nearby matches into events
        3. Search audio transcripts independently
        4. Fuse both streams with Reciprocal Rank Fusion
        5. Return top_k results mixing both modalities

    Args:
        query_text: Natural language query.
        top_k: Number of final results to return after fusion.
        visual_weight: Weight for visual modality in RRF.
        audio_weight: Weight for audio modality in RRF.

    Returns:
        A list of dicts with timestamps, text, modality, scores, etc.
    """
    _load_db()

    query_vector = embed_query(query_text)

    # Step 1: Search each modality independently
    visual_results = _search_by_modality(query_vector, "visual", CANDIDATES_PER_MODALITY)
    audio_results = _search_by_modality(query_vector, "audio", CANDIDATES_PER_MODALITY)

    # Step 1b: Search captions if they exist in the DB
    try:
        caption_results = _search_by_modality(query_vector, "caption", CANDIDATES_PER_MODALITY)
    except Exception:
        caption_results = []  # Graceful fallback if no captions in DB

    # Step 2: Cluster visual results into temporal events
    visual_clusters = _cluster_visual_results(visual_results)

    # Step 3: Fuse with RRF (now includes captions)
    fused = _reciprocal_rank_fusion(
        visual_clusters, audio_results,
        caption_results=caption_results,
        visual_weight=visual_weight,
        audio_weight=audio_weight,
    )

    # Step 4: Extract top_k and clean up
    cleaned = []
    for entry in fused[:top_k]:
        r = entry["data"]
        result = {
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "timestamp": r["timestamp"],
            "text": r["text"],
            "modality": r["modality"],
            "scene_id": r["scene_id"],
            "_distance": r.get("_distance", None),
            "rrf_score": entry["rrf_score"],
            "sources": entry["sources"],
        }

        # Add cluster-specific metadata if it's a visual cluster
        if r["modality"] == "visual":
            result["frame_count"] = r.get("frame_count", 1)
            result["cluster_score"] = r.get("cluster_score", 0)

        cleaned.append(result)

    return cleaned


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "show me all suit-up scenes of Iron Man",
        "explosion or action scene",
        "Tony Stark talking to someone",
        "flying in the sky",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: \"{query}\"")
        print(f"{'='*70}")

        results = retrieve(query, top_k=5)

        for i, r in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Time:     [{r['start_time']:.2f}s - {r['end_time']:.2f}s]")
            print(f"    Modality: {r['modality']}")
            print(f"    Text:     {r['text'][:80]}...")

            if r["modality"] == "visual":
                print(f"    Frames:   {r.get('frame_count', '?')} frames in cluster")
                print(f"    ClScore:  {r.get('cluster_score', '?')}")
            else:
                print(f"    Distance: {r['_distance']:.4f}")

            print(f"    RRF:      {r['rrf_score']:.6f}")
        print()
