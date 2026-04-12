"""
============================================================
Video-RAG LLM Synthesizer (Step 5)
============================================================
PURPOSE:
    The intelligence layer of the Video-RAG pipeline. This module
    uses a Groq-hosted LLM (Llama 3) in a 4-phase pipeline:

    Phase 1: QUERY EXPANSION
        Takes a vague user query ("suit up scene") and generates
        3-5 hyper-specific sub-queries to cast a wider retrieval net.

    Phase 2: HYBRID RETRIEVAL
        Runs each expanded query through the retriever module to
        collect a broad pool of candidate results.

    Phase 3: LLM RERANK/FILTER
        Feeds all candidate metadata back to the LLM to discard
        false positives (e.g., villain building a suit vs. hero
        suiting up).

    Phase 4: SYNTHESIS
        Generates a conversational response with timestamp citations
        for the confirmed results.

USAGE:
    As a module:
        from llm_synthesizer import process_query
        response = process_query("show me all suit-up scenes")

    Standalone test:
        python llm_synthesizer.py
============================================================
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
from retriever import retrieve, embed_query

load_dotenv()

# ============================================================
# Configuration
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

# How many expanded sub-queries the LLM should generate
NUM_SUB_QUERIES = 4

# How many results to fetch per sub-query (wide net)
RESULTS_PER_QUERY = 10

# Final number of clips to return after LLM filtering
MAX_FINAL_CLIPS = 5

# ============================================================
# Groq Client
# ============================================================

_client = None


def _get_client() -> Groq:
    """Lazy-load the Groq client."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env file!")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """
    Make a single LLM call to Groq.

    Args:
        system_prompt: The system instruction.
        user_prompt: The user message.
        temperature: Creativity control (lower = more deterministic).

    Returns:
        The LLM's response text.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# Phase 1: Query Expansion
# ============================================================

def expand_query(user_query: str) -> list[str]:
    """
    Use the LLM to expand a vague user query into multiple
    specific sub-queries for better retrieval coverage.

    Example:
        "suit up scene" -> [
            "Tony Stark assembling Iron Man armor",
            "metal suit pieces attaching to body",
            "helmet locking onto face",
            "JARVIS deploying the suit"
        ]

    Args:
        user_query: The original user query.

    Returns:
        A list of expanded sub-query strings.
    """
    system_prompt = """You are a query expansion engine for a Video-RAG system that searches through movie scenes.

Your job: Take the user's vague query and generate exactly {num} specific, visually descriptive sub-queries that would help find the relevant scenes in a movie.

Rules:
- Each sub-query should describe a DIFFERENT visual aspect of the scene
- Focus on what the scene LOOKS like, not dialogue
- Be specific about objects, actions, colors, and settings
- Include character names if relevant
- Return ONLY a JSON array of strings, nothing else

Example:
User: "suit up scene"
Output: ["Tony Stark assembling red and gold Iron Man armor", "metal suit pieces flying onto a person's body", "Iron Man helmet closing over face", "robotic arms attaching armor plates in workshop"]""".format(num=NUM_SUB_QUERIES)

    user_prompt = f'User query: "{user_query}"\nGenerate {NUM_SUB_QUERIES} expanded sub-queries as a JSON array:'

    response = _llm_call(system_prompt, user_prompt, temperature=0.4)

    # Parse JSON array from LLM response
    try:
        # Try direct JSON parse
        sub_queries = json.loads(response)
        if isinstance(sub_queries, list):
            return sub_queries[:NUM_SUB_QUERIES]
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON array from response text
    try:
        start = response.index("[")
        end = response.rindex("]") + 1
        sub_queries = json.loads(response[start:end])
        if isinstance(sub_queries, list):
            return sub_queries[:NUM_SUB_QUERIES]
    except (ValueError, json.JSONDecodeError):
        pass

    # Last resort: return the original query
    print(f"  [WARN] Query expansion failed, using original query.")
    return [user_query]


# ============================================================
# Phase 2: Multi-Query Retrieval
# ============================================================

def multi_query_retrieve(sub_queries: list[str], results_per_query: int = RESULTS_PER_QUERY) -> list[dict]:
    """
    Run each expanded sub-query through the retriever and merge
    all results, deduplicating by timestamp proximity.

    Args:
        sub_queries: List of expanded query strings.
        results_per_query: How many results to fetch per query.

    Returns:
        A deduplicated list of candidate results.
    """
    all_results = []
    seen_timestamps = set()

    for query in sub_queries:
        results = retrieve(query, top_k=results_per_query)

        for r in results:
            # Deduplicate: skip results within 5 seconds of one already seen
            ts_key = round(r["start_time"] / 5) * 5  # bucket by 5s
            if ts_key not in seen_timestamps:
                seen_timestamps.add(ts_key)
                r["matched_query"] = query  # Track which sub-query found it
                all_results.append(r)

    # Sort by RRF score (best first)
    all_results.sort(key=lambda x: x["rrf_score"], reverse=True)

    return all_results


# ============================================================
# Phase 3: LLM Rerank / Filter
# ============================================================

def rerank_results(user_query: str, candidates: list[dict]) -> list[dict]:
    """
    Pass all candidate results to the LLM for intelligent filtering.
    The LLM discards false positives based on semantic understanding.

    Args:
        user_query: The original user query.
        candidates: List of candidate results from multi-query retrieval.

    Returns:
        Filtered list of results that the LLM confirms are relevant.
    """
    if not candidates:
        return []

    # Build a readable summary of each candidate for the LLM
    candidate_summaries = []
    for i, r in enumerate(candidates):
        minutes = int(r["start_time"] // 60)
        seconds = int(r["start_time"] % 60)
        end_min = int(r["end_time"] // 60)
        end_sec = int(r["end_time"] % 60)

        summary = {
            "id": i,
            "time": f"{minutes:02d}:{seconds:02d} - {end_min:02d}:{end_sec:02d}",
            "modality": r["modality"],
            "description": r["text"][:150],
            "matched_query": r.get("matched_query", "original"),
        }
        candidate_summaries.append(summary)

    system_prompt = """You are a precise video scene filter for a Movie RAG system.

Your job: Given the user's original query and a list of candidate scene results, determine which candidates are TRUE matches and which are FALSE POSITIVES.

Rules:
- A TRUE match must semantically match what the user is ACTUALLY asking for
- Consider the INTENT behind the query, not just keyword matches
- If the user asks for "Iron Man suit-up", a villain building their own suit is NOT a match
- If the user asks for "Tony Stark talking", a scene where someone else talks is NOT a match
- Be STRICT but fair. When in doubt, keep the result.
- Return ONLY a JSON array of the IDs (integers) of the TRUE matches, sorted by relevance
- Return at most {max_clips} results

Example output: [2, 0, 5]""".format(max_clips=MAX_FINAL_CLIPS)

    user_prompt = f"""User's original query: "{user_query}"

Candidate scenes to evaluate:
{json.dumps(candidate_summaries, indent=2)}

Return a JSON array of IDs that are TRUE matches for the user's query (most relevant first):"""

    response = _llm_call(system_prompt, user_prompt, temperature=0.1)

    # Parse the LLM's response to get the filtered IDs
    try:
        selected_ids = json.loads(response)
        if isinstance(selected_ids, list):
            filtered = []
            for idx in selected_ids[:MAX_FINAL_CLIPS]:
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    filtered.append(candidates[idx])
            return filtered
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract JSON from response
    try:
        start = response.index("[")
        end = response.rindex("]") + 1
        selected_ids = json.loads(response[start:end])
        if isinstance(selected_ids, list):
            filtered = []
            for idx in selected_ids[:MAX_FINAL_CLIPS]:
                if isinstance(idx, int) and 0 <= idx < len(candidates):
                    filtered.append(candidates[idx])
            return filtered
    except (ValueError, json.JSONDecodeError):
        pass

    # Last resort: return top candidates as-is
    print(f"  [WARN] LLM rerank failed, returning top candidates unfiltered.")
    return candidates[:MAX_FINAL_CLIPS]


# ============================================================
# Phase 4: Synthesis
# ============================================================

def synthesize_response(user_query: str, filtered_results: list[dict]) -> str:
    """
    Generate a conversational response with timestamp citations.

    Args:
        user_query: The original user query.
        filtered_results: The LLM-filtered results.

    Returns:
        A natural language response string.
    """
    if not filtered_results:
        return f"I couldn't find any scenes matching \"{user_query}\" in the video. Try rephrasing your query with more specific visual details."

    # Build context for the LLM
    scene_descriptions = []
    for i, r in enumerate(filtered_results):
        start_min = int(r["start_time"] // 60)
        start_sec = int(r["start_time"] % 60)
        end_min = int(r["end_time"] // 60)
        end_sec = int(r["end_time"] % 60)

        desc = (
            f"Scene {i+1} ({r['modality']}): "
            f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}] "
            f"{r['text'][:200]}"
        )
        scene_descriptions.append(desc)

    system_prompt = """You are a helpful video assistant for a Movie RAG chatbot.

Your job: Given the user's query and the matching scenes found in the video, write a brief, conversational response.

Rules:
- Mention each scene with its exact timestamp in MM:SS format
- Be concise — 2-4 sentences max
- Sound natural and helpful, like a movie expert
- If multiple clips were found, briefly describe what happens in each
- Do NOT make up information not present in the scene descriptions"""

    user_prompt = f"""User asked: "{user_query}"

Found scenes:
{chr(10).join(scene_descriptions)}

Write a brief conversational response:"""

    return _llm_call(system_prompt, user_prompt, temperature=0.5)


# ============================================================
# Main Pipeline: process_query()
# ============================================================

def process_query(user_query: str) -> dict:
    """
    The main entry point. Runs the full 4-phase pipeline:
        Query Expansion → Multi-Query Retrieval → LLM Rerank → Synthesis

    Args:
        user_query: The user's natural language question.

    Returns:
        A dict containing:
            - "response": The LLM's conversational answer
            - "clips": List of filtered result dicts with timestamps
            - "expanded_queries": The sub-queries generated
            - "candidates_before_filter": How many candidates were found
            - "elapsed_time": Total pipeline time in seconds
    """
    start_time = time.time()

    # Phase 1: Query Expansion
    print(f"\n{'='*60}")
    print(f"Phase 1: Expanding query...")
    expanded_queries = expand_query(user_query)
    print(f"  Generated {len(expanded_queries)} sub-queries:")
    for i, q in enumerate(expanded_queries):
        print(f"    {i+1}. {q}")

    # Phase 2: Multi-Query Retrieval
    print(f"\nPhase 2: Retrieving candidates...")
    candidates = multi_query_retrieve(expanded_queries)
    print(f"  Found {len(candidates)} unique candidates")

    # Phase 3: LLM Rerank/Filter
    print(f"\nPhase 3: LLM filtering false positives...")
    filtered = rerank_results(user_query, candidates)
    print(f"  Kept {len(filtered)} results after filtering")

    # Phase 4: Synthesis
    print(f"\nPhase 4: Generating response...")
    response_text = synthesize_response(user_query, filtered)

    elapsed = time.time() - start_time
    print(f"\nPipeline complete in {elapsed:.2f}s")

    return {
        "response": response_text,
        "clips": filtered,
        "expanded_queries": expanded_queries,
        "candidates_before_filter": len(candidates),
        "elapsed_time": round(elapsed, 2),
    }


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    test_queries = [
        "show me all suit-up scenes of Iron Man",
        "explosion or action scene",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"USER QUERY: \"{query}\"")
        print(f"{'='*70}")

        result = process_query(query)

        print(f"\n--- LLM Response ---")
        print(result["response"])

        print(f"\n--- Verified Clips ---")
        for i, clip in enumerate(result["clips"]):
            start_m = int(clip["start_time"] // 60)
            start_s = int(clip["start_time"] % 60)
            end_m = int(clip["end_time"] // 60)
            end_s = int(clip["end_time"] % 60)
            print(f"  Clip {i+1}: [{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}] "
                  f"({clip['modality']}) {clip['text'][:60]}...")

        print(f"\n  Total time: {result['elapsed_time']}s")
        print(f"  Candidates before filter: {result['candidates_before_filter']}")
        print(f"  Expanded queries: {result['expanded_queries']}")
