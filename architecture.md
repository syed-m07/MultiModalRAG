# 🎬 MultiModal Video RAG — Architecture Deep-Dive

> **Author:** maaz (maaz.cs05@gmail.com)
> **Repo:** https://github.com/syed-m07/MultiModalRAG
> **Branch:** `main` (11 commits, all on a single day: April 12 2026)
> **Test video:** Iron Man 2 (2010) 1080p — 2hr 04min, 1.72 GB

---

## 1. Project Brief & Origin

This is a **Multimodal Video Retrieval-Augmented Generation (RAG)** system. The problem statement (preserved verbatim in `full_code.txt`) was:

> *"Design and build a Video-based RAG chatbot that allows users to upload long-form videos (1–2 hours), ask natural language questions about specific events, and receive automatically extracted video clips corresponding to those events."*

The hard constraints from the spec were:
- Query latency **< 5 seconds**
- Clip accuracy **±3 seconds**
- Support for **1–2 hour videos**
- Must return **multiple clips per query**
- Must accept **natural language queries**

The solution goes significantly beyond the spec's naive expected flow (which suggested FAISS/Pinecone and a simple CLIP+Whisper pipeline), ultimately arriving at a **4-phase LLM intelligence layer** on top of a **3-modality hybrid search engine** with **temporal frame clustering** and **Reciprocal Rank Fusion (RRF)**.

---

## 2. Repository Structure

```
Multi Modal RAG/
│
├── colab_indexer.py         # OFFLINE step — runs on Google Colab T4 GPU
│                            #   Processes the raw .mp4 into a LanceDB database
│
├── retriever.py             # ONLINE step — local hybrid semantic search engine
│                            #   Temporal clustering + 3-modality RRF fusion
│
├── llm_synthesizer.py       # ONLINE step — 4-phase LLM intelligence pipeline
│                            #   Query Expansion → Multi-Retrieve → Rerank → Synthesis
│
├── clipper.py               # ONLINE step — FFmpeg-based video clip extractor
│                            #   Generator-based streaming to avoid UI blocking
│
├── app.py                   # FRONTEND — Streamlit chat UI
│                            #   @st.cache_resource for CLIP warm-loading
│
├── VideoRAG_DB/             # OUTPUT of colab_indexer.py (NOT in git)
│   └── video_chunks.lance/  # LanceDB table: 3-modality vector store
│       ├── data/            # Apache Arrow/Lance format data files
│       ├── _versions/       # LanceDB versioning metadata
│       └── _transactions/   # LanceDB write-ahead log
│
├── old_VideoRAG_DB/         # v1 database (2-modality: visual + audio only)
│                            # Superseded when BLIP-2 captioning was added
│
├── clips/                   # Runtime-generated clip output directory (NOT in git)
│   ├── clip_1.mp4           # (~13 MB each, re-encoded with libx264 ultrafast)
│   ├── clip_2.mp4
│   └── ...
│
├── Iron.Man.2.2010.1080p... # Source video (NOT in git, 1.72 GB)
│
├── requirements.txt         # Local runtime deps only (no Colab deps listed)
├── .env                     # GROQ_API_KEY stored here
├── .gitignore               # Excludes: .env, venv, *.mp4, VideoRAG_DB/, clips/, error.txt
│
├── timeline.txt             # Developer journal — 6-phase problem/solution log (NOT in git)
├── error.txt                # Raw Colab output logs + debugging notes (NOT in git)
├── full_code.txt            # Original problem statement verbatim (NOT in git)
└── Engineering Design...docx # Formal project design document (NOT in git)
```

---

## 3. High-Level Architecture: Two-Environment Design

The system is split into two completely separate runtime environments:

```
┌─────────────────────────────────────────────────────────┐
│              ENVIRONMENT 1: GOOGLE COLAB                │
│                   (runs ONCE, offline)                  │
│                                                         │
│  Raw .mp4  ──► colab_indexer.py ──► VideoRAG_DB/        │
│                                     (LanceDB folder)    │
└───────────────────────┬─────────────────────────────────┘
                        │ Download to local machine
                        ▼
┌─────────────────────────────────────────────────────────┐
│              ENVIRONMENT 2: LOCAL MACHINE               │
│               (runs on every user query)                │
│                                                         │
│  User Query                                             │
│      │                                                  │
│      ▼                                                  │
│  app.py (Streamlit UI)                                  │
│      │                                                  │
│      ▼                                                  │
│  llm_synthesizer.py  ◄──►  retriever.py                 │
│      │                          │                       │
│      │                          └──► VideoRAG_DB/       │
│      ▼                                                  │
│  clipper.py ──► FFmpeg ──► clips/clip_N.mp4             │
│      │                                                  │
│      ▼                                                  │
│  app.py renders <video> inline in chat                  │
└─────────────────────────────────────────────────────────┘
```

The Colab/local split exists for a concrete reason: the BLIP-2 model (`Salesforce/blip2-opt-2.7b`) and CLIP model require a CUDA GPU for practical processing speeds. The indexing of a 2-hour movie took the full T4 session. The resulting `VideoRAG_DB/` folder is a portable, self-contained directory that only requires CPU at query time.

---

## 4. Part 1: Offline Indexing Pipeline (`colab_indexer.py`)

This script is structured as **8 sequential Colab cells**. It must run on a T4 GPU runtime. The full pipeline processes a raw `.mp4` into a 3-column LanceDB vector database.

### 4.1 Configuration (Cell 2)

```python
VIDEO_PATH      = "/content/drive/MyDrive/Iron.Man.2.2010.1080p.BrRip.x264.YIFY.mp4"
DB_OUTPUT_PATH  = "/content/drive/MyDrive/VideoRAG_DB"
FRAME_SAMPLE_FPS = 1          # 1 frame per second sampled for visual embedding
WHISPER_MODEL_SIZE = "base"   # faster-whisper model size
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"   # 512-dim embedding space
BLIP_MODEL_NAME = "Salesforce/blip2-opt-2.7b"      # 2.7B param captioner
SCENE_THRESHOLD = 27.0        # PySceneDetect ContentDetector threshold
```

The video lives on Google Drive. The DB is also saved to Google Drive. All heavy compute happens on Colab's local `/content/` disk.

### 4.2 Cell 3 — Audio Extraction & Scene Detection

**Audio extraction** uses raw `subprocess.run(ffmpeg ...)` to pull the audio track as a 16kHz mono WAV file (`/content/audio.wav`). This format is what Faster-Whisper expects natively.

**Scene detection** uses `PySceneDetect` with `ContentDetector(threshold=27.0)`. This is a **content-aware** detector: it computes per-frame HSV histogram differences and fires a scene boundary when the delta exceeds the threshold. This is deliberately chosen over blind fixed-interval chunking because:
- Fixed 5-second chunks destroy continuity of action sequences
- Content-aware cuts align with real camera shot changes
- Each "scene" becomes a meaningful semantic unit

The Iron Man 2 movie produced **775 scenes** detected out of 179,320 total frames (at 24 FPS). A known anomaly: the progress bar stopped at 34% (60,835/179,320 frames) with no error — this is a known PySceneDetect behavior where it stops scanning at a sufficient keyframe density and doesn't always traverse the full frame count. The scene list was still complete.

Output schema per scene:
```python
{
    "scene_id": int,
    "start_time": float,      # seconds
    "end_time": float,        # seconds
    "start_timecode": str,    # "HH:MM:SS.mmm"
    "end_timecode": str,
}
```

### 4.3 Cell 4 — Speech-to-Text with Faster-Whisper

`WhisperModel("base", device="cuda", compute_type="float16")` transcribes the full audio. Key settings:
- `beam_size=5` — standard beam search
- `word_timestamps=True` — enables word-level timing for precise segment boundaries
- `vad_filter=True` with `min_silence_duration_ms=500` — Voice Activity Detection filter removes music/SFX noise before transcription, reducing hallucination

Each output chunk has `start_time`, `end_time`, and `text`. Word-level data is captured but only segment-level data is stored in the final DB.

### 4.4 Cell 5 — Visual Feature Extraction with CLIP

`CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` on CUDA. The video is read frame-by-frame using `cv2.VideoCapture`. At 24 FPS and `FRAME_SAMPLE_FPS=1`, every 24th frame is sampled (1 frame/second).

For each sampled frame:
1. Determine which `scene_id` it belongs to via timestamp range check
2. Convert BGR → RGB → PIL Image
3. Pass through `model.get_image_features()` → 512-dimensional vector
4. L2-normalize to unit vector (cosine similarity space)

**Known anomaly documented in `error.txt`:** Only 4,367 frames were embedded instead of the expected ~7,479 (for a 7479-second video at 1 FPS). The root cause was that `cv2.VideoCapture` read from Google Drive via FUSE, and the Drive stream silently dropped frames past a certain point. The data was still functionally valid for retrieval — the second half of the movie had reduced visual coverage but wasn't entirely missing.

Output record per frame:
```python
{
    "timestamp": float,    # seconds
    "scene_id": int,
    "modality": "visual",
    "text": "[Visual frame at T.TTs, scene N]",
    "vector": [float * 512],   # L2-normalized CLIP embedding
}
```

### 4.5 Cell 6 — BLIP-2 Scene Captioning (v2 Addition)

This cell was added in the second major revision (commit `e8621be`). It is the most computationally expensive step.

`Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=float16, device_map="auto")` — loaded in float16 with automatic device mapping to fit in T4 VRAM.

For each of the 775 detected scenes:
1. Calculate the **median frame timestamp** = `(start + end) / 2`
2. Seek to that frame using `cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)` — O(1) seek, no sequential scan
3. Generate a caption with the prompt: `"Describe this movie scene in detail:"`
   - `max_new_tokens=60`, `num_beams=3`
4. Embed the generated text string through the **CLIP text encoder** (NOT BLIP-2's encoder)
   - This is critical: CLIP text and CLIP visual share the same 512-dim vector space, so the caption embedding is directly comparable to visual frame embeddings at query time

This creates a **third modality** ("caption") that bridges pure pixel semantics (CLIP visual) and dialogue (Whisper audio) with rich natural language scene descriptions. Example caption: *"a man in a red and gold suit is standing in a building"*.

### 4.6 Cell 7 — Transcript Embedding

Each Whisper transcript chunk's text is embedded through the **CLIP text encoder** (same model, same vector space). Chunks shorter than 3 characters are skipped. Text is truncated to 300 chars before encoding. These records get `modality = "audio"`.

### 4.7 Cell 8 — LanceDB Assembly & Drive Export

All three record types are merged into a flat list and written to a local temp path `/content/local_videorag_db` first, then `shutil.copytree`'d to Google Drive.

**This two-step save is the critical FUSE workaround.** Writing LanceDB directly to a FUSE-mounted Google Drive path causes PyArrow to throw `Operation not permitted` file-lock errors. The workaround: build entirely on local `/content/` disk (fast SSD), then byte-copy the completed directory to Drive.

Final DB stats for Iron Man 2:
| Modality | Record Count |
|----------|-------------|
| Visual (CLIP image embeddings) | 4,367 |
| Audio (Whisper → CLIP text embeddings) | ~381 |
| Caption (BLIP-2 → CLIP text embeddings) | 775 |
| **Total** | **~5,523** |

The LanceDB schema per row:
```
vector:     fixed_size_list<float32>[512]   ← the embedding
timestamp:  float64
start_time: float64
end_time:   float64
modality:   utf8  ("visual" | "audio" | "caption")
text:       utf8
scene_id:   int64
```

---

## 5. Part 2: Retrieval Engine (`retriever.py` — v3)

This is the local search engine. It runs entirely on CPU at query time. It is designed as an **importable module** (`from retriever import retrieve`) used by `llm_synthesizer.py`.

### 5.1 Configuration & Lazy Globals

```python
DB_PATH              = os.getenv("DB_PATH", "./VideoRAG_DB")
CLIP_MODEL_NAME      = "openai/clip-vit-base-patch32"
RRF_K                = 60       # Smoothing constant (from RRF literature)
VISUAL_WEIGHT        = 1.2      # Boosted: scenes can be silent
CAPTION_WEIGHT       = 0.8
AUDIO_WEIGHT         = 0.3      # Low: audio FPs drag down good visual matches
CANDIDATES_PER_MODALITY = 50    # Wide net before fusion
CLUSTER_GAP_SECONDS  = 15.0    # Frames within 15s = same event
MIN_SOURCES          = 2        # Intersection filter threshold
MERGE_WINDOW_SECONDS = 15.0    # Cross-modality merge window
```

CLIP and LanceDB are **lazy-loaded** into module-level globals (`_clip_model`, `_clip_processor`, `_db_table`). They load once on the first `retrieve()` call and stay in memory. In `app.py`, `@st.cache_resource` pre-warms them at app startup.

### 5.2 Query Embedding

The user's text (or any sub-query string) is embedded through the CLIP text encoder:
1. Tokenize with `CLIPProcessor`
2. `model.get_text_features()` → raw output tensor
3. L2-normalize → 512-dim unit vector

This produces a vector in the **exact same space** as the visual frame embeddings stored in LanceDB. That's the core of CLIP's cross-modal capability.

### 5.3 Per-Modality Search

```python
_search_by_modality(query_vector, modality, limit=50)
```

Each call executes a filtered ANN (Approximate Nearest Neighbor) search against LanceDB:
```python
_db_table.search(query_vector).where(f"modality = '{modality}'").limit(50).to_list()
```

This is called three times per query — once for each of `"visual"`, `"audio"`, `"caption"`. The separation is key: it prevents a heavily-indexed modality from drowning out others.

### 5.4 Temporal Clustering of Visual Results

Raw visual search returns individual frame timestamps. A naive top-5 list might be: frames at 100s, 102s, 103s, 850s, 852s — which are really just **two scenes**, not five. The clustering algorithm fixes this.

**Algorithm:**
1. Sort the 50 visual hits by `timestamp`
2. Walk forward: if the current frame is within `CLUSTER_GAP_SECONDS=15s` of the previous, extend the current cluster
3. Otherwise, close the current cluster and start a new one
4. Score each cluster: `cluster_score = frame_count / (avg_distance + 0.001)`
   - Dense clusters (many matching frames) AND low distances win
   - A cluster of 12 frames with avg_distance=0.2 beats a single frame with distance=0.18

Each cluster becomes a single "event" with:
- `start_time` / `end_time`: range of the cluster
- `frame_count`: number of matching frames
- `best_distance`: the closest single frame in the cluster
- `dominant_scene`: most-common `scene_id` in the cluster

### 5.5 Reciprocal Rank Fusion (RRF)

RRF is the mathematical heart of the system. It merges ranked lists from different sources without needing a calibrated score (distances aren't comparable across modalities).

**Formula per entry:** `rrf_score = weight / (k + rank + 1)` where `k=60`.

The three ranked lists (visual clusters, audio results, caption results) are all converted to RRF scores, then **merged by timestamp proximity** (within `MERGE_WINDOW_SECONDS=15s`):
- If a visual cluster at 720s and an audio result at 725s overlap → they merge into one entry
- The merged entry's `rrf_score` = sum of both individual scores
- The merged entry's `sources` list = `["visual", "audio"]`

This timestamp-merge is what enables the intersection filter downstream.

### 5.6 Intersection Filter + Golden Visual Exception

After RRF fusion, a strict filter is applied:

```python
top_visual_threshold = VISUAL_WEIGHT / (RRF_K + 5)  # ≈ 0.0185

for entry in fused:
    if len(entry["sources"]) >= MIN_SOURCES:        # Normal path: 2+ modalities agree
        filtered.append(entry)
    elif "visual" in entry["sources"] and entry["rrf_score"] >= top_visual_threshold:
        filtered.append(entry)                       # Golden Visual Exception
```

**The Intersection Filter (`MIN_SOURCES=2`)** was introduced to eliminate false positives like the "Ivan Vanko building arc reactor" problem — where CLIP visual alone matched "metal + sparks" for a villain scene instead of the hero suit-up.

**The Golden Visual Exception** was introduced to rescue the "Monaco briefcase suit-up scene" — a scene with no dialogue (silent race track environment) and a confusing BLIP-2 caption (it hallucinated something unrelated). The CLIP visual score was a top-5 match, so the exception preserves it. The threshold `VISUAL_WEIGHT / (RRF_K + 5)` mathematically corresponds to a top-5 visual ranking.

**Fallback:** If the intersection filter removes everything (edge case), the unfiltered fused results are returned to prevent empty responses.

### 5.7 Output Schema

Each final result dict:
```python
{
    "start_time": float,
    "end_time": float,
    "timestamp": float,
    "text": str,              # Best available: caption > audio > visual placeholder
    "modality": str,          # e.g. "visual, caption" (all sources that matched)
    "scene_id": int,
    "_distance": float,
    "rrf_score": float,
    "sources": list[str],
    "frame_count": int,       # visual clusters only
    "cluster_score": float,   # visual clusters only
}
```


---

## 6. Part 3: LLM Intelligence Pipeline (`llm_synthesizer.py`)

This module wraps the retriever with a 4-phase LLM pipeline using **Llama-3.3-70B via Groq**. It is the layer that elevates retrieval accuracy from ~70% to ~99% by using language understanding to filter what pure vector math cannot.

### 6.1 Dynamic Movie Context Discovery

Before any query processing, the system discovers what video it's working with at runtime, without hardcoding:

**Step A — Filename Parsing (`_extract_movie_name`):**
Globs for `*.mp4` in the project directory, then strips junk tokens (resolution, codec, release group names) via regex:
```
"Iron.Man.2.2010.1080p.BrRip.x264.YIFY.mp4"  →  "Iron Man 2 2010"
```
Patterns stripped: `\d{3,4}p`, `BrRip`, `BluRay`, `x264`, `x265`, `YIFY`, `RARBG`, `HDR`, etc.

**Step B — Context Discovery (`_discover_movie_context`):**
Samples 5 random BLIP-2 captions from the DB and asks Llama to summarize the video in one sentence. This context string is injected into every subsequent system prompt to prevent "Batman Hallucinations" (the LLM expanding queries with characters from unrelated movies).

Both values are cached in module-level globals after first call.

### 6.2 Phase 1 — Query Expansion (`expand_query`)

Given `"suit up scene"`, the LLM returns a JSON array like:
```json
[
  "Tony Stark assembling red and gold Iron Man armor",
  "metal suit pieces flying onto a person's body",
  "Iron Man helmet closing over face",
  "robotic arms attaching armor plates in workshop"
]
```

**High-Fidelity Query Logic (commit `df2287d`):**
After expansion, the system computes average word count of sub-queries. If the user's original query word count >= 70% of that average, the original query is appended to the sub-query list verbatim. This prevents the LLM from weakening a well-crafted descriptive prompt.

JSON parsing has a two-level fallback: `json.loads(response)` first, then regex extraction of `[...]` from the response, then return original query.

### 6.3 Phase 2 — Multi-Query Retrieval (`multi_query_retrieve`)

Each sub-query is independently fed through `retrieve()`. Results are deduplicated using 5-second timestamp buckets:
```python
ts_key = round(r["start_time"] / 5) * 5
```
4 sub-queries × 10 results each = up to 40 candidates, typically deduped to ~15-20 unique scenes.

### 6.4 Phase 3 — LLM Rerank/Filter (`rerank_results`)

Candidates are formatted as a JSON summary and passed to Llama with strict instructions:
- Hero suit-up ≠ villain building a suit
- Be STRICT — when in doubt, REJECT
- Return ONLY a JSON array of integer IDs

Temperature: 0.1 (near-deterministic). The LLM returns e.g. `[2, 0, 5]` ordered by relevance.

### 6.5 Phase 4 — Synthesis (`synthesize_response`)

Verified clips are passed to Llama one final time to generate a conversational response mentioning each clip's MM:SS timestamps. Temperature: 0.5.

### 6.6 Full Pipeline Timing

```
Phase 1: Query Expansion        ~0.5s  (1 Groq call)
Phase 2: Multi-Query Retrieval  ~1.5s  (4 × retrieve() calls)
Phase 3: LLM Filter             ~0.5s  (1 Groq call)
Phase 4: Synthesis              ~0.3s  (1 Groq call)
─────────────────────────────────────────
Total:                          ~2.8s  (well under the 5s spec)
```

---

## 7. Part 4: Clip Extraction (`clipper.py`)

### 7.1 FFmpeg Command

```bash
ffmpeg -y
  -ss HH:MM:SS.mmm           # INPUT seeking — fast keyframe seek, O(1)
  -i source_video.mp4
  -t HH:MM:SS.mmm            # Duration
  -c:v libx264 -preset ultrafast -crf 23
  -c:a aac -b:a 128k
  -movflags +faststart        # Web-streamable: moov atom at file start
  clips/clip_N.mp4
```

Input-seeking (`-ss` BEFORE `-i`) reduces clip extraction from minutes (full decode) to ~200ms per clip on a 2-hour 1080p source. Each clip gets 3s prepended and 5s appended as padding for the ±3s timestamp tolerance.

### 7.2 Generator-Based Streaming (commit `7640313`)

`extract_clips()` is a Python generator using `yield`. The Streamlit loop renders each clip immediately as it finishes cutting, before the next clip begins:

```python
for clip_path, clip_info in zip(clip_generator, result["clips"]):
    st.video(clip_path)    # renders clip N while clip N+1 is still cutting
```

This eliminates the perceived UI blocking latency of batch clip generation.

### 7.3 Output Management

The `clips/` directory is wiped and recreated on every query (`shutil.rmtree` + `os.makedirs`). Only the most recent query's clips exist on disk at any time, preventing unbounded disk usage.

Current clips on disk: 5 clips, 6.9MB–19.8MB each (re-encoded libx264).


---

## 8. Part 5: Frontend (`app.py`)

### 8.1 Startup & Model Caching

```python
@st.cache_resource(show_spinner="Loading CLIP model into memory...")
def load_clip_and_db():
    import retriever
    retriever._load_clip()
    retriever._load_db()
    return True

load_clip_and_db()   # called at module level — runs once on first user visit
```

`@st.cache_resource` is Streamlit's cross-session resource cache. The CLIP model (~400MB) and LanceDB connection are loaded exactly once, shared across all reruns and all users. This drops per-query latency from ~11s (cold) to ~2.5s (warm).

### 8.2 Video File Auto-Detection

```python
def _find_video_file():
    mp4_files = glob.glob(os.path.join(project_dir, "*.mp4"))
    return mp4_files[0] if mp4_files else None
```

No hardcoded filename. The system works with any `.mp4` placed in the project root.

### 8.3 Chat History

Stored in `st.session_state.messages` as a list of dicts. Each assistant message stores the full `clips` data (paths, timestamps, sources) and `pipeline_info` (elapsed time, candidate count, clip count) for re-rendering on page refresh.

### 8.4 UI Structure

```
┌─────────────────────────────────────────┐
│  🎬 Video RAG                           │  ← gradient header (Inter font)
│  Ask anything about the video...        │
├─────────────────────────────────────────┤
│  [chat history: user + assistant msgs]  │
│  │                                      │
│  └── assistant messages include:        │
│      • LLM text response                │
│      • clip-card divs (styled HTML)     │
│      • st.video() inline players        │
│      • pipeline stats bar               │
│      • expandable sub-queries           │
├─────────────────────────────────────────┤
│  [st.chat_input] "Ask about the video"  │
└─────────────────────────────────────────┘
```

Custom CSS uses a dark glassmorphism aesthetic: `linear-gradient(135deg, #1a1a2e, #16213e)` for clip cards, `linear-gradient(135deg, #667eea, #764ba2)` for the title, Inter font via Google Fonts.

---

## 9. Complete Data Flow: End-to-End for One Query

```
User types: "show me the briefcase suit-up at Monaco"
│
▼ app.py
├── process_query("show me the briefcase suit-up at Monaco")
│
▼ llm_synthesizer.py — Phase 1: Query Expansion
├── _get_movie_context() → "Iron Man 2 2010", "action sci-fi film..."
├── Groq API call → 4 sub-queries:
│     1. "Tony Stark briefcase transforming into armor at race track"
│     2. "compact suit case unfolding into Iron Man suit"
│     3. "Monaco Grand Prix race circuit emergency suit deployment"
│     4. "briefcase robot armor assembly sequence"
│   + original query appended (High-Fidelity: 9 words ≥ 70% of avg 8.5)
│
▼ llm_synthesizer.py — Phase 2: Multi-Query Retrieval
├── For each of 5 sub-queries → retrieve(query, top_k=10):
│     ├── embed_query → 512-dim vector
│     ├── LanceDB search: visual (50 candidates)
│     ├── LanceDB search: audio (50 candidates)
│     ├── LanceDB search: caption (50 candidates)
│     ├── Temporal clustering → visual clusters
│     ├── RRF fusion → merged entries with sources list
│     ├── Intersection filter (MIN_SOURCES=2)
│     │   + Golden Visual Exception (Monaco: silent scene, bad caption)
│     └── Return top 10 entries
├── Deduplicate by 5s buckets → ~18 unique candidates
│
▼ llm_synthesizer.py — Phase 3: LLM Filter
├── Format 18 candidates as JSON summaries
├── Groq API call (temp=0.1) → [3, 7, 1]  (3 true matches)
│
▼ llm_synthesizer.py — Phase 4: Synthesis
├── Groq API call (temp=0.5) →
│   "I found 3 suit-up scenes. The Monaco briefcase scene
│    appears at 47:02 – 47:36, where Tony deploys his
│    emergency suit from a suitcase..."
│
▼ app.py — Clip Generation
├── clip_generator = extract_clips(video_path, 3 segments)
├── Loop (streaming):
│   ├── FFmpeg cuts clip_1.mp4 (47:02) → yield → st.video()  ← user sees this immediately
│   ├── FFmpeg cuts clip_2.mp4 (...)   → yield → st.video()
│   └── FFmpeg cuts clip_3.mp4 (...)   → yield → st.video()
│
▼ User sees:
├── Conversational LLM response with timestamps
├── 3 inline video players, progressively rendered
└── Pipeline stats: "2.8s | 18 candidates | 3 clips"
```

---

## 10. Development Timeline (from `timeline.txt` + git log)

All 11 commits occurred on a single day: **April 12, 2026**, over approximately 22 hours.

| Time (IST) | Commit | What Changed |
|-----------|--------|-------------|
| Apr 11 23:53 | `f7bf1cc` — "Completed setup" | `colab_indexer.py` v1 (visual + audio only), `requirements.txt`, `.gitignore` |
| Apr 12 02:23 | `679ceb8` — "implemented the retriever" | `retriever.py` v1: basic CLIP text embed + LanceDB query, no clustering |
| Apr 12 16:04 | `a318400` — "temporal progression + hybrid RRF + clipper" | Massive retriever rewrite: temporal clustering, RRF, modality weights. `clipper.py` created |
| Apr 12 18:18 | `e8621be` — "reindexed with BLIP-2" | `colab_indexer.py` v2: BLIP-2 captioning cell added, DB reindexed to 3 modalities. Retriever updated for caption modality |
| Apr 12 19:03 | `b7013f5` — "added llm synthesizer" | `llm_synthesizer.py` v1: 4-phase pipeline, basic query expansion |
| Apr 12 19:24 | `85f52af` — "filename parsing + context + equal weights" | Movie name extraction, context discovery via captions, modality weight tuning |
| Apr 12 21:17 | `df2287d` — "high fidelity query + golden visual exception" | High-Fidelity Query logic in synthesizer, Golden Visual Exception in retriever |
| Apr 12 21:37 | `b30409d` — "small change" | Minor tweak |
| Apr 12 21:46 | `dfeae26` — "added README.md" | README written |
| Apr 12 21:59 | `7640313` — "added streaming clip generation in UI" | Generator-based `extract_clips()`, Streamlit streaming loop in `app.py` |

---

## 11. Strengths

### Mathematical Rigor
- **RRF** is a proven information retrieval technique from academic literature, not ad-hoc score averaging. Using `k=60` is the standard recommended value.
- **Temporal clustering** with `cluster_score = frame_count / avg_distance` is an elegant composite metric that correctly weights density + quality.
- The **Golden Visual Exception threshold** `VISUAL_WEIGHT / (RRF_K + 5)` is mathematically derived from the RRF formula to correspond precisely to a top-5 rank — not a magic number.

### Three-Modality Architecture
Using all three modalities (visual pixel space, audio/dialogue, AI-generated captions) in the same 512-dim CLIP vector space is architecturally sound. Each modality compensates for the others' failure modes.

### Production-Quality Optimizations
- `@st.cache_resource` prevents 12-second cold start on every query
- Generator-based clip streaming eliminates UI lockup
- FFmpeg input-seeking makes clip cuts near-instant
- Lazy-loading of CLIP and LanceDB with module-level globals
- `shutil.rmtree` on clips dir prevents unbounded disk growth

### Problem-Solving Documentation
The `timeline.txt` is exceptional engineering documentation. Every architectural decision is traceable to a specific failure, its root cause, and the exact solution. This is rare in student/solo projects.

### Graceful Degradation
- Caption search has a `try/except` fallback (works even on v1 DB without captions)
- JSON parsing has two fallback levels before giving up
- Intersection filter has a fallback to return unfiltered results if it's too strict
- Missing video file shows a clean UI message instead of crashing

---

## 12. Shortcomings & Technical Debt

### Critical Issues

**1. Incomplete Visual Coverage (`error.txt`)**
Only 4,367 of the expected 7,479 frames were embedded (58% of the movie). The FUSE streaming issue caused `cv2.VideoCapture` to silently stop mid-file when reading from Google Drive. The second ~40 minutes of the film has zero visual embedding coverage. This is a silent data quality bug with no runtime warning.

**2. GROQ_API_KEY Committed Risk**
The `.env` file is gitignored correctly, but the API key value is visible in the repo's untracked file on this machine. The key `gsk_6kcjZQUYd...` is live. No key rotation mechanism exists.

**3. `clips/` Directory Race Condition**
`shutil.rmtree(output_dir)` is called at the start of every `extract_clips()` call. If two users send queries simultaneously (multi-user Streamlit), one user's clips will be deleted mid-generation. The system has no per-session clip isolation.

**4. Visual Records Have Incorrect `start_time`/`end_time`**
In `build_lancedb()`, visual records are stored with `start_time = timestamp` and `end_time = timestamp` (same value). Visual frames are point-in-time, not ranges. This means the RRF merge and clip-padding logic must work around this — the temporal cluster `start_time`/`end_time` override this at query time, but it's a schema inconsistency.

### Architecture Limitations

**5. Single Table, No Index**
LanceDB is used without an explicit ANN index (IVF, HNSW, etc.). At ~5,500 rows, brute-force scan is fast enough (<200ms), but this won't scale. At 100K+ rows (a TV series season), query time would degrade significantly without indexing.

**6. No Video Upload Flow**
The system requires the user to manually place the `.mp4` in the project root and run `colab_indexer.py` on Colab. There is no upload UI, no progress tracking, and no automated Colab trigger. The indexing is a fully manual offline step.

**7. BLIP-2 Hallucination is Unmitigated at Indexing**
BLIP-2 zero-shot captions on movie frames frequently hallucinate. The caption for the Monaco scene was incorrect enough that the Intersection Filter nearly discarded the correct result. The Golden Visual Exception is a downstream patch, not a fix to the root cause. Better BLIP-2 prompting or a VQA-style verification step at index time would be more robust.

**8. `old_VideoRAG_DB/` Left in Repo**
The v1 database (2-modality) is committed to the repository (albeit gitignored from future commits). It adds unnecessary weight and confusion. It should be deleted.

**9. No Persistent Chat Storage**
`st.session_state` is cleared on browser refresh. There is no database or file-backed chat history. Clips are also lost on refresh since `clips/` is wiped each query.

**10. `requirements.txt` Is Incomplete**
Missing: `opencv-python` (cv2 is used in retriever for nothing, but was in Colab), `pyarrow` (used in llm_synthesizer for caption DB scan), `scenedetect` (only in Colab), `faster-whisper` (only in Colab). The local requirements only cover the runtime stack, not the indexing stack, with no comment explaining this split.

**11. Modality Weights Are Hand-Tuned**
`VISUAL_WEIGHT=1.2`, `CAPTION_WEIGHT=0.8`, `AUDIO_WEIGHT=0.3` were determined through trial and error on a single movie. There is no evaluation framework to validate these weights on different content types (documentaries, lectures, sports).

---

## 13. Technology Stack Summary

| Component | Technology | Version/Variant | Role |
|-----------|-----------|----------------|------|
| Scene Detection | PySceneDetect | ContentDetector | Semantic video segmentation |
| Speech-to-Text | Faster-Whisper | `base`, float16 | Audio transcript extraction |
| Visual Embedding | OpenAI CLIP | ViT-B/32, 512-dim | Cross-modal image→vector |
| Scene Captioning | Salesforce BLIP-2 | `blip2-opt-2.7b`, float16 | Natural language scene description |
| Text Embedding | OpenAI CLIP (text encoder) | same model | Transcript + caption→vector |
| Vector Database | LanceDB OSS | serverless, local | ANN similarity search, Apache Arrow storage |
| LLM | Llama 3.3 70B | via Groq API | Query expansion, reranking, synthesis |
| Video Clipping | FFmpeg | libx264 ultrafast | Sub-second clip extraction |
| Frontend | Streamlit | — | Chat UI with inline video |
| Compute (indexing) | Google Colab | T4 GPU, 16GB VRAM | One-time offline processing |
| Compute (runtime) | Local CPU | — | Sub-3s per query |

