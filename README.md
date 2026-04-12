# 🎬 High-Accuracy Multimodal Video RAG 

A multimodal AI search engine for long-form video. This system allows users to find highly specific, granular scenes in massive video files (like 2-hour movies) using natural language. It mathematically processes Visuals, Audio, and AI-generated Captions to perform hyper-accurate, high-speed video retrieval, instantly playing the extracted clip in a web interface.

---

## 🚀 The Core Philosophy
Static images and raw text are easy to search. Video is incredibly difficult because it introduces the dimension of **time**. This project solves the "Temporal Blindspot" of standard vector searches by using Temporal Frame Clustering, Reciprocal Rank Fusion (RRF), and a 4-Phase LLM Intelligence layer to achieve extreme precision.

## 🛠️ Tools & Technologies Used (And WHY)

| Technology | Purpose | Why We Used It |
|---|---|---|
| **PySceneDetect** | Video Segmentation | Slicing a movie into standard 5-second chunks destroys scenes. PySceneDetect mathematically calculates pixel thresholds to cut the video based on actual camera cuts/scene changes. |
| **Faster-Whisper** | Audio Extraction | Extracts high-fidelity spoken dialogue natively from the video. It is significantly faster and uses less VRAM than standard OpenAI Whisper. |
| **OpenAI CLIP (ViT-B/32)** | Embedding Engine | CLIP maps text and images into the *exact same vector space*. It allows us to search visual frames using natural language strings natively. |
| **BLIP-2 (Salesforce)** | Video Captioning | Sometimes CLIP fails to understand complex actions. We used BLIP-2 to look at frames and generate rich text descriptions (e.g., "A man in red armor flying"), turning pure vision into heavily searchable semantic text. |
| **LanceDB OSS** | Vector Database | A serverless, purely local vector database. LanceDB performs blazing fast vector similarity searches (< 200ms) without requiring API keys, cloud subscriptions, or credit cards. |
| **Llama-3.3-70B (via Groq)**| Intelligence Layer | We needed a model capable of immense reasoning to filter out false positives. We ran it on Groq because Groq enables ~800 tokens/second inference, keeping our search latency under 3 seconds. |
| **FFmpeg** | Video Splicing | Standard video-cutting requires re-encoding which takes minutes. We utilize FFmpeg with "Input-Seeking" (`-ss` before `-i`), cutting video chunks in literal milliseconds. |
| **Streamlit** | Frontend UI | Allowed us to build a seamless WhatsApp-style chat interface rapidly. We heavily leveraged `@st.cache_resource` to keep the Heavy CLIP model globally cached in RAM, eliminating 10-second cold-start delays. |

---

## 🧠 The Architecture Pipeline (How it works)

The system operates in two completely separate environments: **Indexing** (done once) and **Retrieval** (done dynamically by the user).

### Part 1: Offline Indexing (`colab_indexer.py`)
Because processing a 2-hour movie requires substantial VRAM, the indexer was designed to run on a Google Colab T4 GPU.
1. The video is ingested and **PySceneDetect** slices it into discrete semantic scenes.
2. **Faster-Whisper** pulls transcribes all audio. 
3. **BLIP-2** generates rich visual captions for keyframes.
4. **CLIP** embeds the visual frames, the audio texts, and the caption texts into 512-dimension vectors.
5. Everything is packaged gracefully into a highly portable `LanceDB` database folder and downloaded locally.

### Part 2: Multimodal Search (`retriever.py`)
When a user searches for a specific scene, the engine does not just do a naive vector lookup.
1. **Temporal Clustering:** A pure vector-search might return frame 1024, frame 1040, and frame 1042. Instead of returning them individually, the retriever groups nearby frames into cohesive "Event Clusters".
2. **Reciprocal Rank Fusion (RRF):** The engine searches the Visual, Audio, and Caption databases independently. It mathematically merges the rankings. If a scene is found via Visuals *and* Audio, its score skyrockets.
3. **Cross-Modal Intersection:** The retriever employs a strict `MIN_SOURCES = 2` filter. A clip is discarded as a hallucination unless at least TWO modalities (e.g., Visuals + Captions) confirm the match. 
4. **Golden Visual Exception:** If a visual match is mathematically exceptional (Top 5 RRF Score), it bypasses the intersection filter to prevent bad captions from ruining a perfect visual identification.

### Part 3: LLM Synthesizer (`llm_synthesizer.py`)
To achieve sub-3-second human-level accuracy, the pipeline hands off the data to Llama-3.3-70B to act as a strict mediator across four phases:
1. **Dynamic Context & High-Fidelity Expansion:** The system extracts the movie contextual name from the raw `.mp4` file name. It assesses the user's prompt. If the prompt is vague (e.g. "Action Scene"), the LLM expands it into 4 super-descriptive sub-queries unique to that specific movie. If the user's prompt is highly descriptive, it is classified as a "High-Fidelity Query" and safely injected straight into the search.
2. **Multi-Query Retrieval:** LanceDB executes the list of diverse queries concurrently, widening the net so nothing is missed.
3. **The LLM Filter (False Positive Rejection):** Llama evaluates the candidate clips against the user's *actual* intent. If the user asks for "Tony Stark suiting up", Llama strictly discards the scenes of the villain building a suit, correcting inherent flaws in vector mathematics.
4. **Synthesis:** The LLM generates a conversational response containing the exact timestamps.

### Part 4: Dynamic Frontend Extraction (`app.py` & `clipper.py`)
1. The backend passes the verified timestamps `[start_time, end_time]` to `clipper.py`.
2. FFmpeg slices the chunks directly out of the local source `.mp4` in ~200 milliseconds. 
3. The Streamlit UI renders an inline HTML5 `<video>` player dynamically inside the Chat Bot, allowing the user to press play and watch the results instantly. 

---

## 💻 Getting Started (Local Setup)

This project requires a pre-indexed LanceDB folder and the source `.mp4` file to operate locally.

1. **Clone & Setup Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Supply the API Key**
   Create a `.env` file in the root directory and add your Groq key:
   ```env
   GROQ_API_KEY="gsk_your_api_key_here..."
   ```

3. **Run the App**
   Ensure your `.mp4` video file and the `VideoRAG_DB/` folder are in the root directory.
   ```bash
   streamlit run app.py
   ```
   *Note: Your first query will take ~10-15 seconds as the CLIP model is cached into memory. All subsequent queries will run in under 3 seconds.*
