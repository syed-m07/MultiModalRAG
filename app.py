"""
============================================================
Video-RAG Streamlit App (Step 6)
============================================================
PURPOSE:
    The user-facing chat interface for the Video-RAG pipeline.
    Users type natural language queries, and the app returns
    conversational responses with playable video clips.

    Uses @st.cache_resource to load CLIP once into memory,
    dropping per-query latency from ~11s to ~2.5s.

RUN:
    streamlit run app.py
============================================================
"""

import os
import time
import glob
import streamlit as st

# ============================================================
# Page Configuration (MUST be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="Video RAG",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Cached Model Loading (Runs ONCE, stays in memory forever)
# ============================================================

@st.cache_resource(show_spinner="Loading CLIP model into memory...")
def load_clip_and_db():
    """
    Load the CLIP model, processor, and LanceDB connection
    exactly once. Streamlit caches this across all reruns.
    """
    import retriever
    retriever._load_clip()
    retriever._load_db()
    return True


# Trigger the cached load immediately on app start
load_clip_and_db()

# Now import the pipeline modules (CLIP is already warm)
from llm_synthesizer import process_query
from clipper import extract_clips


# ============================================================
# Auto-detect Video File
# ============================================================

def _find_video_file() -> str:
    """Find the .mp4 file in the project directory."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    mp4_files = glob.glob(os.path.join(project_dir, "*.mp4"))
    if mp4_files:
        return mp4_files[0]
    return None


VIDEO_PATH = _find_video_file()


# ============================================================
# Custom CSS for Premium Look
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #888;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* Clip card */
    .clip-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .clip-card .clip-title {
        color: #667eea;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .clip-card .clip-meta {
        color: #aaa;
        font-size: 0.8rem;
    }

    /* Pipeline info */
    .pipeline-info {
        background: rgba(102, 126, 234, 0.08);
        border-radius: 8px;
        padding: 0.8rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #888;
        border-left: 3px solid #667eea;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# App Header
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🎬 Video RAG</h1>
    <p>Ask anything about the video — get instant clips with AI-powered search</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Session State for Chat History
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Render Chat History
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # If this message has clips, re-render them
        if "clips" in message:
            for clip_info in message["clips"]:
                clip_path = clip_info["path"]
                if os.path.exists(clip_path):
                    start_m = int(clip_info["start"] // 60)
                    start_s = int(clip_info["start"] % 60)
                    end_m = int(clip_info["end"] // 60)
                    end_s = int(clip_info["end"] % 60)

                    st.markdown(f"""<div class="clip-card">
                        <div class="clip-title">📎 Clip — {start_m:02d}:{start_s:02d} → {end_m:02d}:{end_s:02d}</div>
                        <div class="clip-meta">Sources: {clip_info['sources']}</div>
                    </div>""", unsafe_allow_html=True)
                    st.video(clip_path)

        # If this message has pipeline info, show it
        if "pipeline_info" in message:
            info = message["pipeline_info"]
            st.markdown(f"""<div class="pipeline-info">
                ⚡ Pipeline: {info['elapsed_time']}s &nbsp;|&nbsp;
                🔍 Candidates: {info['candidates_before_filter']} &nbsp;|&nbsp;
                🎯 Verified clips: {info['num_clips']}
            </div>""", unsafe_allow_html=True)


# ============================================================
# Chat Input
# ============================================================

user_query = st.chat_input("Ask about the video... (e.g., 'show me the suit-up scenes')")

if user_query:
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_query)

    # Save to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_query,
    })

    # Process with the full 4-phase pipeline
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching through the video..."):
            result = process_query(user_query)

        # Display the LLM response
        st.markdown(result["response"])

        # Generate and display video clips
        clips_data = []
        if result["clips"] and VIDEO_PATH:
            with st.spinner("🎬 Cutting video clips..."):
                try:
                    clip_paths = extract_clips(
                        video_path=VIDEO_PATH,
                        segments=result["clips"],
                    )

                    for i, (clip_path, clip_info) in enumerate(zip(clip_paths, result["clips"])):
                        if os.path.exists(clip_path):
                            start_m = int(clip_info["start_time"] // 60)
                            start_s = int(clip_info["start_time"] % 60)
                            end_m = int(clip_info["end_time"] // 60)
                            end_s = int(clip_info["end_time"] % 60)
                            sources_str = ", ".join(clip_info.get("sources", ["unknown"]))

                            st.markdown(f"""<div class="clip-card">
                                <div class="clip-title">📎 Clip {i+1} — {start_m:02d}:{start_s:02d} → {end_m:02d}:{end_s:02d}</div>
                                <div class="clip-meta">Sources: {sources_str}</div>
                            </div>""", unsafe_allow_html=True)
                            st.video(clip_path)

                            clips_data.append({
                                "path": clip_path,
                                "start": clip_info["start_time"],
                                "end": clip_info["end_time"],
                                "sources": sources_str,
                            })

                except Exception as e:
                    st.warning(f"⚠️ Could not generate clips: {e}")

        elif not VIDEO_PATH:
            st.info("📁 No .mp4 file found in the project directory. Clips cannot be generated.")

        # Pipeline stats
        pipeline_info = {
            "elapsed_time": result["elapsed_time"],
            "candidates_before_filter": result["candidates_before_filter"],
            "num_clips": len(clips_data),
        }

        st.markdown(f"""<div class="pipeline-info">
            ⚡ Pipeline: {result['elapsed_time']}s &nbsp;|&nbsp;
            🔍 Candidates: {result['candidates_before_filter']} &nbsp;|&nbsp;
            🎯 Verified clips: {len(clips_data)}
        </div>""", unsafe_allow_html=True)

        # Expandable details
        with st.expander("🔎 View expanded sub-queries"):
            for i, q in enumerate(result["expanded_queries"]):
                st.markdown(f"**{i+1}.** {q}")

        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "clips": clips_data,
            "pipeline_info": pipeline_info,
        })
