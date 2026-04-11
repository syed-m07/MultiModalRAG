"""
============================================================
Video-RAG Colab Indexer
============================================================
PURPOSE:
    This script is designed to run inside Google Colab with
    a T4 GPU runtime. It processes a long-form video (1-2 hrs),
    extracts multimodal features (audio transcripts + visual
    embeddings), and saves the resulting LanceDB database to
    Google Drive.

HOW TO USE:
    1. Open Google Colab (colab.research.google.com).
    2. Set Runtime -> Change runtime type -> T4 GPU.
    3. Copy-paste the contents of this file into a Colab cell.
    4. Run the cell.
    5. When finished, download the "VideoRAG_DB" folder from
       your Google Drive and place it into your local project
       root directory.

DEPENDENCIES (installed automatically below):
    faster-whisper, transformers, lancedb, Pillow, torch,
    scenedetect[opencv], pyarrow
============================================================
"""

# ============================================================
# CELL 1: Mount Google Drive & Install Dependencies
# ============================================================

# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages = [
    "faster-whisper",
    "transformers",
    "lancedb",
    "Pillow",
    "torch",
    "scenedetect[opencv]",
    "pyarrow",
]
for pkg in packages:
    install(pkg)

print("All dependencies installed successfully.")


# ============================================================
# CELL 2: Configuration
# ============================================================

import os

# --- USER CONFIGURATION ---
#   Upload your video directly to your Google Drive.
#   Then set the path here to read directly from Drive:
VIDEO_PATH = "/content/drive/MyDrive/Iron.Man.2.2010.1080p.BrRip.x264.YIFY.mp4"

# Where to save the LanceDB database (on your Google Drive)
DB_OUTPUT_PATH = "/content/drive/MyDrive/VideoRAG_DB"

# Frame sampling rate (1 frame per second is recommended)
FRAME_SAMPLE_FPS = 1

# Whisper model size: "tiny", "base", "small", "medium", "large-v3"
# Use "base" for speed, "large-v3" for best accuracy
WHISPER_MODEL_SIZE = "base"

# CLIP model identifier
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Scene detection threshold (lower = more scenes detected)
SCENE_THRESHOLD = 27.0

print(f"Video path: {VIDEO_PATH}")
print(f"DB output path: {DB_OUTPUT_PATH}")


# ============================================================
# CELL 3: Video Preprocessing - Audio Extraction & Scene Detection
# ============================================================

import json
from scenedetect import open_video, SceneManager, ContentDetector

def extract_audio(video_path, audio_output_path="/content/audio.wav"):
    """Extract audio track from video using FFmpeg."""
    cmd = f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_output_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise RuntimeError("Audio extraction failed.")
    print(f"Audio extracted to: {audio_output_path}")
    return audio_output_path

def detect_scenes(video_path, threshold=SCENE_THRESHOLD):
    """Detect scene boundaries using PySceneDetect."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "scene_id": i,
            "start_time": start.get_seconds(),
            "end_time": end.get_seconds(),
            "start_timecode": str(start),
            "end_timecode": str(end),
        })
    print(f"Detected {len(scenes)} scenes.")
    return scenes

# --- Execute ---
audio_path = extract_audio(VIDEO_PATH)
scenes = detect_scenes(VIDEO_PATH)

# Preview the first 5 scenes
for s in scenes[:5]:
    print(f"  Scene {s['scene_id']}: {s['start_timecode']} -> {s['end_timecode']}")


# ============================================================
# CELL 4: Speech-to-Text with Faster-Whisper
# ============================================================

from faster_whisper import WhisperModel

def transcribe_audio(audio_path, model_size=WHISPER_MODEL_SIZE):
    """Transcribe audio using Faster-Whisper with word-level timestamps."""
    print(f"Loading Whisper model: {model_size}...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print("Transcribing... (this may take a while for long videos)")
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,        # Enable VAD to skip silence
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    transcript_chunks = []
    for segment in segments:
        chunk = {
            "start_time": round(segment.start, 2),
            "end_time": round(segment.end, 2),
            "text": segment.text.strip(),
            "words": [],
        }
        if segment.words:
            for word in segment.words:
                chunk["words"].append({
                    "word": word.word,
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                })
        transcript_chunks.append(chunk)

    print(f"Transcription complete. {len(transcript_chunks)} segments generated.")
    return transcript_chunks

# --- Execute ---
transcript_chunks = transcribe_audio(audio_path)

# Preview the first 5 transcript chunks
for tc in transcript_chunks[:5]:
    print(f"  [{tc['start_time']:.2f}s - {tc['end_time']:.2f}s] {tc['text'][:80]}...")


# ============================================================
# CELL 5: Visual Feature Extraction with CLIP
# ============================================================

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def extract_frames_and_embed(video_path, scenes, clip_model_name=CLIP_MODEL_NAME, sample_fps=FRAME_SAMPLE_FPS):
    """
    Extract frames from the video aligned to scenes, and generate
    CLIP visual embeddings for each sampled frame.
    """
    print(f"Loading CLIP model: {clip_model_name}...")
    model = CLIPModel.from_pretrained(clip_model_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = model.to("cuda")
    model.eval()

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    print(f"Video: {duration:.1f}s, {video_fps:.1f} FPS, {total_frames} total frames")

    # Calculate frame interval for sampling
    frame_interval = max(1, int(video_fps / sample_fps))

    visual_records = []
    frame_count = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / video_fps

            # Find which scene this frame belongs to
            scene_id = -1
            for scene in scenes:
                if scene["start_time"] <= timestamp <= scene["end_time"]:
                    scene_id = scene["scene_id"]
                    break

            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Generate CLIP embedding
            with torch.no_grad():
                inputs = processor(images=pil_image, return_tensors="pt").to("cuda")
                # Use get_image_features and handle potential output object
                temp_output = model.get_image_features(**inputs)
                
                if hasattr(temp_output, 'pooler_output') and temp_output.pooler_output is not None:
                    embedding = temp_output.pooler_output
                else:
                    embedding = temp_output

                # Normalize to unit vector for cosine similarity
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embedding_list = embedding.cpu().squeeze().tolist()

            visual_records.append({
                "timestamp": round(timestamp, 2),
                "scene_id": scene_id,
                "modality": "visual",
                "text": f"[Visual frame at {timestamp:.2f}s, scene {scene_id}]",
                "vector": embedding_list,
            })
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} frames ({timestamp:.1f}s / {duration:.1f}s)")

        frame_count += 1

    cap.release()
    print(f"Visual embedding complete. {processed_count} frames embedded.")
    return visual_records

# --- Execute ---
visual_records = extract_frames_and_embed(VIDEO_PATH, scenes)
print(f"Generated {len(visual_records)} visual embeddings.")


# ============================================================
# CELL 6: Embed Transcript Chunks with CLIP Text Encoder
# ============================================================

def embed_transcripts(transcript_chunks, clip_model_name=CLIP_MODEL_NAME):
    """
    Embed transcript text chunks using the CLIP text encoder so they
    live in the same vector space as the visual embeddings.
    """
    print(f"Loading CLIP text encoder: {clip_model_name}...")
    model = CLIPModel.from_pretrained(clip_model_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = model.to("cuda")
    model.eval()

    text_records = []
    for i, chunk in enumerate(transcript_chunks):
        text = chunk["text"]
        if not text or len(text.strip()) < 3:
            continue

        # CLIP has a 77-token limit; truncate gracefully
        truncated_text = text[:300]

        with torch.no_grad():
            inputs = processor(text=[truncated_text], return_tensors="pt", padding=True, truncation=True).to("cuda")
            
            # Use model.get_text_features to get text embeddings.
            temp_embedding = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            
            if hasattr(temp_embedding, 'pooler_output') and temp_embedding.pooler_output is not None:
                embedding = temp_embedding.pooler_output
            else:
                embedding = temp_embedding # Fallback, if it's already the tensor (standard behavior)

            # Normalize to unit vector for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding_list = embedding.cpu().squeeze().tolist()

        text_records.append({
            "timestamp": chunk["start_time"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "modality": "audio",
            "text": text,
            "vector": embedding_list,
        })

        if (i + 1) % 50 == 0:
            print(f"  Embedded {i + 1}/{len(transcript_chunks)} transcript chunks")

    print(f"Transcript embedding complete. {len(text_records)} chunks embedded.")
    return text_records

# --- Execute ---
text_records = embed_transcripts(transcript_chunks)
print(f"Generated {len(text_records)} text embeddings.")


# ============================================================
# CELL 7: Build and Save LanceDB to Google Drive
# ============================================================

import lancedb
import pyarrow as pa
import shutil

def build_lancedb(visual_records, text_records, db_path=DB_OUTPUT_PATH):
    """
    Merge visual and text records into a single LanceDB table.
    Note: We build locally first to avoid Google Drive FUSE 'Operation not permitted' errors.
    """
    vector_dim = len(visual_records[0]["vector"])
    print(f"Vector dimensionality: {vector_dim}")

    all_records = []
    for rec in visual_records:
        all_records.append({
            "vector": rec["vector"],
            "timestamp": rec["timestamp"],
            "start_time": rec.get("timestamp", 0.0),
            "end_time": rec.get("timestamp", 0.0),
            "modality": rec["modality"],
            "text": rec["text"],
            "scene_id": rec.get("scene_id", -1),
        })

    for rec in text_records:
        all_records.append({
            "vector": rec["vector"],
            "timestamp": rec["timestamp"],
            "start_time": rec["start_time"],
            "end_time": rec["end_time"],
            "modality": rec["modality"],
            "text": rec["text"],
            "scene_id": -1,
        })

    print(f"Total records to insert: {len(all_records)}")

    # Use a local temporary path for building to avoid Drive FUSE issues
    local_tmp_path = "/content/local_videorag_db"
    if os.path.exists(local_tmp_path): shutil.rmtree(local_tmp_path)
    
    db = lancedb.connect(local_tmp_path)
    table = db.create_table("video_chunks", data=all_records, mode="overwrite")

    print(f"LanceDB table created locally with {table.count_rows()} rows.")
    
    # Now move it to Google Drive
    if os.path.exists(db_path): shutil.rmtree(db_path)
    shutil.copytree(local_tmp_path, db_path)
    
    print(f"Database successfully moved to Google Drive: {db_path}")
    return table

# --- Execute ---
table = build_lancedb(visual_records, text_records)

# Quick sanity check
print("\n--- Sanity Check: Test Query ---")
test_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to("cuda")
test_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
test_model.eval()

test_query = "a person talking"
with torch.no_grad():
    inputs = test_processor(text=[test_query], return_tensors="pt", padding=True).to("cuda")
    temp_query_vec = test_model.get_text_features(**inputs)
    
    # Fix: Extract tensor from output object if necessary
    if hasattr(temp_query_vec, 'pooler_output') and temp_query_vec.pooler_output is not None:
        query_vec = temp_query_vec.pooler_output
    else:
        query_vec = temp_query_vec
        
    query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
    query_vec = query_vec.cpu().squeeze().tolist()

results = table.search(query_vec).limit(3).to_list()
print(f"Query: '{test_query}'")
for i, r in enumerate(results):
    print(f"  Result {i+1}: [{r['start_time']:.2f}s - {r['end_time']:.2f}s] ({r['modality']}) {r['text'][:60]}...")