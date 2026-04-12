"""
============================================================
Video-RAG Clipper
============================================================
PURPOSE:
    Takes timestamp ranges returned by the retriever and uses
    FFmpeg to extract precise video clips from the original
    source video. Uses input-seeking (-ss before -i) for
    near-instant clipping even on 2-hour videos.

USAGE:
    As a module:
        from clipper import extract_clips
        clip_paths = extract_clips(
            video_path="./Iron.Man.2.mp4",
            segments=[
                {"start_time": 717.22, "end_time": 719.50},
                {"start_time": 3085.11, "end_time": 3086.97},
            ],
        )

    Standalone test:
        python clipper.py
============================================================
"""

import os
import subprocess
import shutil


# ============================================================
# Configuration
# ============================================================

# Directory to store generated clips
CLIPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")

# Padding in seconds added before/after each clip to ensure
# the event is fully captured (accounts for ±3s accuracy)
PADDING_BEFORE = 3.0
PADDING_AFTER = 5.0

# Default video path (can be overridden in function calls)
DEFAULT_VIDEO_PATH = os.getenv("VIDEO_PATH", "./Iron.Man.2.2010.1080p.BrRip.x264.YIFY.mp4")


# ============================================================
# Core Clipping Logic
# ============================================================

def _seconds_to_timecode(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format for FFmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def extract_single_clip(
    video_path: str,
    start_time: float,
    end_time: float,
    output_path: str,
    padding_before: float = PADDING_BEFORE,
    padding_after: float = PADDING_AFTER,
) -> str:
    """
    Extract a single clip from the source video using FFmpeg.

    Uses input-seeking (-ss before -i) for speed. The -movflags
    +faststart flag ensures the clip is streamable in web browsers
    without needing to download the entire file first.

    Args:
        video_path: Path to the source video file.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        output_path: Where to save the output clip.
        padding_before: Seconds to add before start_time.
        padding_after: Seconds to add after end_time.

    Returns:
        The output_path if successful.
    """
    # Apply padding (clamp start to 0)
    padded_start = max(0.0, start_time - padding_before)
    padded_end = end_time + padding_after
    duration = padded_end - padded_start

    start_tc = _seconds_to_timecode(padded_start)
    duration_tc = _seconds_to_timecode(duration)

    cmd = [
        "ffmpeg",
        "-y",                        # Overwrite output
        "-ss", start_tc,             # Input seeking (fast, uses keyframes)
        "-i", video_path,            # Input file
        "-t", duration_tc,           # Duration of the clip
        "-c:v", "libx264",           # Re-encode video for clean cuts
        "-preset", "ultrafast",      # Fastest encoding speed
        "-crf", "23",                # Quality (lower = better, 23 is default)
        "-c:a", "aac",               # Re-encode audio
        "-b:a", "128k",              # Audio bitrate
        "-movflags", "+faststart",   # Web-streamable output
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error for clip {output_path}:")
        print(result.stderr[-500:])
        raise RuntimeError(f"FFmpeg failed to extract clip: {output_path}")

    return output_path


def extract_clips(
    video_path: str = None,
    segments: list[dict] = None,
    output_dir: str = CLIPS_DIR,
) -> list[str]:
    """
    Extract multiple clips from the source video based on
    retriever output segments.

    Args:
        video_path: Path to the source video file.
        segments: List of dicts with at least 'start_time' and 'end_time'.
        output_dir: Directory to save the clips.

    Returns:
        Generator yielding file paths to the generated clips as they finish.
    """
    if video_path is None:
        video_path = DEFAULT_VIDEO_PATH

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Source video not found: {video_path}\n"
            f"Please ensure the video file exists at this path."
        )

    # Clean and recreate the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # We no longer accumulate clip_paths, we yield directly

    for i, seg in enumerate(segments):
        start = seg["start_time"]
        end = seg["end_time"]
        output_path = os.path.join(output_dir, f"clip_{i+1}.mp4")

        print(f"  Cutting clip {i+1}: [{start:.2f}s - {end:.2f}s] "
              f"(padded: [{max(0, start - PADDING_BEFORE):.2f}s - {end + PADDING_AFTER:.2f}s])")

        extract_single_clip(video_path, start, end, output_path)

        # Verify the file was created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            size_kb = os.path.getsize(output_path) / 1024
            print(f"    -> Saved: {output_path} ({size_kb:.1f} KB)")
            yield output_path
        else:
            print(f"    -> WARNING: Clip file is empty or missing!")
            yield None


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    # Test with hardcoded timestamps from retriever.py output
    test_segments = [
        {"start_time": 717.22, "end_time": 719.50, "text": "Iron Man suit"},
        {"start_time": 3085.11, "end_time": 3086.97, "text": "Iron Man antique"},
        {"start_time": 436.76, "end_time": 437.50, "text": "blow something up"},
    ]

    print("=" * 60)
    print("Clipper Test: Extracting 3 clips from retriever results")
    print(f"Video: {DEFAULT_VIDEO_PATH}")
    print("=" * 60)

    try:
        paths = list(extract_clips(segments=test_segments))
        print("\nAll clips extracted successfully!")
        for p in paths:
            print(f"  -> {p}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nTo test the clipper, place the video file in the project root.")
