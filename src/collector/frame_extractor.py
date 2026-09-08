import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional


def is_frame_blurry(frame: np.ndarray, threshold: float = 80.0) -> bool:
    """Check if an image frame is blurry using Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def resize_frame_maintaining_aspect(frame: np.ndarray, max_dim: int = 640) -> np.ndarray:
    """Resize image so its longest dimension equals max_dim, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    
    scale = max_dim / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.0,
    max_dim: int = 640,
    blur_threshold: float = 80.0,
    skip_blurry: bool = True
) -> List[str]:
    """
    Extracts distinct, token-optimized frames from a video file.
    
    Args:
        video_path: Path to the input video file (.mp4, .avi, .mov, etc.)
        output_dir: Directory where extracted JPEG frames will be saved
        interval_seconds: Time gap in seconds between extracted frames
        max_dim: Maximum dimension (width or height) to resize frames for token efficiency
        blur_threshold: Minimum Laplacian variance to consider a frame sharp
        skip_blurry: Whether to skip frames that fail blur detection
        
    Returns:
        List of saved frame file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found at: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback default
        
    frame_interval = int(round(fps * interval_seconds))
    frame_interval = max(1, frame_interval)

    saved_paths = []
    frame_idx = 0
    extracted_count = 0
    video_stem = video_path.stem

    print(f"🎬 Processing '{video_path.name}' | FPS: {fps:.2f} | Interval: every {frame_interval} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if skip_blurry and is_frame_blurry(frame, blur_threshold):
                frame_idx += 1
                continue

            optimized_frame = resize_frame_maintaining_aspect(frame, max_dim)
            filename = f"{video_stem}_f{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, filename)
            
            # Save at 90% JPEG quality for balanced visual clarity and small byte size
            cv2.imwrite(out_path, optimized_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_paths.append(out_path)
            extracted_count += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {extracted_count} clean frames to '{output_dir}'.")
    return saved_paths
