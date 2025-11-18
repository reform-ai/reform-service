"""
Receives and processes uploaded video files from frontend
Extracts frames from uploaded video for processing
"""

import tempfile
import os
import re
import uuid
from fastapi import UploadFile
import cv2


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes filename to prevent path traversal and other security issues.
    Returns a safe filename with only alphanumeric, dots, hyphens, and underscores.
    """
    if not filename:
        return None
    
    # Remove path components (prevent directory traversal)
    filename = os.path.basename(filename)
    
    # Remove any remaining path separators
    filename = filename.replace('/', '').replace('\\', '')
    
    # Extract extension
    name, ext = os.path.splitext(filename)
    
    # Sanitize name part: only alphanumeric, dots, hyphens, underscores
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    
    # Sanitize extension: only alphanumeric and dots
    ext = re.sub(r'[^a-zA-Z0-9.]', '', ext)
    
    # Limit length
    name = name[:100]
    ext = ext[:10]
    
    # If name is empty after sanitization, use UUID
    if not name or name == '_':
        name = str(uuid.uuid4())[:8]
    
    return name + ext if ext else name


def accept_video_file(file: UploadFile) -> dict:
    """
    Accepts and validates video file from FormData upload
    Returns file metadata if valid
    """
    if not file.content_type or not file.content_type.startswith('video/'):
        raise ValueError("File must be a video")
    
    sanitized_filename = sanitize_filename(file.filename)
    
    return {
        "filename": sanitized_filename or file.filename,
        "original_filename": file.filename,
        "content_type": file.content_type,
        "file": file
    }


async def save_video_temp(file: UploadFile) -> str:
    """
    Saves uploaded video file to temporary location using streaming.
    Uses sanitized filename to prevent path traversal attacks.
    Returns path to temporary file
    """
    await file.seek(0)
    
    # Sanitize filename and extract safe extension
    sanitized = sanitize_filename(file.filename) if file.filename else None
    if sanitized:
        suffix = os.path.splitext(sanitized)[1]
        # Ensure suffix starts with dot and is safe
        if not suffix.startswith('.'):
            suffix = '.mp4'
        # Limit extension length and sanitize
        suffix = suffix[:10]
        suffix = re.sub(r'[^a-zA-Z0-9.]', '', suffix)
    else:
        suffix = '.mp4'
    
    # Use NamedTemporaryFile which automatically handles secure temp directory
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    chunk_size = 1024 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        temp_file.write(chunk)
    
    temp_file.close()
    return temp_file.name


def process_frames_from_source(source, validate: bool = True) -> tuple:
    """
    General frame processor - accepts video path (str) or frame list.
    Returns tuple of (frames list, fps, frame_validation, fps_validation).
    Usable by both upload (file path) and livestream (frame list).
    """
    if isinstance(source, str):
        # File path - extract frames
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            if validate:
                from src.shared.upload_video.video_validation import validate_extracted_frames, detect_fps_from_video
                frame_validation = validate_extracted_frames([])
                fps, fps_validation = detect_fps_from_video(source, 0)
                return [], fps, frame_validation, fps_validation
            return [], 30.0, None, None
        # Get video properties first to determine if downsampling is needed
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_from_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps_from_cap if fps_from_cap > 0 else 0
        
        # Memory optimization: downsample frames for long videos
        # Target: max 300 frames (~10 seconds at 30fps) to stay under ~150MB memory
        # This accounts for MediaPipe processing + frame storage + visualization
        # MediaPipe and frame copies can use significant memory
        MAX_FRAMES = 300
        frame_skip = 1
        if total_frames > MAX_FRAMES:
            frame_skip = max(1, total_frames // MAX_FRAMES)
        
        frames = []
        frame_index = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Only keep every Nth frame if downsampling
                if frame_index % frame_skip == 0:
                    frames.append(frame)
                frame_index += 1
        except Exception as e:
            cap.release()
            if validate:
                from src.shared.upload_video.video_validation import validate_extracted_frames, detect_fps_from_video
                frame_validation = validate_extracted_frames(frames)
                if "errors" not in frame_validation:
                    frame_validation["errors"] = []
                frame_validation["errors"].insert(0, f"Error during frame extraction: {str(e)}")
                frame_validation["is_valid"] = False
                fps, fps_validation = detect_fps_from_video(source, len(frames))
                return frames, fps, frame_validation, fps_validation
            return frames, 30.0, None, None
        cap.release()
        # Adjust FPS if frames were downsampled
        if frame_skip > 1:
            fps_from_cap = fps_from_cap / frame_skip
        video_path = source
    elif isinstance(source, list):
        # Frame list - use directly
        frames = source
        video_path = None
        fps_from_cap = None
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")
    
    if validate:
        from src.shared.upload_video.video_validation import validate_extracted_frames, detect_fps_from_video
        frame_validation = validate_extracted_frames(frames)
        if video_path:
            # Use adjusted FPS if available, otherwise detect from video
            if fps_from_cap:
                fps, fps_validation = fps_from_cap, {"is_valid": True, "fps": fps_from_cap, "warnings": []}
            else:
                fps, fps_validation = detect_fps_from_video(video_path, len(frames))
        else:
            fps, fps_validation = 30.0, {"is_valid": True, "fps": 30.0, "warnings": []}
        return frames, fps, frame_validation, fps_validation
    
    if video_path:
        # Use adjusted FPS if available, otherwise detect from video
        if fps_from_cap:
            fps = fps_from_cap
        else:
            from src.shared.upload_video.video_validation import detect_fps_from_video
            fps, _ = detect_fps_from_video(video_path, len(frames))
    else:
        fps = fps_from_cap if fps_from_cap else 30.0
    return frames, fps, None, None


def extract_frames(video_path: str, validate: bool = True) -> tuple:
    """
    Upload-specific wrapper - extracts frames from video file path.
    Returns tuple of (frames list, fps, frame_validation, fps_validation).
    Maintains backward compatibility.
    """
    return process_frames_from_source(video_path, validate)


def save_frames_as_video(frames: list, output_path: str, fps: float = 30.0) -> str:
    """
    Saves frames as video file using OpenCV
    Returns path to saved video
    """
    if not frames:
        raise ValueError("No frames to save")
    
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return output_path

