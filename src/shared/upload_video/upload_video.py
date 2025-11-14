"""
Receives and processes uploaded video files from frontend
Extracts frames from uploaded video for processing
"""

import tempfile
import os
from fastapi import UploadFile
import cv2


def accept_video_file(file: UploadFile) -> dict:
    """
    Accepts and validates video file from FormData upload
    Returns file metadata if valid
    """
    if not file.content_type or not file.content_type.startswith('video/'):
        raise ValueError("File must be a video")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "file": file
    }


def save_video_temp(file: UploadFile) -> str:
    """
    Saves uploaded video file to temporary location
    Returns path to temporary file
    """
    file.file.seek(0)
    suffix = os.path.splitext(file.filename)[1] if file.filename else '.mp4'
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = file.file.read()
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def extract_frames(video_path: str) -> tuple:
    """
    Extracts frames from video file using OpenCV
    Returns tuple of (frames list, fps)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps


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

