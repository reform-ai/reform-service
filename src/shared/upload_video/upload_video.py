"""
Receives and processes uploaded video files from frontend
Extracts frames from uploaded video for processing
"""

from fastapi import UploadFile


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

