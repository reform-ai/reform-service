"""
Reform Service - FastAPI server
Main entry point for the exercise form analysis service

FastAPI is the web framework (defines routes, endpoints, middleware)
Uvicorn is the ASGI server (runs the FastAPI application)
"""

import os
import tempfile
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.shared.upload_video.upload_video import (
    accept_video_file,
    save_video_temp,
    extract_frames,
    save_frames_as_video
)
from src.shared.pose_estimation.pose_estimation import (
    process_frames_with_pose,
    draw_landmarks_on_frames
)

app = FastAPI(
    title="Reform Service",
    description="Exercise form analysis service using LLM and Computer Vision",
    version="0.1.0"
)

# Create outputs directory for visualization videos
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Mount static files for serving visualization videos
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Configure CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Reform Service API is running", "status": "ok"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    """
    Accepts video file upload via FormData
    Validates file and processes with pose estimation
    Returns visualization with landmarks
    """
    temp_path = None
    output_path = None
    try:
        # Validate file was provided
        if not video.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate content type is video (using accept_video_file function)
        file_info = accept_video_file(video)
        
        # Read file size for validation (need to reset file pointer after)
        contents = await video.read()
        file_size = len(contents)
        await video.seek(0)  # Reset file pointer for processing
        
        # Validate file size (not empty)
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        # Validate file size (reasonable limit, e.g., 500MB)
        MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large: {file_size} bytes. Maximum size: {MAX_FILE_SIZE} bytes"
            )
        
        # Validate file has content (basic check)
        if len(contents) < 100:
            raise HTTPException(
                status_code=400, 
                detail="File appears to be too small to be a valid video"
            )
        
        # Log validation success
        print(f"✅ Video received and validated successfully!")
        print(f"   Filename: {video.filename}")
        print(f"   Size: {file_size} bytes ({round(file_size / (1024 * 1024), 2)} MB)")
        print(f"   Content Type: {video.content_type}")
        
        # Process video: extract frames, pose estimation, visualization
        temp_path = save_video_temp(video)
        frames, fps = extract_frames(temp_path)
        
        landmarks_list = process_frames_with_pose(frames)
        landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30, 31, 32]
        annotated_frames = draw_landmarks_on_frames(frames, landmarks_list, landmark_indices)
        
        # Save to outputs directory with unique filename
        video_id = str(uuid.uuid4())
        output_filename = f"{video_id}.mp4"
        output_path = OUTPUTS_DIR / output_filename
        save_frames_as_video(annotated_frames, str(output_path), fps)
        
        visualization_url = f"http://127.0.0.1:8000/outputs/{output_filename}"
        
        return {
            "status": "success",
            "message": "Video processed successfully",
            "filename": file_info["filename"],
            "content_type": file_info["content_type"],
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "frame_count": len(frames),
            "visualization_path": str(output_path),
            "visualization_url": visualization_url,
            "validated": True
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

