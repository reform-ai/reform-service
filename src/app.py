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
    Processes with pose estimation and returns visualization
    """
    temp_path = None
    output_path = None
    try:
        file_info = accept_video_file(video)
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
            "message": "Video processed successfully",
            "filename": file_info["filename"],
            "content_type": file_info["content_type"],
            "frame_count": len(frames),
            "visualization_path": output_path,
            "visualization_url": visualization_url
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

