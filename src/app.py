"""
Reform Service - FastAPI server
Main entry point for the exercise form analysis service

FastAPI is the web framework (defines routes, endpoints, middleware)
Uvicorn is the ASGI server (runs the FastAPI application)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.shared.upload_video.upload_video import accept_video_file

app = FastAPI(
    title="Reform Service",
    description="Exercise form analysis service using LLM and Computer Vision",
    version="0.1.0"
)

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
    Returns confirmation with file metadata
    """
    try:
        file_info = accept_video_file(video)
        return {
            "message": "Video file received successfully",
            "filename": file_info["filename"],
            "content_type": file_info["content_type"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

