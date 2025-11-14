"""
Reform Service - FastAPI server
Main entry point for the exercise form analysis service

FastAPI is the web framework (defines routes, endpoints, middleware)
Uvicorn is the ASGI server (runs the FastAPI application)
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

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
    """Receive uploaded video file from frontend"""
    try:
        # Validate file was provided
        if not video.filename:
            return {
                "status": "error",
                "message": "No file provided"
            }
        
        # Validate content type is video
        if not video.content_type or not video.content_type.startswith('video/'):
            return {
                "status": "error",
                "message": f"Invalid file type: {video.content_type}. Expected video file."
            }
        
        # Read the video file
        contents = await video.read()
        file_size = len(contents)
        
        # Validate file size (not empty)
        if file_size == 0:
            return {
                "status": "error",
                "message": "Empty file received"
            }
        
        # Validate file size (reasonable limit, e.g., 500MB)
        MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
        if file_size > MAX_FILE_SIZE:
            return {
                "status": "error",
                "message": f"File too large: {file_size} bytes. Maximum size: {MAX_FILE_SIZE} bytes"
            }
        
        # Validate file has content (basic check)
        if len(contents) < 100:  # Very small files are likely not valid videos
            return {
                "status": "error",
                "message": "File appears to be too small to be a valid video"
            }
        
        # Success - video received and validated
        print(f"✅ Video received and validated successfully!")
        print(f"   Filename: {video.filename}")
        print(f"   Size: {file_size} bytes ({round(file_size / (1024 * 1024), 2)} MB)")
        print(f"   Content Type: {video.content_type}")
        
        return {
            "status": "success",
            "filename": video.filename,
            "content_type": video.content_type,
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "message": "Video received and validated successfully",
            "validated": True
        }
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing video: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

