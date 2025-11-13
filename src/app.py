"""
Reform Service - FastAPI server
Main entry point for the exercise form analysis service

FastAPI is the web framework (defines routes, endpoints, middleware)
Uvicorn is the ASGI server (runs the FastAPI application)
"""

from fastapi import FastAPI
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

