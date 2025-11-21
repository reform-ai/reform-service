"""Reform Service - FastAPI server for exercise form analysis."""

import os
import tempfile
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from collections import deque
import time
from datetime import datetime
from threading import Lock
from src.shared.upload_video.upload_video import (
    accept_video_file,
    save_video_temp
)
# Import exercise registry - this will auto-register exercises
from src.shared.exercises import get_exercise, EXERCISES
from src.shared.auth.database import init_db, get_db, reset_daily_anonymous_limit_if_needed, calculate_token_cost, AnonymousAnalysis
from src.shared.auth.routes import router as auth_router
from src.shared.auth.dependencies import security
from src.shared.social.routes import router as social_router
from src.shared.payment.routes import router as payment_router
from src.shared.contact.routes import router as contact_router
from fastapi.security import HTTPAuthorizationCredentials

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 5
_upload_rate_limit_store = {}
_upload_rate_limit_lock = Lock()

app = FastAPI(
    title="Reform Service",
    description="Exercise form analysis service using LLM and Computer Vision",
    version="0.1.0"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    import logging
    try:
        init_db()
        # Also initialize social feed tables
        from src.shared.social.database import Base as SocialBase
        from src.shared.auth.database import engine
        SocialBase.metadata.create_all(bind=engine, checkfirst=True)
        # Initialize contact rate limiting tables
        from src.shared.contact.database import Base as ContactBase
        ContactBase.metadata.create_all(bind=engine, checkfirst=True)
        logging.info("Database initialization completed on startup")
    except Exception as e:
        # Log error but don't crash the app
        # Database will be created on first use if needed
        logging.error(f"Database initialization error on startup: {str(e)}")
        # Try to continue - database operations will handle errors gracefully
        # Tables will be created on first auth request as fallback

# Include auth routes
app.include_router(auth_router)

# Include social feed routes
app.include_router(social_router)

# Include payment/token routes
app.include_router(payment_router)

# Include contact routes
app.include_router(contact_router)

# Use /tmp/outputs on Heroku (ephemeral filesystem), otherwise use local outputs directory
if os.environ.get("DYNO"):  # Heroku sets DYNO environment variable
    OUTPUTS_DIR = Path("/tmp/outputs")
else:
    OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
# CORS configuration - must be added before exception handlers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://reformgym.fit",
        "https://reform-client-c95dd550c494.herokuapp.com",
        "https://reform-client-beta-a7a168806e67.herokuapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global exception handlers to ensure CORS headers are always added
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure CORS headers are added to FastAPI HTTP exceptions."""
    headers = {}
    origin = request.headers.get("origin")
    if origin in [
        "http://localhost:3000",
        "https://reformgym.fit",
        "https://reform-client-c95dd550c494.herokuapp.com",
        "https://reform-client-beta-a7a168806e67.herokuapp.com"
    ]:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "*"
        headers["Access-Control-Allow-Headers"] = "*"
    
    # Handle both string and dict detail formats
    if isinstance(exc.detail, dict):
        content = exc.detail
    elif isinstance(exc.detail, str):
        content = {"detail": exc.detail}
    else:
        content = {"detail": str(exc.detail)}
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content,
        headers=headers
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Ensure CORS headers are added to Starlette HTTP exceptions."""
    headers = {}
    origin = request.headers.get("origin")
    if origin in [
        "http://localhost:3000",
        "https://reformgym.fit",
        "https://reform-client-c95dd550c494.herokuapp.com",
        "https://reform-client-beta-a7a168806e67.herokuapp.com"
    ]:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "*"
        headers["Access-Control-Allow-Headers"] = "*"
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail} if isinstance(exc.detail, (str, dict)) else {"detail": str(exc.detail)},
        headers=headers
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure CORS headers are added to validation errors."""
    headers = {}
    origin = request.headers.get("origin")
    if origin in [
        "http://localhost:3000",
        "https://reformgym.fit",
        "https://reform-client-c95dd550c494.herokuapp.com",
        "https://reform-client-beta-a7a168806e67.herokuapp.com"
    ]:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "*"
        headers["Access-Control-Allow-Headers"] = "*"
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
        headers=headers
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Ensure CORS headers are added to all exceptions."""
    import logging
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    headers = {}
    origin = request.headers.get("origin")
    if origin in [
        "http://localhost:3000",
        "https://reformgym.fit",
        "https://reform-client-c95dd550c494.herokuapp.com",
        "https://reform-client-beta-a7a168806e67.herokuapp.com"
    ]:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Access-Control-Allow-Methods"] = "*"
        headers["Access-Control-Allow-Headers"] = "*"
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=headers
    )


@app.get("/")
async def root():
    return {"message": "Reform Service API is running", "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/check-anonymous-limit")
async def check_anonymous_limit(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Check if anonymous user has reached analysis limit. Returns limit status without requiring file upload."""
    # Check if user is authenticated
    user = None
    if credentials:
        from src.shared.auth.auth import verify_token
        from src.shared.auth.database import User
        payload = verify_token(credentials.credentials)
        if payload:
            user_id = payload.get("sub")
            if user_id:
                db = next(get_db())
                try:
                    user = db.query(User).filter(User.id == user_id).first()
                except Exception:
                    pass
                finally:
                    db.close()
    
    # If authenticated, no limit applies (they use token system)
    if credentials is not None and user is not None:
        return {
            "has_limit": False,
            "is_authenticated": True,
            "message": "No limit for authenticated users"
        }
    
    # Skip anonymous limit check when running locally (not on Heroku)
    if not os.environ.get("DYNO"):
        return {
            "has_limit": False,
            "limit_reached": False,
            "analyses_completed": 0,
            "limit": 1,
            "message": "No limit for local testing"
        }
    
    # Check anonymous limit
    client_ip = _get_client_ip(request)
    db = next(get_db())
    try:
        from src.shared.auth.database import Base, engine
        Base.metadata.create_all(bind=engine, checkfirst=True)
        
        anonymous_record = db.query(AnonymousAnalysis).filter(
            AnonymousAnalysis.ip_address == client_ip
        ).first()
        
        # Reset daily limit if it's a new day
        if anonymous_record:
            reset_daily_anonymous_limit_if_needed(anonymous_record)
            db.commit()
            # Re-query to get fresh state after reset
            anonymous_record = db.query(AnonymousAnalysis).filter(
                AnonymousAnalysis.ip_address == client_ip
            ).first()
        
        if anonymous_record and anonymous_record.analysis_count >= 1:
            return {
                "has_limit": True,
                "limit_reached": True,
                "analyses_completed": anonymous_record.analysis_count,
                "limit": 1,
                "message": "Anonymous users are limited to 1 analysis per day. Please sign up for unlimited analyses."
            }
        else:
            return {
                "has_limit": True,
                "limit_reached": False,
                "analyses_completed": anonymous_record.analysis_count if anonymous_record else 0,
                "limit": 1,
                "message": "You have 1 free analysis remaining today"
            }
    except Exception as e:
        import logging
        logging.warning(f"Failed to check anonymous limit: {str(e)}")
        return {
            "has_limit": True,
            "limit_reached": False,
            "error": "Could not check limit status"
        }
    finally:
        db.close()


@app.get("/robots.txt")
async def robots_txt():
    """Returns robots.txt to disallow crawlers from accessing output and tmp directories."""
    return Response(
        content="User-agent: *\nDisallow: /outputs/\nDisallow: /tmp/\nDisallow: /temp/\n",
        media_type="text/plain"
    )


@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """Serves output video files only if they exist and are valid. Prevents directory listing."""
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=403, detail={"error": "forbidden", "message": "Invalid filename"})
    
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail={"error": "not_found", "message": "File not found"})
    
    if not file_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
        raise HTTPException(status_code=403, detail={"error": "forbidden", "message": "Invalid file type"})
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=filename
    )


def get_required_landmarks(exercise: int) -> list:
    """
    General function to get required landmarks for an exercise.
    Usable by both upload and livestream.
    Returns list of required landmark indices or None if not applicable.
    """
    from src.shared.exercises import get_exercise
    exercise_module = get_exercise(exercise)
    if exercise_module:
        return exercise_module.get_required_landmarks()
    return None


def route_to_exercise_calculation(exercise: int, landmarks_list: list, validation_result: dict = None) -> dict:
    """Routes to appropriate exercise calculation module."""
    from src.shared.exercises import get_exercise
    exercise_module = get_exercise(exercise)
    if exercise_module:
        return exercise_module.calculate_form(landmarks_list, validation_result)
    elif exercise == 2:
        return {"exercise": 2, "message": "Exercise 2 not implemented"}
    elif exercise == 3:
        return {"exercise": 3, "message": "Exercise 3 not implemented"}
    else:
        raise ValueError(f"Invalid exercise: {exercise}")


def validate_exercise_type(exercise: int) -> tuple:
    """
    General exercise type validator - returns validation result without raising exceptions.
    Usable by both upload and livestream.
    Returns tuple of (is_valid: bool, error_message: str).
    """
    if exercise not in [1, 2, 3]:
        return False, f"Invalid exercise type: {exercise}. Must be 1 (Squat), 2 (Bench), or 3 (Deadlift)"
    return True, None


def _validate_exercise(exercise: int) -> None:
    """Upload-specific wrapper - validates exercise type and raises HTTPException if invalid. Maintains backward compatibility."""
    is_valid, error_message = validate_exercise_type(exercise)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_message)


async def _validate_file(video: UploadFile, file_size: int = None) -> tuple:
    """Validates video file and returns file info and size."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    file_info = accept_video_file(video)
    
    if file_size is None:
        raise ValueError("File size must be provided")
    
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file received")
    
    MAX_FILE_SIZE = 250 * 1024 * 1024
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {file_size} bytes. Maximum: {MAX_FILE_SIZE} bytes")
    
    if file_size < 100:
        raise HTTPException(status_code=400, detail="File appears to be too small to be a valid video")
    
    return file_info, file_size


def check_camera_angle(landmarks_list: list, exercise: int = 1) -> dict:
    """
    General camera angle checker - returns camera angle info without raising exceptions.
    Usable by both upload and livestream. Check 'should_reject' flag to handle rejection.
    """
    from src.shared.exercises import get_exercise
    exercise_module = get_exercise(exercise)
    if exercise_module and exercise_module.get_camera_angle_detector():
        return exercise_module.get_camera_angle_detector()(landmarks_list)
    # Fallback for exercises without camera angle detection
    return {"should_reject": False, "angle_estimate": None, "message": "Camera angle detection not available"}


def _check_camera_angle(landmarks_list: list, exercise: int = 1) -> dict:
    """Upload-specific wrapper - checks camera angle and raises HTTPException if too extreme. Maintains backward compatibility."""
    camera_angle_info = check_camera_angle(landmarks_list, exercise)
    if camera_angle_info.get("should_reject", False):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "camera_angle_too_extreme",
                "message": camera_angle_info["message"],
                "angle_estimate": camera_angle_info["angle_estimate"],
                "recommendation": "Please record again with the person standing perpendicular to the camera for accurate measurements."
            }
        )
    return camera_angle_info


def _extract_active_angles(angles: list, phases: dict) -> list:
    """Extracts angles only from active exercise phases (e.g., squat phases, bench reps, etc.)."""
    if not phases or not phases.get("reps"):
        return angles
    active_angles = []
    for rep in phases["reps"]:
        active_angles.extend([angles[i] if i < len(angles) else None
                             for i in range(rep["start_frame"], rep["end_frame"] + 1)])
    return active_angles


def _perform_angle_analyses(calculation_results: dict, quad_angles: list, ankle_angles: list,
                           phases: dict, torso_asymmetry: list, quad_asymmetry: list,
                           ankle_asymmetry: list, fps: float, camera_angle_info: dict = None,
                           landmarks_list: list = None, validation_result: dict = None) -> dict:
    """
    Performs all angle analyses and returns form_analysis dict.
    NOTE: This function is kept for backward compatibility but is now deprecated.
    New exercises should use the ExerciseBase.analyze_form() method instead.
    """
    from src.exercise_1.llm_form_analysis.llm_form_analysis import (
        analyze_torso_angle, analyze_quad_angle, analyze_ankle_angle, analyze_asymmetry,
        analyze_rep_consistency, analyze_glute_dominance, analyze_knee_valgus, _is_front_view
    )
    quad_angles_raw = calculation_results["angles_per_frame"].get("quad_angle", [])
    torso_angles_raw = calculation_results["angles_per_frame"].get("torso_angle", [])
    torso_analysis = analyze_torso_angle(torso_angles_raw, quad_angles_raw, validation_result)
    quad_analysis = analyze_quad_angle(quad_angles)
    ankle_analysis = analyze_ankle_angle(ankle_angles)
    torso_asymmetry_analysis = analyze_asymmetry(torso_asymmetry, "torso")
    quad_asymmetry_analysis = analyze_asymmetry(quad_asymmetry, "quad")
    ankle_asymmetry_analysis = analyze_asymmetry(ankle_asymmetry, "ankle")
    asymmetry_data = calculation_results.get("asymmetry_per_frame", {})
    rep_consistency = analyze_rep_consistency(
        calculation_results["angles_per_frame"], asymmetry_data, phases.get("reps", [])
    ) if phases and phases.get("reps") else None
    glute_dominance = analyze_glute_dominance(
        quad_angles_raw, torso_angles_raw, phases.get("reps", []), fps
    ) if phases and phases.get("reps") else None
    knee_valgus = None
    if _is_front_view(camera_angle_info) and landmarks_list and phases and phases.get("reps"):
        knee_valgus = analyze_knee_valgus(landmarks_list, phases.get("reps", []))
    result = {
        "torso_angle": torso_analysis, "quad_angle": quad_analysis, "ankle_angle": ankle_analysis,
        "torso_asymmetry": torso_asymmetry_analysis, "quad_asymmetry": quad_asymmetry_analysis,
        "ankle_asymmetry": ankle_asymmetry_analysis
    }
    if rep_consistency:
        result["rep_consistency"] = rep_consistency
    if glute_dominance and glute_dominance.get("status") != "error":
        result["glute_dominance"] = glute_dominance
    if knee_valgus and knee_valgus.get("status") != "error":
        result["knee_valgus"] = knee_valgus
    from src.exercise_1.llm_form_analysis.llm_form_analysis import calculate_final_score
    result["final_score"] = calculate_final_score(result)
    return result


def _extract_all_active_data(calculation_results: dict, phases: dict) -> tuple:
    """
    Extracts all active angles and asymmetry data from exercise phases.
    NOTE: This function is kept for backward compatibility but is now deprecated.
    New exercises should use ExerciseBase.extract_active_data() method instead.
    """
    quad_angles_raw = calculation_results["angles_per_frame"].get("quad_angle", [])
    quad_angles = _extract_active_angles(quad_angles_raw, phases)
    ankle_angles = _extract_active_angles(
        calculation_results["angles_per_frame"].get("ankle_angle", []), phases
    )
    asymmetry_data = calculation_results.get("asymmetry_per_frame", {})
    torso_asymmetry = _extract_active_angles(asymmetry_data.get("torso_asymmetry", []), phases)
    quad_asymmetry = _extract_active_angles(asymmetry_data.get("quad_asymmetry", []), phases)
    ankle_asymmetry = _extract_active_angles(asymmetry_data.get("ankle_asymmetry", []), phases)
    return quad_angles, ankle_angles, torso_asymmetry, quad_asymmetry, ankle_asymmetry


def _analyze_exercise_form(exercise: int, calculation_results: dict, fps: float,
                          camera_angle_info: dict = None, landmarks_list: list = None, validation_result: dict = None) -> tuple:
    """
    Analyzes exercise form and returns form_analysis and phases (e.g., squat_phases).
    Uses the exercise registry to route to the appropriate exercise implementation.
    """
    from src.shared.exercises import get_exercise
    exercise_module = get_exercise(exercise)
    
    form_analysis = None
    phases = None
    
    if exercise_module and calculation_results.get("angles_per_frame"):
        # Detect phases (reps, etc.)
        phases = exercise_module.detect_phases(calculation_results, fps)
        
        # Perform form analysis
        form_analysis = exercise_module.analyze_form(
            calculation_results, phases, fps, camera_angle_info, landmarks_list, validation_result
        )
    
    return form_analysis, phases


def process_analysis_pipeline(exercise: int, frames: list = None, fps: float = 30.0, landmarks_list: list = None, validation_result: dict = None) -> tuple:
    """Core analysis pipeline - processes frames and returns analysis results. General function usable by both upload and livestream.
    Note: frames parameter is kept for backward compatibility but is not actually used - only landmarks_list is needed.
    Returns: (calculation_results, camera_angle_info, form_analysis, phases)
    Note: phases is returned as squat_phases in response for backward compatibility.
    """
    camera_angle_info = check_camera_angle(landmarks_list, exercise)
    calculation_results = route_to_exercise_calculation(exercise, landmarks_list, validation_result)
    form_analysis, phases = _analyze_exercise_form(
        exercise, calculation_results, fps, camera_angle_info, landmarks_list, validation_result
    )
    return calculation_results, camera_angle_info, form_analysis, phases


def _process_video_analysis(video: UploadFile, exercise: int, frames: list, fps: float, landmarks_list: list, validation_result: dict = None) -> tuple:
    """Upload-specific wrapper for process_analysis_pipeline. Handles camera angle rejection for upload. Maintains backward compatibility."""
    calc_results, cam_info, form_analysis, phases = process_analysis_pipeline(exercise, frames, fps, landmarks_list, validation_result)
    if cam_info.get("should_reject", False):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "camera_angle_too_extreme",
                "message": cam_info["message"],
                "angle_estimate": cam_info["angle_estimate"],
                "recommendation": "Please record again with the person standing perpendicular to the camera for accurate measurements."
            }
        )
    return calc_results, cam_info, form_analysis, phases


def build_analysis_response(exercise: int, frame_count: int, calculation_results: dict, camera_angle_info: dict,
                            form_analysis: dict, phases: dict) -> dict:
    """
    Builds general analysis response dictionary. Usable by both upload and livestream.
    Note: phases is stored as 'squat_phases' in response for backward compatibility.
    """
    from src.shared.exercises import get_exercise
    exercise_module = get_exercise(exercise)
    exercise_name = exercise_module.exercise_name if exercise_module else {1: "Squat", 2: "Bench", 3: "Deadlift"}.get(exercise, "Unknown")
    
    return {
        "status": "success",
        "exercise": exercise,
        "exercise_name": exercise_name,
        "frame_count": frame_count,
        "calculation_results": calculation_results,
        "camera_angle_info": camera_angle_info,
        "form_analysis": form_analysis,
        "squat_phases": phases,  # Keep 'squat_phases' key for backward compatibility
        "validated": True
    }


def _build_response(exercise: int, file_info: dict, file_size: int, frame_count: int, output_path: Path,
                   output_filename: str, calculation_results: dict, camera_angle_info: dict,
                   form_analysis: dict, phases: dict, visualization_url: str = None) -> dict:
    """Upload-specific response builder. Adds upload metadata to general analysis response."""
    analysis_response = build_analysis_response(
        exercise, frame_count, calculation_results, camera_angle_info, form_analysis, phases
    )
    if visualization_url is None:
        # visualization_url should always be provided by the caller
        visualization_url = ""
    analysis_response.update({
        "message": "Video processed successfully",
        "filename": file_info["filename"],
        "content_type": file_info["content_type"],
        "size": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "visualization_path": str(output_path),
        "visualization_url": visualization_url
    })
    return analysis_response


def create_visualization_streaming(video_path: str, landmarks_list: list, fps: float,
                                   calculation_results: dict = None, form_analysis: dict = None,
                                   output_dir: Path = None, output_filename: str = None,
                                   frame_skip: int = 1) -> tuple:
    """
    Streaming version: Creates visualization by reading frames from video file.
    Never keeps all frames in memory - processes one frame at a time.
    """
    from src.shared.pose_estimation.pose_estimation import create_visualization_streaming as viz_stream
    import uuid
    
    landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30, 31, 32]
    
    per_frame_status = None
    if calculation_results and form_analysis:
        from src.shared.visualization.per_frame_status import calculate_per_frame_status, smooth_per_frame_status
        glute_dominance_status = None
        if form_analysis.get("glute_dominance"):
            glute_dominance_status = form_analysis["glute_dominance"].get("status")
        
        per_frame_status = calculate_per_frame_status(
            angles_per_frame=calculation_results.get("angles_per_frame", {}),
            asymmetry_per_frame=calculation_results.get("asymmetry_per_frame"),
            glute_dominance_status=glute_dominance_status
        )
        
        per_frame_status = smooth_per_frame_status(per_frame_status, fps, window_duration_seconds=0.2)
    
    if output_dir is None:
        output_dir = OUTPUTS_DIR
    if output_filename is None:
        video_id = str(uuid.uuid4())
        output_filename = f"{video_id}.mp4"  # Use .mp4 with H.264 (browser-compatible)
    else:
        base_name = os.path.splitext(output_filename)[0]
        output_filename = f"{base_name}.mp4"
    
    output_path = output_dir / output_filename
    
    actual_output_path = viz_stream(str(video_path), landmarks_list, str(output_path), landmark_indices, 
                                    per_frame_status, fps, frame_skip)
    
    if actual_output_path != str(output_path):
        output_filename = os.path.basename(actual_output_path)
    
    return actual_output_path, output_filename


def _cleanup_old_outputs(outputs_dir: Path) -> None:
    """Cleans up old output video files when a new upload is processed."""
    try:
        if not outputs_dir.exists():
            return
        
        for file_path in outputs_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.mp4', '.avi', '.mov']:
                try:
                    file_path.unlink()
                except Exception:
                    pass
    except Exception:
        pass


def _handle_upload_errors(e: Exception):
    """Handles upload errors and raises appropriate HTTPException."""
    if isinstance(e, HTTPException):
        raise e
    elif isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    else:
        import logging
        logging.error(f"Unexpected error during video upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": "An unexpected error occurred while processing your video. Please try again later."
        })


def _get_client_ip(request: Request) -> str:
    """Extracts client IP address, considering common proxy headers."""
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    client = request.client
    if client and client.host:
        return client.host
    return "unknown"


def validate_video_data(frames: list, fps: float, landmarks_list: list, validation_result: dict, exercise: int, 
                       skip_file_validations: bool = False) -> dict:
    """
    General validation orchestrator - validates frames, fps, duration, and landmarks.
    Usable by both upload and livestream (livestream can skip file validations).
    Returns dict with validation results or raises HTTPException on failure.
    """
    validation_results = {"all_valid": True, "errors": []}
    
    # Frame validation (if frames were extracted with validation)
    if frames:
        frame_count = len(frames)
        if frame_count == 0:
            raise HTTPException(status_code=400, detail={
                "error": "frame_extraction_failed",
                "message": "No frames extracted from video",
                "frame_count": 0,
                "valid_frame_count": 0,
                "recommendation": "Please try re-exporting the video"
            })
    else:
        frame_count = 0
    
    # FPS validation
    from src.shared.upload_video.video_validation import validate_fps
    fps_validation = validate_fps(fps)
    if not fps_validation.get("is_valid", True):
        raise HTTPException(status_code=400, detail={
            "error": "fps_validation_failed",
            "message": fps_validation.get("errors", ["FPS validation failed"])[0],
            "fps": fps_validation.get("fps", 0),
            "warnings": fps_validation.get("warnings", []),
            "recommendation": fps_validation.get("recommendation", "Please use a video with valid FPS metadata")
        })
    
    # Duration validation (only if frames available)
    if not skip_file_validations and frame_count > 0:
        from src.shared.upload_video.video_validation import validate_video_duration
        duration_validation = validate_video_duration(frame_count, fps, max_duration_seconds=120.0)
        if not duration_validation.get("is_valid", True):
            raise HTTPException(status_code=400, detail={
                "error": "video_duration_exceeded",
                "message": duration_validation.get("errors", ["Video duration validation failed"])[0],
                "duration_seconds": duration_validation.get("duration_seconds", 0),
                "frame_count": duration_validation.get("frame_count", 0),
                "fps": duration_validation.get("fps", 0),
                "recommendation": duration_validation.get("recommendation", "Please select a video shorter than 120 seconds")
            })
    
    # Landmark validation
    if validation_result and not validation_result.get("overall_valid", True):
        raise HTTPException(status_code=400, detail={
            "error": "insufficient_pose_detection",
            "message": validation_result.get("errors", ["Insufficient pose detection"])[0],
            "valid_frame_percentage": validation_result.get("valid_frame_percentage", 0.0),
            "recommendation": validation_result.get("recommendation", "Ensure person is fully visible")
        })
    
    validation_results["all_valid"] = True
    return validation_results


async def validate_uploaded_file(temp_path: str, video: UploadFile, file_size: int) -> dict:
    """
    Upload-specific file validation - validates file headers, content, and format.
    Returns file_info dict or raises HTTPException on failure.
    """
    file_info, _ = await _validate_file(video, file_size)
    from src.shared.upload_video.video_validation import validate_file_headers, validate_video_format
    header_validation = validate_file_headers(temp_path)
    if not header_validation.get("is_valid", False):
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail={
            "error": "invalid_file_headers",
            "message": header_validation.get("errors", ["Invalid file headers"])[0],
            "detected_format": header_validation.get("detected_format", None),
            "recommendation": header_validation.get("recommendation", "Please upload a valid video file")
        })
    from src.shared.upload_video.video_validation import validate_file_content
    content_validation = validate_file_content(temp_path)
    if not content_validation.get("is_valid", False):
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail={
            "error": "invalid_file_content",
            "message": content_validation.get("errors", ["File content validation failed"])[0],
            "warnings": content_validation.get("warnings", []),
            "fps": content_validation.get("fps", 0),
            "frame_count": content_validation.get("frame_count", 0),
            "recommendation": content_validation.get("recommendation", "Please upload a valid video file")
        })
    format_validation = validate_video_format(temp_path)
    if not format_validation.get("is_valid", False):
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=400, detail={
            "error": "video_format_unsupported",
            "message": format_validation.get("errors", ["Video format not supported"])[0],
            "codec": format_validation.get("codec", "unknown"),
            "recommendation": format_validation.get("recommendation", "Please convert to MP4 (H.264) format")
        })
    return file_info


def _check_rate_limit(client_ip: str) -> None:
    """Enforces simple IP-based rate limiting for uploads."""
    # Skip rate limiting when running locally (not on Heroku)
    if not os.environ.get("DYNO"):
        return
    
    now = time.time()
    with _upload_rate_limit_lock:
        request_times = _upload_rate_limit_store.setdefault(client_ip, deque())
        while request_times and request_times[0] <= now - RATE_LIMIT_WINDOW_SECONDS:
            request_times.popleft()
        if not request_times:
            _upload_rate_limit_store.pop(client_ip, None)
            request_times = _upload_rate_limit_store.setdefault(client_ip, deque())
        if len(request_times) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(request_times[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many uploads from this IP. Limit: {RATE_LIMIT_MAX_REQUESTS} per {RATE_LIMIT_WINDOW_SECONDS}s.",
                    "retry_after_seconds": max(retry_after, 1)
                }
            )
        request_times.append(now)
        if not request_times:
            del _upload_rate_limit_store[client_ip]


@app.post("/upload-video")
async def upload_video(
    request: Request,
    video: UploadFile = File(...),
    exercise: int = Form(...),
    # Optional: check for Authorization header to apply token limits for logged-in users
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Accepts video file upload and processes with pose estimation using streaming to minimize memory."""
    temp_path = None
    # Initialize token variables for signed-in users (must be initialized before try block)
    tokens_used = None
    tokens_remaining = None
    
    try:
        # Check if user is authenticated and apply token system
        user_id = None
        if credentials:
            from src.shared.auth.auth import verify_token
            payload = verify_token(credentials.credentials)
            if payload:
                user_id = payload.get("sub")
        else:
            pass  # Anonymous user
        
        client_ip = _get_client_ip(request)
        _check_rate_limit(client_ip)
        _validate_exercise(exercise)
        
        # Check authentication status
        is_authenticated = credentials is not None and user_id is not None
        
        # For logged-in users: check if they have at least 1 token BEFORE upload (base cost is always 1)
        if is_authenticated:
            # Fetch user to check token count using new transaction system
            from src.shared.auth.database import User
            from src.shared.payment.token_utils import calculate_token_balance, has_sufficient_tokens
            db = next(get_db())
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    db.close()
                    raise HTTPException(
                        status_code=401,
                        detail="User not found"
                    )
                # Check token balance using transaction system
                if not has_sufficient_tokens(db, user_id, 1):
                    balance = calculate_token_balance(db, user_id)
                    db.close()
                    raise HTTPException(
                        status_code=402,
                        detail={
                            "error": "insufficient_tokens",
                            "message": f"Insufficient tokens. You need at least 1 token but only have {balance.total} remaining.",
                            "tokens_required": 1,
                            "tokens_remaining": balance.total,
                            "free_tokens": balance.free_tokens,
                            "purchased_tokens": balance.purchased_tokens
                        }
                    )
            except HTTPException:
                raise
            except Exception as e:
                import logging
                logging.warning(f"Failed to check user tokens: {str(e)}")
                db.rollback()
            finally:
                db.close()
        
        # For logged-out users: check IP limit BEFORE upload (efficient early rejection)
        # Skip anonymous limit check when running locally (not on Heroku)
        if not is_authenticated and os.environ.get("DYNO"):
            # Check if IP has already completed 1 analysis today (before uploading file)
            db = next(get_db())
            try:
                # Ensure table exists (fallback if startup init failed)
                from src.shared.auth.database import Base, engine
                Base.metadata.create_all(bind=engine, checkfirst=True)
                
                anonymous_record = db.query(AnonymousAnalysis).filter(
                    AnonymousAnalysis.ip_address == client_ip
                ).first()
                
                # Reset daily limit if it's a new day
                if anonymous_record:
                    reset_daily_anonymous_limit_if_needed(anonymous_record)
                    db.commit()
                    # Re-query to get fresh state after reset
                    anonymous_record = db.query(AnonymousAnalysis).filter(
                        AnonymousAnalysis.ip_address == client_ip
                    ).first()
                
                if anonymous_record and anonymous_record.analysis_count >= 1:
                    db.close()
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "analysis_limit_reached",
                            "message": "Anonymous users are limited to 1 analysis per day. Please sign up for unlimited analyses.",
                            "analyses_completed": anonymous_record.analysis_count,
                            "limit": 1
                        }
                    )
            except HTTPException:
                # Re-raise HTTP exceptions (like the 403 above)
                raise
            except Exception as e:
                # Log database errors but allow request to continue
                # This prevents database issues from blocking all anonymous users
                import logging
                logging.warning(f"Failed to check anonymous analysis limit: {str(e)}")
                db.rollback()
            finally:
                db.close()
        
        # Now upload the file
        temp_path = await save_video_temp(video)
        file_size = os.path.getsize(temp_path)
        
        # Check file size limit for anonymous users (after upload but before processing)
        if not is_authenticated:
            MAX_FILE_SIZE_ANONYMOUS = 100 * 1024 * 1024  # 100MB
            if file_size >= MAX_FILE_SIZE_ANONYMOUS:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "file_too_large_anonymous",
                        "message": "File size exceeds 100MB limit for anonymous users. Please sign up for larger file support.",
                        "max_size_mb": 100,
                        "file_size_mb": round(file_size / (1024 * 1024), 2)
                    }
                )
        
        # Full token check for logged-in users (after we know file size)
        if is_authenticated:
            # Re-fetch user from DB in this session to get latest token count
            # (user object from earlier session is detached, so we need to query again)
            db = next(get_db())
            try:
                from src.shared.auth.database import User
                current_user = db.query(User).filter(User.id == user_id).first()
                if not current_user:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    db.close()
                    raise HTTPException(
                        status_code=401,
                        detail="User not found"
                    )
                # Check tokens using new transaction system
                from src.shared.payment.token_utils import calculate_token_balance, has_sufficient_tokens
                token_cost = calculate_token_cost(file_size)
                token_cost_int = int(token_cost) + (1 if token_cost % 1 > 0 else 0)
                
                if not has_sufficient_tokens(db, user_id, token_cost_int):
                    balance = calculate_token_balance(db, user_id)
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    db.close()
                    raise HTTPException(
                        status_code=402,  # Payment Required
                        detail={
                            "error": "insufficient_tokens",
                            "message": f"Insufficient tokens. You need {token_cost:.1f} tokens but only have {balance.total} remaining.",
                            "tokens_required": round(token_cost, 1),
                            "tokens_remaining": balance.total,
                            "free_tokens": balance.free_tokens,
                            "purchased_tokens": balance.purchased_tokens
                        }
                    )
            except HTTPException:
                raise
            except Exception as e:
                import logging
                logging.error(f"Failed to check user tokens after file upload: {str(e)}")
                db.rollback()
            finally:
                db.close()
        
        file_info = await validate_uploaded_file(temp_path, video, file_size)
        
        import cv2
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail={
                "error": "video_open_failed",
                "message": "Could not open video file for processing"
            })
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_from_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        
        MAX_FRAMES = 300
        frame_skip = 1
        if total_frames > MAX_FRAMES:
            frame_skip = max(1, total_frames // MAX_FRAMES)
        
        from src.shared.pose_estimation.pose_estimation import process_video_streaming_pose
        required_landmarks = get_required_landmarks(exercise)
        landmarks_list, fps, frame_count, validation_result = process_video_streaming_pose(
            temp_path, validate=True, required_landmarks=required_landmarks, frame_skip=frame_skip
        )
        
        if validation_result and not validation_result.get("is_valid", True):
            raise HTTPException(status_code=400, detail={
                "error": "landmark_validation_failed",
                "message": validation_result.get("errors", ["Landmark validation failed"])[0],
                "frame_count": frame_count,
                "valid_landmark_count": validation_result.get("valid_landmark_count", 0),
                "recommendation": validation_result.get("recommendation", "Please ensure person is fully visible in video")
            })
        
        from src.shared.upload_video.video_validation import validate_fps, validate_video_duration
        fps_validation = validate_fps(fps)
        if not fps_validation.get("is_valid", True):
            raise HTTPException(status_code=400, detail={
                "error": "fps_validation_failed",
                "message": fps_validation.get("errors", ["FPS validation failed"])[0]
            })
        duration_validation = validate_video_duration(frame_count, fps, max_duration_seconds=120.0)
        if not duration_validation.get("is_valid", True):
            raise HTTPException(status_code=400, detail={
                "error": "video_duration_exceeded",
                "message": duration_validation.get("errors", ["Video duration validation failed"])[0]
            })
        
        calc_results, cam_info, form_analysis, phases = _process_video_analysis(
            video, exercise, None, fps, landmarks_list, validation_result
        )
        
        _cleanup_old_outputs(OUTPUTS_DIR)
        
        output_path, output_filename = create_visualization_streaming(
            temp_path, landmarks_list, fps, calc_results, form_analysis,
            OUTPUTS_DIR, None, frame_skip
        )
        
        # Verify file exists before constructing URL
        if not os.path.exists(output_path):
            visualization_url = None
        else:
            # Construct URL properly for both local and Heroku
            # On Heroku, use X-Forwarded-Proto and Host headers
            # On localhost, use request.base_url directly
            if os.environ.get("DYNO"):  # Heroku
                scheme = request.headers.get("X-Forwarded-Proto", "https")
                host = request.headers.get("Host", request.url.hostname)
                visualization_url = f"{scheme}://{host}/outputs/{output_filename}"
            else:  # Localhost
                base_url = str(request.base_url).rstrip('/')
                visualization_url = f"{base_url}/outputs/{output_filename}"
        
        # Track anonymous analysis or deduct tokens for logged-in users after successful analysis
        # Only track/increment AFTER successful completion (we're past all error points here)
        tokens_used = None
        tokens_remaining = None
        
        # Only track anonymous analysis if user is truly not authenticated
        # Use user_id (set at the start) instead of user object which may not exist here
        is_authenticated = credentials is not None and user_id is not None
        # Skip tracking anonymous analysis when running locally (not on Heroku)
        if not is_authenticated and os.environ.get("DYNO"):
            # Track anonymous analysis by IP (reset daily limit if new day before incrementing)
            db = next(get_db())
            try:
                anonymous_record = db.query(AnonymousAnalysis).filter(
                    AnonymousAnalysis.ip_address == client_ip
                ).first()
                
                # Reset daily limit if it's a new day (before incrementing)
                if anonymous_record:
                    reset_daily_anonymous_limit_if_needed(anonymous_record)
                    anonymous_record.analysis_count += 1
                    anonymous_record.last_analysis_at = datetime.utcnow()
                else:
                    anonymous_record = AnonymousAnalysis(
                        ip_address=client_ip,
                        analysis_count=1,
                        first_analysis_at=datetime.utcnow(),
                        last_analysis_at=datetime.utcnow()
                    )
                    db.add(anonymous_record)
                
                db.commit()
            except Exception as e:
                db.rollback()
                # Log error but don't fail the request
                import logging
                logging.error(f"Failed to track anonymous analysis: {str(e)}")
            finally:
                db.close()
        else:
            # Deduct tokens for logged-in users
            token_cost = calculate_token_cost(file_size)
            db = next(get_db())
            try:
                # Re-fetch user from DB in this session to get latest token count
                # (user object from earlier session is detached, so we need to query again using user_id)
                from src.shared.auth.database import User
                current_user = db.query(User).filter(User.id == user_id).first()
                if not current_user:
                    tokens_used = None
                    tokens_remaining = None
                else:
                    # Deduct tokens using the new transaction system
                    from src.shared.payment.token_utils import deduct_tokens, calculate_token_balance
                    
                    # Deduct tokens (round up to nearest integer)
                    token_cost_int = int(token_cost) + (1 if token_cost % 1 > 0 else 0)
                    success = deduct_tokens(
                        db=db,
                        user_id=user_id,
                        amount=token_cost_int,
                        source='analysis_usage',
                        metadata={'file_size_bytes': file_size, 'exercise': exercise, 'calculated_cost': token_cost}
                    )
                    
                    if not success:
                        # This shouldn't happen since we checked earlier, but handle it
                        db.rollback()
                        import logging
                        logging.error(f"Failed to deduct tokens for user {user_id}")
                        tokens_used = None
                        tokens_remaining = None
                    else:
                        # Commit the token deduction
                        db.commit()
                        
                        # Get updated balance
                        balance = calculate_token_balance(db, user_id)
                        tokens_used = round(token_cost, 1)
                        tokens_remaining = balance.total
            except Exception as e:
                db.rollback()
                import logging
                logging.error(f"Failed to deduct tokens: {str(e)}")
                # Set tokens to None on error so response doesn't include invalid data
                tokens_used = None
                tokens_remaining = None
            finally:
                db.close()
        
        response = _build_response(exercise, file_info, file_size, frame_count, Path(output_path),
                                  output_filename, calc_results, cam_info, form_analysis, phases, visualization_url)
        
        # Add token information to response if user is logged in
        if is_authenticated:
            if tokens_used is not None and tokens_remaining is not None:
                response["tokens_used"] = tokens_used
                response["tokens_remaining"] = tokens_remaining
        
        return response
    except Exception as e:
        _handle_upload_errors(e)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

