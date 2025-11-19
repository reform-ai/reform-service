"""Reform Service - FastAPI server for exercise form analysis."""

import os
import tempfile
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Depends
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from collections import deque
import time
from datetime import datetime
from threading import Lock
from src.shared.upload_video.upload_video import (
    accept_video_file,
    save_video_temp
)
from src.exercise_1.calculation.calculation import calculate_squat_form
from src.shared.auth.database import init_db, get_db, reset_daily_tokens_if_needed, calculate_token_cost, AnonymousAnalysis
from src.shared.auth.routes import router as auth_router
from src.shared.auth.dependencies import security
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
        logging.info("Database initialization completed on startup")
    except Exception as e:
        # Log error but don't crash the app
        # Database will be created on first use if needed
        logging.error(f"Database initialization error on startup: {str(e)}")
        # Try to continue - database operations will handle errors gracefully
        # Tables will be created on first auth request as fallback

# Include auth routes
app.include_router(auth_router)

# Use /tmp/outputs on Heroku (ephemeral filesystem), otherwise use local outputs directory
if os.environ.get("DYNO"):  # Heroku sets DYNO environment variable
    OUTPUTS_DIR = Path("/tmp/outputs")
else:
    OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://reformgym.fit",
        "https://reform-client-c95dd550c494.herokuapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
    # Check anonymous limit
    client_ip = _get_client_ip(request)
    db = next(get_db())
    try:
        from src.shared.auth.database import Base, engine
        Base.metadata.create_all(bind=engine, checkfirst=True)
        
        anonymous_record = db.query(AnonymousAnalysis).filter(
            AnonymousAnalysis.ip_address == client_ip
        ).first()
        
        if anonymous_record and anonymous_record.analysis_count >= 1:
            return {
                "has_limit": True,
                "limit_reached": True,
                "analyses_completed": anonymous_record.analysis_count,
                "limit": 1,
                "message": "Anonymous users are limited to 1 analysis. Please sign up for unlimited analyses."
            }
        else:
            return {
                "has_limit": True,
                "limit_reached": False,
                "analyses_completed": anonymous_record.analysis_count if anonymous_record else 0,
                "limit": 1,
                "message": "You have 1 free analysis remaining"
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
    if exercise == 1:
        from src.exercise_1.calculation.landmark_validation import get_squat_required_landmarks
        return get_squat_required_landmarks()
    elif exercise == 2:
        # TODO: Add bench required landmarks when implemented
        return None
    elif exercise == 3:
        # TODO: Add deadlift required landmarks when implemented
        return None
    else:
        return None


def route_to_exercise_calculation(exercise: int, landmarks_list: list, validation_result: dict = None) -> dict:
    """Routes to appropriate exercise calculation module."""
    if exercise == 1:
        return calculate_squat_form(landmarks_list, validation_result)
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
    
    MAX_FILE_SIZE = 500 * 1024 * 1024
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {file_size} bytes. Maximum: {MAX_FILE_SIZE} bytes")
    
    if file_size < 100:
        raise HTTPException(status_code=400, detail="File appears to be too small to be a valid video")
    
    return file_info, file_size


def check_camera_angle(landmarks_list: list) -> dict:
    """
    General camera angle checker - returns camera angle info without raising exceptions.
    Usable by both upload and livestream. Check 'should_reject' flag to handle rejection.
    """
    from src.exercise_1.calculation.calculation import detect_camera_angle
    return detect_camera_angle(landmarks_list)


def _check_camera_angle(landmarks_list: list) -> dict:
    """Upload-specific wrapper - checks camera angle and raises HTTPException if too extreme. Maintains backward compatibility."""
    camera_angle_info = check_camera_angle(landmarks_list)
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


def _extract_active_angles(angles: list, squat_phases: dict) -> list:
    """Extracts angles only from active squat phases."""
    if not squat_phases.get("reps"):
        return angles
    active_angles = []
    for rep in squat_phases["reps"]:
        active_angles.extend([angles[i] if i < len(angles) else None
                             for i in range(rep["start_frame"], rep["end_frame"] + 1)])
    return active_angles


def _perform_angle_analyses(calculation_results: dict, quad_angles: list, ankle_angles: list,
                           squat_phases: dict, torso_asymmetry: list, quad_asymmetry: list,
                           ankle_asymmetry: list, fps: float, camera_angle_info: dict = None,
                           landmarks_list: list = None, validation_result: dict = None) -> dict:
    """Performs all angle analyses and returns form_analysis dict."""
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
        calculation_results["angles_per_frame"], asymmetry_data, squat_phases.get("reps", [])
    ) if squat_phases and squat_phases.get("reps") else None
    glute_dominance = analyze_glute_dominance(
        quad_angles_raw, torso_angles_raw, squat_phases.get("reps", []), fps
    ) if squat_phases and squat_phases.get("reps") else None
    knee_valgus = None
    if _is_front_view(camera_angle_info) and landmarks_list and squat_phases and squat_phases.get("reps"):
        knee_valgus = analyze_knee_valgus(landmarks_list, squat_phases.get("reps", []))
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


def _extract_all_active_data(calculation_results: dict, squat_phases: dict) -> tuple:
    """Extracts all active angles and asymmetry data from squat phases."""
    quad_angles_raw = calculation_results["angles_per_frame"].get("quad_angle", [])
    quad_angles = _extract_active_angles(quad_angles_raw, squat_phases)
    ankle_angles = _extract_active_angles(
        calculation_results["angles_per_frame"].get("ankle_angle", []), squat_phases
    )
    asymmetry_data = calculation_results.get("asymmetry_per_frame", {})
    torso_asymmetry = _extract_active_angles(asymmetry_data.get("torso_asymmetry", []), squat_phases)
    quad_asymmetry = _extract_active_angles(asymmetry_data.get("quad_asymmetry", []), squat_phases)
    ankle_asymmetry = _extract_active_angles(asymmetry_data.get("ankle_asymmetry", []), squat_phases)
    return quad_angles, ankle_angles, torso_asymmetry, quad_asymmetry, ankle_asymmetry


def _analyze_exercise_form(exercise: int, calculation_results: dict, fps: float,
                          camera_angle_info: dict = None, landmarks_list: list = None, validation_result: dict = None) -> tuple:
    """Analyzes exercise form and returns form_analysis and squat_phases."""
    form_analysis = None
    squat_phases = None
    if exercise == 1 and calculation_results.get("angles_per_frame"):
        from src.exercise_1.llm_form_analysis.llm_form_analysis import detect_squat_phases
        quad_angles_raw = calculation_results["angles_per_frame"].get("quad_angle", [])
        squat_phases = detect_squat_phases(quad_angles_raw, fps)
        quad_angles, ankle_angles, torso_asymmetry, quad_asymmetry, ankle_asymmetry = _extract_all_active_data(
            calculation_results, squat_phases
        )
        form_analysis = _perform_angle_analyses(
            calculation_results, quad_angles, ankle_angles, squat_phases,
            torso_asymmetry, quad_asymmetry, ankle_asymmetry, fps, camera_angle_info, landmarks_list, validation_result
        )
    return form_analysis, squat_phases


def process_analysis_pipeline(exercise: int, frames: list = None, fps: float = 30.0, landmarks_list: list = None, validation_result: dict = None) -> tuple:
    """Core analysis pipeline - processes frames and returns analysis results. General function usable by both upload and livestream.
    Note: frames parameter is kept for backward compatibility but is not actually used - only landmarks_list is needed.
    """
    camera_angle_info = check_camera_angle(landmarks_list)
    calculation_results = route_to_exercise_calculation(exercise, landmarks_list, validation_result)
    form_analysis, squat_phases = _analyze_exercise_form(
        exercise, calculation_results, fps, camera_angle_info, landmarks_list, validation_result
    )
    return calculation_results, camera_angle_info, form_analysis, squat_phases


def _process_video_analysis(video: UploadFile, exercise: int, frames: list, fps: float, landmarks_list: list, validation_result: dict = None) -> tuple:
    """Upload-specific wrapper for process_analysis_pipeline. Handles camera angle rejection for upload. Maintains backward compatibility."""
    calc_results, cam_info, form_analysis, squat_phases = process_analysis_pipeline(exercise, frames, fps, landmarks_list, validation_result)
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
    return calc_results, cam_info, form_analysis, squat_phases


def build_analysis_response(exercise: int, frame_count: int, calculation_results: dict, camera_angle_info: dict,
                            form_analysis: dict, squat_phases: dict) -> dict:
    """Builds general analysis response dictionary. Usable by both upload and livestream."""
    exercise_names = {1: "Squat", 2: "Bench", 3: "Deadlift"}
    return {
        "status": "success",
        "exercise": exercise,
        "exercise_name": exercise_names[exercise],
        "frame_count": frame_count,
        "calculation_results": calculation_results,
        "camera_angle_info": camera_angle_info,
        "form_analysis": form_analysis,
        "squat_phases": squat_phases,
        "validated": True
    }


def _build_response(exercise: int, file_info: dict, file_size: int, frame_count: int, output_path: Path,
                   output_filename: str, calculation_results: dict, camera_angle_info: dict,
                   form_analysis: dict, squat_phases: dict, visualization_url: str = None) -> dict:
    """Upload-specific response builder. Adds upload metadata to general analysis response."""
    analysis_response = build_analysis_response(
        exercise, frame_count, calculation_results, camera_angle_info, form_analysis, squat_phases
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
        raise HTTPException(status_code=500, detail={
            "error": "internal_server_error",
            "message": "An unexpected error occurred while processing your video"
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
    
    try:
        # Check if user is authenticated and apply token system
        user_id = None
        if credentials:
            from src.shared.auth.auth import verify_token
            payload = verify_token(credentials.credentials)
            if payload:
                user_id = payload.get("sub")
        
        client_ip = _get_client_ip(request)
        _check_rate_limit(client_ip)
        _validate_exercise(exercise)
        
        # Check authentication status
        is_authenticated = credentials is not None and user_id is not None
        
        # For logged-in users: check if they have at least 1 token BEFORE upload (base cost is always 1)
        if is_authenticated:
            import logging
            logging.info(f"[TOKEN_CHECK_1] Starting pre-upload token check for user_id: {user_id}")
            # Fetch user to check token count
            from src.shared.auth.database import User
            db = next(get_db())
            try:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    db.close()
                    raise HTTPException(
                        status_code=401,
                        detail="User not found"
                    )
                logging.info(f"[TOKEN_CHECK_1] User found, tokens_remaining: {user.tokens_remaining}")
                # Reset tokens if it's a new day
                tokens_before = user.tokens_remaining
                logging.info(f"[TOKEN_CHECK_1] Before reset_daily_tokens_if_needed, tokens_before: {tokens_before}")
                try:
                    reset_daily_tokens_if_needed(user, db)
                    logging.info(f"[TOKEN_CHECK_1] After reset_daily_tokens_if_needed, user.tokens_remaining: {user.tokens_remaining}")
                except Exception as e:
                    logging.error(f"[TOKEN_CHECK_1] ERROR in reset_daily_tokens_if_needed: {str(e)}")
                    if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                        logging.error(f"[TOKEN_CHECK_1] *** INVALID STATE ERROR DETECTED in reset_daily_tokens_if_needed ***")
                    raise
                # If tokens were reset, commit the change
                if tokens_before != user.tokens_remaining:
                    logging.info(f"[TOKEN_CHECK_1] Tokens were reset ({tokens_before} -> {user.tokens_remaining}), committing...")
                    db.commit()
                    logging.info(f"[TOKEN_CHECK_1] Commit successful, re-querying user...")
                    # Re-query to get fresh state after commit (refresh can cause invalid state error)
                    user = db.query(User).filter(User.id == user_id).first()
                    if user:
                        logging.info(f"[TOKEN_CHECK_1] Re-queried user, tokens_remaining: {user.tokens_remaining}")
                    else:
                        logging.error(f"[TOKEN_CHECK_1] User not found after re-query!")
                # Quick check: ensure user has at least 1 token before upload
                # Full check with file size will happen after upload
                try:
                    token_count = user.tokens_remaining
                    logging.info(f"[TOKEN_CHECK_1] Reading user.tokens_remaining: {token_count}")
                except Exception as e:
                    logging.error(f"[TOKEN_CHECK_1] *** ERROR reading user.tokens_remaining: {str(e)} ***")
                    if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                        logging.error(f"[TOKEN_CHECK_1] *** INVALID STATE ERROR DETECTED when reading tokens_remaining ***")
                    raise
                if token_count < 1:
                    db.close()
                    raise HTTPException(
                        status_code=402,
                        detail={
                            "error": "insufficient_tokens",
                            "message": f"Insufficient tokens. You need at least 1 token but only have {token_count} remaining.",
                            "tokens_required": 1,
                            "tokens_remaining": token_count,
                            "tokens_reset": "Daily tokens reset at midnight UTC"
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
        if not is_authenticated:
            # Check if IP has already completed 1 analysis (before uploading file)
            db = next(get_db())
            try:
                # Ensure table exists (fallback if startup init failed)
                from src.shared.auth.database import Base, engine
                Base.metadata.create_all(bind=engine, checkfirst=True)
                
                anonymous_record = db.query(AnonymousAnalysis).filter(
                    AnonymousAnalysis.ip_address == client_ip
                ).first()
                
                if anonymous_record and anonymous_record.analysis_count >= 1:
                    db.close()
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "analysis_limit_reached",
                            "message": "Anonymous users are limited to 1 analysis. Please sign up for unlimited analyses.",
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
            MAX_FILE_SIZE_ANONYMOUS = 200 * 1024 * 1024  # 200MB
            if file_size >= MAX_FILE_SIZE_ANONYMOUS:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "file_too_large_anonymous",
                        "message": "File size exceeds 200MB limit for anonymous users. Please sign up for larger file support.",
                        "max_size_mb": 200,
                        "file_size_mb": round(file_size / (1024 * 1024), 2)
                    }
                )
        
        # Full token check for logged-in users (after we know file size)
        if is_authenticated:
            import logging
            logging.info(f"[TOKEN_CHECK_2] Starting post-upload token check for user_id: {user_id}, file_size: {file_size}")
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
                logging.info(f"[TOKEN_CHECK_2] User found, tokens_remaining: {current_user.tokens_remaining}")
                # Reset tokens if it's a new day
                tokens_before = current_user.tokens_remaining
                logging.info(f"[TOKEN_CHECK_2] Before reset_daily_tokens_if_needed, tokens_before: {tokens_before}")
                try:
                    reset_daily_tokens_if_needed(current_user, db)
                    logging.info(f"[TOKEN_CHECK_2] After reset_daily_tokens_if_needed, current_user.tokens_remaining: {current_user.tokens_remaining}")
                except Exception as e:
                    logging.error(f"[TOKEN_CHECK_2] ERROR in reset_daily_tokens_if_needed: {str(e)}")
                    if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                        logging.error(f"[TOKEN_CHECK_2] *** INVALID STATE ERROR DETECTED in reset_daily_tokens_if_needed ***")
                    raise
                # If tokens were reset, commit the change
                if tokens_before != current_user.tokens_remaining:
                    logging.info(f"[TOKEN_CHECK_2] Tokens were reset ({tokens_before} -> {current_user.tokens_remaining}), committing...")
                    db.commit()
                    logging.info(f"[TOKEN_CHECK_2] Commit successful, re-querying user...")
                    # Re-query to get fresh state after commit (refresh can cause invalid state error)
                    current_user = db.query(User).filter(User.id == user_id).first()
                    if current_user:
                        logging.info(f"[TOKEN_CHECK_2] Re-queried user, tokens_remaining: {current_user.tokens_remaining}")
                    else:
                        logging.error(f"[TOKEN_CHECK_2] User not found after re-query!")
                token_cost = calculate_token_cost(file_size)
                logging.info(f"[TOKEN_CHECK_2] Token cost calculated: {token_cost}")
                try:
                    token_count_check = current_user.tokens_remaining
                    logging.info(f"[TOKEN_CHECK_2] Reading current_user.tokens_remaining for check: {token_count_check}")
                except Exception as e:
                    logging.error(f"[TOKEN_CHECK_2] *** ERROR reading current_user.tokens_remaining: {str(e)} ***")
                    if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                        logging.error(f"[TOKEN_CHECK_2] *** INVALID STATE ERROR DETECTED when reading tokens_remaining ***")
                    raise
                if token_count_check < token_cost:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    db.close()
                    raise HTTPException(
                        status_code=402,  # Payment Required
                        detail={
                            "error": "insufficient_tokens",
                            "message": f"Insufficient tokens. You need {token_cost:.1f} tokens but only have {token_count_check} remaining.",
                            "tokens_required": round(token_cost, 1),
                            "tokens_remaining": token_count_check,
                            "tokens_reset": "Daily tokens reset at midnight UTC"
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
        
        calc_results, cam_info, form_analysis, squat_phases = _process_video_analysis(
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
        if not is_authenticated:
            # Track anonymous analysis by IP
            db = next(get_db())
            try:
                anonymous_record = db.query(AnonymousAnalysis).filter(
                    AnonymousAnalysis.ip_address == client_ip
                ).first()
                
                if anonymous_record:
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
            import logging
            logging.info(f"[TOKEN_DEDUCTION] Starting token deduction for user_id: {user_id}, file_size: {file_size}")
            token_cost = calculate_token_cost(file_size)
            logging.info(f"[TOKEN_DEDUCTION] Token cost: {token_cost}")
            db = next(get_db())
            try:
                # Re-fetch user from DB in this session to get latest token count
                # (user object from earlier session is detached, so we need to query again using user_id)
                from src.shared.auth.database import User
                current_user = db.query(User).filter(User.id == user_id).first()
                if not current_user:
                    logging.error(f"[TOKEN_DEDUCTION] User {user_id} not found when trying to deduct tokens")
                    tokens_used = None
                    tokens_remaining = None
                else:
                    # Store initial token count before any modifications
                    try:
                        initial_tokens = current_user.tokens_remaining
                        logging.info(f"[TOKEN_DEDUCTION] Initial tokens: {initial_tokens}")
                    except Exception as e:
                        logging.error(f"[TOKEN_DEDUCTION] *** ERROR reading initial_tokens: {str(e)} ***")
                        if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                            logging.error(f"[TOKEN_DEDUCTION] *** INVALID STATE ERROR DETECTED when reading initial_tokens ***")
                        raise
                    
                    # Reset tokens if it's a new day (in case day changed during analysis)
                    try:
                        reset_daily_tokens_if_needed(current_user, db)
                        logging.info(f"[TOKEN_DEDUCTION] After reset_daily_tokens_if_needed, current_user.tokens_remaining: {current_user.tokens_remaining}")
                    except Exception as e:
                        logging.error(f"[TOKEN_DEDUCTION] *** ERROR in reset_daily_tokens_if_needed: {str(e)} ***")
                        if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                            logging.error(f"[TOKEN_DEDUCTION] *** INVALID STATE ERROR DETECTED in reset_daily_tokens_if_needed ***")
                        raise
                    
                    # Check if tokens were reset and get current count
                    try:
                        tokens_after_reset = current_user.tokens_remaining
                        logging.info(f"[TOKEN_DEDUCTION] Tokens after reset check: {tokens_after_reset} (initial: {initial_tokens})")
                    except Exception as e:
                        logging.error(f"[TOKEN_DEDUCTION] *** ERROR reading tokens_after_reset: {str(e)} ***")
                        if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                            logging.error(f"[TOKEN_DEDUCTION] *** INVALID STATE ERROR DETECTED when reading tokens_after_reset ***")
                        raise
                    
                    if initial_tokens != tokens_after_reset:
                        # Tokens were reset, commit and re-query
                        logging.info(f"[TOKEN_DEDUCTION] Tokens were reset ({initial_tokens} -> {tokens_after_reset}), committing...")
                        db.commit()
                        logging.info(f"[TOKEN_DEDUCTION] Commit successful, re-querying user...")
                        # Re-query to get fresh state after commit (prevents invalid state error)
                        current_user = db.query(User).filter(User.id == user_id).first()
                        if not current_user:
                            raise ValueError(f"User {user_id} not found after token reset")
                        try:
                            initial_tokens = current_user.tokens_remaining
                            logging.info(f"[TOKEN_DEDUCTION] Re-queried user, initial_tokens updated: {initial_tokens}")
                        except Exception as e:
                            logging.error(f"[TOKEN_DEDUCTION] *** ERROR reading initial_tokens after re-query: {str(e)} ***")
                            if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                                logging.error(f"[TOKEN_DEDUCTION] *** INVALID STATE ERROR DETECTED when reading initial_tokens after re-query ***")
                            raise
                    
                    # Calculate final token count after deduction (use initial_tokens to avoid accessing expired object)
                    final_token_count = max(0, initial_tokens - token_cost)
                    logging.info(f"[TOKEN_DEDUCTION] Final token count calculated: {final_token_count} (initial: {initial_tokens}, cost: {token_cost})")
                    
                    # Deduct tokens
                    current_user.tokens_remaining = final_token_count
                    logging.info(f"[TOKEN_DEDUCTION] Set current_user.tokens_remaining to {final_token_count}, committing...")
                    # Commit the token deduction
                    db.commit()
                    logging.info(f"[TOKEN_DEDUCTION] Commit successful")
                    
                    # Use the calculated value instead of reading from expired object
                    tokens_used = round(token_cost, 1)
                    tokens_remaining = int(final_token_count)
                    logging.info(f"[TOKEN_DEDUCTION] Token deduction complete - used: {tokens_used}, remaining: {tokens_remaining}")
            except Exception as e:
                db.rollback()
                import logging
                logging.error(f"[TOKEN_DEDUCTION] *** FAILED to deduct tokens: {str(e)} ***")
                if "invalid state" in str(e).lower() or "detached" in str(e).lower():
                    logging.error(f"[TOKEN_DEDUCTION] *** INVALID STATE ERROR DETECTED in token deduction ***")
                import traceback
                logging.error(f"[TOKEN_DEDUCTION] Traceback: {traceback.format_exc()}")
                # Set tokens to None on error so response doesn't include invalid data
                tokens_used = None
                tokens_remaining = None
            finally:
                db.close()
                logging.info(f"[TOKEN_DEDUCTION] Database session closed")
        
        response = _build_response(exercise, file_info, file_size, frame_count, Path(output_path),
                                  output_filename, calc_results, cam_info, form_analysis, squat_phases, visualization_url)
        
        # Add token information to response if user is logged in
        import logging
        logging.info(f"[RESPONSE] is_authenticated: {is_authenticated}, tokens_used: {tokens_used}, tokens_remaining: {tokens_remaining}")
        if is_authenticated:
            if tokens_used is not None and tokens_remaining is not None:
                response["tokens_used"] = tokens_used
                response["tokens_remaining"] = tokens_remaining
                logging.info(f"[RESPONSE] Added tokens to response - used: {tokens_used}, remaining: {tokens_remaining}")
            else:
                logging.warning(f"[RESPONSE] User authenticated but tokens_used or tokens_remaining is None - not adding to response")
        
        return response
    except Exception as e:
        _handle_upload_errors(e)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

