"""Reform Service - FastAPI server for exercise form analysis."""

import os
import tempfile
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
from src.exercise_1.calculation.calculation import calculate_squat_form

app = FastAPI(
    title="Reform Service",
    description="Exercise form analysis service using LLM and Computer Vision",
    version="0.1.0"
)

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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


def route_to_exercise_calculation(exercise: int, landmarks_list: list) -> dict:
    """Routes to appropriate exercise calculation module."""
    if exercise == 1:
        return calculate_squat_form(landmarks_list)
    elif exercise == 2:
        return {"exercise": 2, "message": "Exercise 2 not implemented"}
    elif exercise == 3:
        return {"exercise": 3, "message": "Exercise 3 not implemented"}
    else:
        raise ValueError(f"Invalid exercise: {exercise}")


def _validate_exercise(exercise: int) -> None:
    """Validates exercise type."""
    if exercise not in [1, 2, 3]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid exercise type: {exercise}. Must be 1 (Squat), 2 (Bench), or 3 (Deadlift)"
        )


async def _validate_file(video: UploadFile) -> tuple:
    """Validates video file and returns file info and size."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    file_info = accept_video_file(video)
    contents = await video.read()
    file_size = len(contents)
    await video.seek(0)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="Empty file received")
    MAX_FILE_SIZE = 500 * 1024 * 1024
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large: {file_size} bytes. Maximum: {MAX_FILE_SIZE} bytes")
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File appears to be too small to be a valid video")
    return file_info, file_size


def _check_camera_angle(landmarks_list: list) -> dict:
    """Checks camera angle and raises exception if too extreme."""
    from src.exercise_1.calculation.calculation import detect_camera_angle
    camera_angle_info = detect_camera_angle(landmarks_list)
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


def _analyze_exercise_form(exercise: int, calculation_results: dict, fps: float) -> tuple:
    """Analyzes exercise form and returns form_analysis and squat_phases."""
    form_analysis = None
    squat_phases = None
    if exercise == 1 and calculation_results.get("angles_per_frame"):
        from src.exercise_1.llm_form_analysis.llm_form_analysis import analyze_torso_angle, detect_squat_phases
        squat_phases = detect_squat_phases(calculation_results["angles_per_frame"].get("quad_angle", []), fps)
        torso_analysis = analyze_torso_angle(
            calculation_results["angles_per_frame"]["torso_angle"],
            calculation_results["angles_per_frame"].get("quad_angle")
        )
        form_analysis = {"torso_angle": torso_analysis}
    return form_analysis, squat_phases


def _process_video_analysis(video: UploadFile, exercise: int, frames: list, fps: float, landmarks_list: list) -> tuple:
    """Processes video analysis and returns results."""
    camera_angle_info = _check_camera_angle(landmarks_list)
    calculation_results = route_to_exercise_calculation(exercise, landmarks_list)
    form_analysis, squat_phases = _analyze_exercise_form(exercise, calculation_results, fps)
    return calculation_results, camera_angle_info, form_analysis, squat_phases


@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...), exercise: int = Form(...)):
    """Accepts video file upload and processes with pose estimation."""
    temp_path = None
    try:
        _validate_exercise(exercise)
        file_info, file_size = await _validate_file(video)
        temp_path = save_video_temp(video)
        frames, fps = extract_frames(temp_path)
        landmarks_list = process_frames_with_pose(frames)
        calculation_results, camera_angle_info, form_analysis, squat_phases = _process_video_analysis(
            video, exercise, frames, fps, landmarks_list
        )
        landmark_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 29, 30, 31, 32]
        annotated_frames = draw_landmarks_on_frames(frames, landmarks_list, landmark_indices)
        video_id = str(uuid.uuid4())
        output_filename = f"{video_id}.mp4"
        output_path = OUTPUTS_DIR / output_filename
        save_frames_as_video(annotated_frames, str(output_path), fps)
        exercise_names = {1: "Squat", 2: "Bench", 3: "Deadlift"}
        return {
            "status": "success",
            "message": "Video processed successfully",
            "exercise": exercise,
            "exercise_name": exercise_names[exercise],
            "filename": file_info["filename"],
            "content_type": file_info["content_type"],
            "size": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "frame_count": len(frames),
            "visualization_path": str(output_path),
            "visualization_url": f"http://127.0.0.1:8000/outputs/{output_filename}",
            "calculation_results": calculation_results,
            "camera_angle_info": camera_angle_info,
            "form_analysis": form_analysis,
            "squat_phases": squat_phases,
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

