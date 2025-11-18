"""
Pose estimation component - shared across all exercises
Detects body parts and keypoints from video frames
"""

import mediapipe as mp
import cv2
import math


def process_video_streaming_pose(video_path: str, validate: bool = False, required_landmarks: list = None, frame_skip: int = 1) -> tuple:
    """
    Streaming version: Processes frames one at a time from video file.
    Never keeps all frames in memory - processes and discards immediately.
    Returns landmarks list, fps, frame_count, and optionally validation result.
    """
    import cv2
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    results = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        pose.close()
        if validate:
            from src.shared.pose_estimation.landmark_validation import validate_landmarks_batch
            validation_result = validate_landmarks_batch([], required_landmarks)
            return [], 30.0, 0, validation_result
        return [], 30.0, 0, None
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_index = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame if downsampling
            if frame_index % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)
                results.append(result.pose_landmarks)
                processed_count += 1
                # Frame is automatically discarded after processing
            
            frame_index += 1
    finally:
        cap.release()
        pose.close()
    
    # Adjust FPS if frames were downsampled
    if frame_skip > 1:
        fps = fps / frame_skip
    
    if validate:
        from src.shared.pose_estimation.landmark_validation import validate_landmarks_batch
        validation_result = validate_landmarks_batch(results, required_landmarks)
        return results, fps, processed_count, validation_result
    
    return results, fps, processed_count, None


def process_frames_with_pose(frames: list, validate: bool = False, required_landmarks: list = None) -> tuple:
    """
    Processes frames with MediaPipe Pose to extract keypoints.
    Uses batch processing to reduce memory usage.
    Returns landmarks list and optionally validation result.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    results = []
    
    # Process in batches to reduce memory pressure
    # Process 50 frames at a time, then release memory
    BATCH_SIZE = 50
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i:i + BATCH_SIZE]
        for frame in batch:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            results.append(result.pose_landmarks)
        # Explicitly delete batch to free memory
        del batch
    
    pose.close()
    if validate:
        from src.shared.pose_estimation.landmark_validation import validate_landmarks_batch
        validation_result = validate_landmarks_batch(results, required_landmarks)
        return results, validation_result
    return results, None


def _get_segment_angle(point1, point2) -> float:
    """Calculates angle of segment from vertical. Matches calculation.py logic."""
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    angle_from_horizontal = math.degrees(math.atan2(-dy, dx))
    if angle_from_horizontal < 0:
        angle_from_horizontal += 360
    if angle_from_horizontal <= 90:
        return 90 - angle_from_horizontal
    elif angle_from_horizontal <= 180:
        return angle_from_horizontal - 90
    elif angle_from_horizontal <= 270:
        return 270 - angle_from_horizontal
    else:
        return angle_from_horizontal - 270


def _get_ankle_segment_angle(point1, point2) -> float:
    """Calculates ankle angle from heel to knee. Matches calculation.py logic."""
    dx = point2.x - point1.x
    dy = point2.y - point1.y
    angle_from_horizontal = math.degrees(math.atan2(-dy, dx))
    if angle_from_horizontal < 0:
        angle_from_horizontal += 360
    if angle_from_horizontal <= 90:
        return angle_from_horizontal
    elif angle_from_horizontal <= 180:
        return 180 - angle_from_horizontal
    elif angle_from_horizontal <= 270:
        return 270 - angle_from_horizontal
    else:
        return 360 - angle_from_horizontal


def _get_status_color(status: str) -> tuple:
    """Returns BGR color tuple for status: good=white, warning=orange, poor=red."""
    if status == "good":
        return (255, 255, 255)
    elif status == "warning":
        return (0, 165, 255)
    elif status == "poor":
        return (0, 0, 255)
    return (255, 255, 255)


def _determine_worse_side_torso(landmarks, frame_status: dict) -> str:
    """Determines which side is worse for torso asymmetry. Returns 'left', 'right', or None."""
    asymmetry_status = frame_status.get("torso_asymmetry")
    if asymmetry_status not in ["warning", "poor"]:
        return None
    
    if (len(landmarks.landmark) <= 24 or
        not all(landmarks.landmark[i] for i in [11, 12, 23, 24])):
        return None
    
    left_angle = _get_segment_angle(landmarks.landmark[23], landmarks.landmark[11])
    right_angle = _get_segment_angle(landmarks.landmark[24], landmarks.landmark[12])
    
    if left_angle is None or right_angle is None:
        return None
    
    asymmetry_value = right_angle - left_angle
    return "right" if asymmetry_value > 0 else "left" if asymmetry_value < 0 else None


def _determine_worse_side_quad(landmarks, frame_status: dict) -> str:
    """Determines which side is worse for quad asymmetry. Returns 'left', 'right', or None."""
    asymmetry_status = frame_status.get("quad_asymmetry")
    if asymmetry_status not in ["warning", "poor"]:
        return None
    
    if (len(landmarks.landmark) <= 26 or
        not all(landmarks.landmark[i] for i in [23, 24, 25, 26])):
        return None
    
    left_angle = _get_segment_angle(landmarks.landmark[23], landmarks.landmark[25])
    right_angle = _get_segment_angle(landmarks.landmark[24], landmarks.landmark[26])
    
    if left_angle is None or right_angle is None:
        return None
    
    asymmetry_value = right_angle - left_angle
    return "right" if asymmetry_value > 0 else "left" if asymmetry_value < 0 else None


def _determine_worse_side_ankle(landmarks, frame_status: dict) -> str:
    """Determines which side is worse for ankle asymmetry. Returns 'left', 'right', or None."""
    asymmetry_status = frame_status.get("ankle_asymmetry")
    if asymmetry_status not in ["warning", "poor"]:
        return None
    
    if (len(landmarks.landmark) <= 30 or
        not all(landmarks.landmark[i] for i in [25, 26, 29, 30])):
        return None
    
    left_angle = _get_ankle_segment_angle(landmarks.landmark[29], landmarks.landmark[25])
    right_angle = _get_ankle_segment_angle(landmarks.landmark[30], landmarks.landmark[26])
    
    if left_angle is None or right_angle is None:
        return None
    
    asymmetry_value = right_angle - left_angle
    return "right" if asymmetry_value > 0 else "left" if asymmetry_value < 0 else None


def _draw_torso_segment(annotated, landmarks, h: int, w: int, frame_status: dict = None):
    """Draws torso segment (shoulder midpoint to hip midpoint) with color-coding."""
    if (len(landmarks.landmark) <= 24 or
        not all(landmarks.landmark[i] for i in [11, 12, 23, 24])):
        return
    
    shoulder_mid_x = int((landmarks.landmark[11].x + landmarks.landmark[12].x) / 2 * w)
    shoulder_mid_y = int((landmarks.landmark[11].y + landmarks.landmark[12].y) / 2 * h)
    hip_mid_x = int((landmarks.landmark[23].x + landmarks.landmark[24].x) / 2 * w)
    hip_mid_y = int((landmarks.landmark[23].y + landmarks.landmark[24].y) / 2 * h)
    
    torso_status = frame_status.get("torso_angle") if frame_status else None
    torso_color = _get_status_color(torso_status)
    
    cv2.line(annotated, (shoulder_mid_x, shoulder_mid_y), (hip_mid_x, hip_mid_y), torso_color, 2)


def _draw_quad_segments(annotated, landmarks, h: int, w: int, frame_status: dict = None):
    """Draws left and right quad segments (hip to knee) with color-coding."""
    quad_status = frame_status.get("quad_angle") if frame_status else None
    quad_color = _get_status_color(quad_status)
    
    if (len(landmarks.landmark) > 25 and
        landmarks.landmark[23] and landmarks.landmark[25]):
        left_hip_x = int(landmarks.landmark[23].x * w)
        left_hip_y = int(landmarks.landmark[23].y * h)
        left_knee_x = int(landmarks.landmark[25].x * w)
        left_knee_y = int(landmarks.landmark[25].y * h)
        cv2.line(annotated, (left_hip_x, left_hip_y), (left_knee_x, left_knee_y), quad_color, 2)
    
    if (len(landmarks.landmark) > 26 and
        landmarks.landmark[24] and landmarks.landmark[26]):
        right_hip_x = int(landmarks.landmark[24].x * w)
        right_hip_y = int(landmarks.landmark[24].y * h)
        right_knee_x = int(landmarks.landmark[26].x * w)
        right_knee_y = int(landmarks.landmark[26].y * h)
        cv2.line(annotated, (right_hip_x, right_hip_y), (right_knee_x, right_knee_y), quad_color, 2)


def _get_landmark_colors(landmarks, frame_status: dict, landmark_indices: list) -> dict:
    """Returns color dict for each landmark index based on asymmetry status."""
    colors = {}
    default_green = (0, 255, 0)
    
    for idx in landmark_indices:
        colors[idx] = default_green
    
    if not frame_status:
        return colors
    
    worse_torso_side = _determine_worse_side_torso(landmarks, frame_status)
    worse_quad_side = _determine_worse_side_quad(landmarks, frame_status)
    worse_ankle_side = _determine_worse_side_ankle(landmarks, frame_status)
    
    torso_status = frame_status.get("torso_asymmetry")
    quad_status = frame_status.get("quad_asymmetry")
    ankle_status = frame_status.get("ankle_asymmetry")
    
    if worse_torso_side and torso_status in ["warning", "poor"]:
        color = _get_status_color(torso_status)
        if worse_torso_side == "left":
            colors[11] = color
        else:
            colors[12] = color
    
    if worse_quad_side and quad_status in ["warning", "poor"]:
        color = _get_status_color(quad_status)
        if worse_quad_side == "left":
            colors[25] = color
        else:
            colors[26] = color
    
    if worse_ankle_side and ankle_status in ["warning", "poor"]:
        color = _get_status_color(ankle_status)
        if worse_ankle_side == "left":
            colors[29] = color
        else:
            colors[30] = color
    
    return colors


def create_visualization_streaming(video_path: str, landmarks_list: list, output_path: str,
                                   landmark_indices: list, per_frame_status: dict = None, 
                                   fps: float = 30.0, frame_skip: int = 1) -> str:
    """
    Streaming version: Reads frames from video, draws landmarks, writes directly to output video.
    Never keeps all frames in memory - processes one frame at a time.
    
    Args:
        video_path: Path to input video file
        landmarks_list: List of MediaPipe pose landmarks (must match processed frames)
        output_path: Path to output video file
        landmark_indices: List of landmark indices to draw
        per_frame_status: Optional dict mapping frame index to status dict
        fps: Frames per second for output video
        frame_skip: Frame skip factor (must match the one used for pose estimation)
    
    Returns:
        Path to output video file
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer - try multiple codecs (headless OpenCV may not have all codecs)
    # Try codecs in order of preference
    # Note: Use .avi extension for better compatibility with some codecs
    import os
    base_path = os.path.splitext(output_path)[0]
    temp_output_path = f"{base_path}.avi"  # Use .avi for better codec support
    
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4 Part 2 (most compatible)
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Xvid codec
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG
    ]
    
    out = None
    used_codec = None
    for codec_name, fourcc in codecs_to_try:
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        if out.isOpened():
            used_codec = codec_name
            break
        if out:
            out.release()
        out = None
    
    if not out or not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video file: {temp_output_path}. Tried codecs: {[c[0] for c in codecs_to_try]}")
    
    frame_index = 0
    landmark_index = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames that match the downsampled landmarks
            if frame_index % frame_skip == 0:
                if landmark_index >= len(landmarks_list):
                    break
                
                annotated = frame.copy()
                landmarks = landmarks_list[landmark_index]
                
                if landmarks:
                    h, w, _ = frame.shape
                    frame_status = per_frame_status.get(landmark_index) if per_frame_status else None
                    
                    _draw_torso_segment(annotated, landmarks, h, w, frame_status)
                    _draw_quad_segments(annotated, landmarks, h, w, frame_status)
                    
                    colors = _get_landmark_colors(landmarks, frame_status, landmark_indices)
                    
                    for idx in landmark_indices:
                        if idx < len(landmarks.landmark) and landmarks.landmark[idx]:
                            lm = landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(annotated, (x, y), 5, colors.get(idx, (0, 255, 0)), -1)
                
                # Write frame directly to output video (don't keep in memory)
                out.write(annotated)
                landmark_index += 1
            
            frame_index += 1
    finally:
        cap.release()
        if out:
            out.release()
    
    # Verify video file was created and has content
    if not os.path.exists(temp_output_path):
        raise ValueError(f"Output video file was not created: {temp_output_path}")
    
    file_size = os.path.getsize(temp_output_path)
    if file_size == 0:
        raise ValueError(f"Output video file is empty: {temp_output_path}")
    
    # Verify video can be opened (basic validation)
    test_cap = cv2.VideoCapture(temp_output_path)
    if not test_cap.isOpened():
        raise ValueError(f"Output video file cannot be opened: {temp_output_path}")
    test_cap.release()
    
    # If original path was .mp4, rename .avi to .mp4 (browsers can play .avi too, but .mp4 is preferred)
    if output_path.endswith('.mp4') and temp_output_path != output_path:
        # Try to rename, but if it fails, return the .avi file
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_output_path, output_path)
            return output_path
        except Exception:
            # If rename fails, return the .avi file (browsers can still play it)
            return temp_output_path
    
    return temp_output_path


def draw_landmarks_on_frames(frames: list, landmarks_list: list, 
                             landmark_indices: list, per_frame_status: dict = None, fps: float = 30.0) -> list:
    """
    Draws specified landmarks on frames with optional color coding based on per-frame status.
    Also draws torso and quad segments for biomechanical visualization.
    
    Args:
        frames: List of video frames
        landmarks_list: List of MediaPipe pose landmarks
        landmark_indices: List of landmark indices to draw
        per_frame_status: Optional dict mapping frame index to status dict
        fps: Frames per second (unused, kept for compatibility)
    
    Returns:
        List of annotated frames with landmarks drawn
    """
    annotated_frames = []
    for frame_idx, (frame, landmarks) in enumerate(zip(frames, landmarks_list)):
        annotated = frame.copy()
        if landmarks:
            h, w, _ = frame.shape
            frame_status = per_frame_status.get(frame_idx) if per_frame_status else None
            
            _draw_torso_segment(annotated, landmarks, h, w, frame_status)
            _draw_quad_segments(annotated, landmarks, h, w, frame_status)
            
            colors = _get_landmark_colors(landmarks, frame_status, landmark_indices)
            
            for idx in landmark_indices:
                if idx < len(landmarks.landmark) and landmarks.landmark[idx]:
                    lm = landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (x, y), 5, colors.get(idx, (0, 255, 0)), -1)
        
        annotated_frames.append(annotated)
    return annotated_frames
