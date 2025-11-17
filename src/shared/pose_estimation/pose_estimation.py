"""
Pose estimation component - shared across all exercises
Detects body parts and keypoints from video frames
"""

import mediapipe as mp
import cv2
import math


def process_frames_with_pose(frames: list, validate: bool = False, required_landmarks: list = None) -> tuple:
    """
    Processes frames with MediaPipe Pose to extract keypoints.
    Returns landmarks list and optionally validation result.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    results = []
    
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        results.append(result.pose_landmarks)
    
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
