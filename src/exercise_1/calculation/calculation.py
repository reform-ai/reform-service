"""
Exercise 1 calculation component
Uses pose estimation data points to determine form correctness
"""

import math


def _calculate_z_diffs_from_frames(landmarks_list: list) -> float:
    """Calculates average z-depth difference across frames using ears, shoulders, hips, heels."""
    num_frames = min(10, len(landmarks_list))
    z_diffs = []
    landmark_pairs = [(7, 8), (11, 12), (23, 24), (29, 30)]
    for i in range(num_frames):
        if not landmarks_list[i]:
            continue
        frame_z_diffs = []
        for left_idx, right_idx in landmark_pairs:
            left_pt = landmarks_list[i].landmark[left_idx]
            right_pt = landmarks_list[i].landmark[right_idx]
            frame_z_diffs.append(right_pt.z - left_pt.z)
        if frame_z_diffs:
            z_diffs.append(sum(frame_z_diffs) / len(frame_z_diffs))
    return sum(z_diffs) / len(z_diffs) if z_diffs else 0.0


def _estimate_angle_from_z_diff(z_diff: float) -> float:
    """Converts z-depth difference to rotation angle. Calibrated: z-diff 0.174 → 45°."""
    abs_z_diff = abs(z_diff)
    if abs_z_diff > 1.0:
        return 90 if abs_z_diff > 1.5 else 80 + (abs_z_diff - 1.0) * 20
    elif abs_z_diff < 0.05:
        return abs_z_diff * 200
    else:
        return min(abs_z_diff * 258, 80)


def _get_orientation_info(angle_estimate: float) -> tuple:
    """Returns orientation string and is_perpendicular flag based on angle."""
    dist_0 = abs(angle_estimate - 0)
    dist_90 = abs(angle_estimate - 90)
    if dist_0 <= 10:
        return "perpendicular (facing camera)", True
    elif dist_90 <= 10:
        return "sideways (facing 90° to camera)", False
    else:
        return "angled", False


def _determine_angle_status(angle_estimate: float) -> dict:
    """Determines status, message, and flags based on angle estimate."""
    dist_0 = abs(angle_estimate - 0)
    dist_90 = abs(angle_estimate - 90)
    is_acceptable = dist_0 <= 10 or dist_90 <= 10
    orientation, is_perp = _get_orientation_info(angle_estimate)
    if is_acceptable:
        msg = f"Person is positioned {orientation} ({round(angle_estimate, 1)}°). Measurements should be accurate."
        return {"status": "ok", "should_reject": False, "should_warn": False,
                "message": msg, "is_perpendicular": is_perp}
    elif dist_0 <= 15 or dist_90 <= 15:
        msg = f"Person is angled {round(angle_estimate, 1)}° (not quite perpendicular or sideways). Asymmetry measurements may be affected. For best results, position perpendicular (0°) or sideways (90°) to camera (within ±10°)."
        return {"status": "warning", "should_reject": False, "should_warn": True,
                "message": msg, "is_perpendicular": False}
    else:
        msg = f"Person is angled {round(angle_estimate, 1)}° (not perpendicular or sideways). Angle is too extreme for accurate measurements. Please record again with person perpendicular (0°) or sideways (90°) to camera (within ±10°)."
        return {"status": "reject", "should_reject": True, "should_warn": False,
                "message": msg, "is_perpendicular": False}


def detect_camera_angle(landmarks_list: list) -> dict:
    """Detects camera angle using z-depth from ears, shoulders, hips, heels. Returns angle info."""
    if not landmarks_list or not landmarks_list[0]:
        return {"is_perpendicular": None, "angle_estimate": None, "status": "error",
                "message": "No landmarks detected", "should_reject": False}
    avg_z_diff = _calculate_z_diffs_from_frames(landmarks_list)
    if avg_z_diff == 0.0:
        return {"is_perpendicular": None, "angle_estimate": None, "status": "error",
                "message": "No valid landmarks found", "should_reject": False}
    angle_estimate = _estimate_angle_from_z_diff(avg_z_diff)
    result = _determine_angle_status(angle_estimate)
    result.update({"angle_estimate": round(angle_estimate, 1), "z_difference": round(avg_z_diff, 3)})
    return result


def get_segment_angle(point1, point2) -> float:
    """Calculates angle of segment from vertical. Returns angle in degrees (0 when upright, 90 when bent forward)."""
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


def calculate_torso_angle_per_frame(landmarks_list: list) -> list:
    """Calculates torso angle for each frame."""
    if not landmarks_list:
        return []
    angles = []
    for landmarks in landmarks_list:
        if not landmarks:
            angles.append(None)
            continue
        left_angle = get_segment_angle(landmarks.landmark[23], landmarks.landmark[11])
        right_angle = get_segment_angle(landmarks.landmark[24], landmarks.landmark[12])
        if left_angle is not None and right_angle is not None:
            angles.append((left_angle + right_angle) / 2)
        else:
            angles.append(None)
    return angles


def calculate_quad_angle_per_frame(landmarks_list: list) -> list:
    """Calculates quad angle for each frame."""
    if not landmarks_list:
        return []
    angles = []
    for landmarks in landmarks_list:
        if not landmarks:
            angles.append(None)
            continue
        left_angle = get_segment_angle(landmarks.landmark[23], landmarks.landmark[25])
        right_angle = get_segment_angle(landmarks.landmark[24], landmarks.landmark[26])
        if left_angle is not None and right_angle is not None:
            angles.append((left_angle + right_angle) / 2)
        else:
            angles.append(None)
    return angles


def get_ankle_segment_angle(point1, point2) -> float:
    """Calculates angle of heel-knee segment. Returns angle in degrees (90 when upright, < 90 when knee forward)."""
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


def calculate_ankle_angle_per_frame(landmarks_list: list) -> list:
    """Calculates ankle angle for each frame from heel-knee segments."""
    if not landmarks_list:
        return []
    angles = []
    for landmarks in landmarks_list:
        if not landmarks:
            angles.append(None)
            continue
        left_angle = get_ankle_segment_angle(landmarks.landmark[29], landmarks.landmark[25])
        right_angle = get_ankle_segment_angle(landmarks.landmark[30], landmarks.landmark[26])
        if left_angle is not None and right_angle is not None:
            angles.append((left_angle + right_angle) / 2)
        else:
            angles.append(None)
    return angles


def calculate_torso_asymmetry_per_frame(landmarks_list: list) -> list:
    """Calculates left-right torso asymmetry. Positive = right leaning, negative = left leaning.
    Note: This measures 2D image asymmetry. If person is not perpendicular to camera,
    perspective effects can create apparent asymmetry even with symmetric form."""
    if not landmarks_list:
        return []
    asymmetry = []
    for landmarks in landmarks_list:
        if not landmarks:
            asymmetry.append(None)
            continue
        left_angle = get_segment_angle(landmarks.landmark[23], landmarks.landmark[11])
        right_angle = get_segment_angle(landmarks.landmark[24], landmarks.landmark[12])
        if left_angle is not None and right_angle is not None:
            asymmetry.append(right_angle - left_angle)
        else:
            asymmetry.append(None)
    return asymmetry


def calculate_quad_asymmetry_per_frame(landmarks_list: list) -> list:
    """Calculates left-right quad asymmetry. Positive = right forward, negative = left forward.
    Note: This measures 2D image asymmetry. If person is not perpendicular to camera,
    perspective effects can create apparent asymmetry even with symmetric form."""
    if not landmarks_list:
        return []
    asymmetry = []
    for landmarks in landmarks_list:
        if not landmarks:
            asymmetry.append(None)
            continue
        left_angle = get_segment_angle(landmarks.landmark[23], landmarks.landmark[25])
        right_angle = get_segment_angle(landmarks.landmark[24], landmarks.landmark[26])
        if left_angle is not None and right_angle is not None:
            asymmetry.append(right_angle - left_angle)
        else:
            asymmetry.append(None)
    return asymmetry


def calculate_ankle_asymmetry_per_frame(landmarks_list: list) -> list:
    """Calculates left-right ankle asymmetry. Positive = right forward, negative = left forward.
    Note: This measures 2D image asymmetry. If person is not perpendicular to camera,
    perspective effects can create apparent asymmetry even with symmetric form."""
    if not landmarks_list:
        return []
    asymmetry = []
    for landmarks in landmarks_list:
        if not landmarks:
            asymmetry.append(None)
            continue
        left_angle = get_ankle_segment_angle(landmarks.landmark[29], landmarks.landmark[25])
        right_angle = get_ankle_segment_angle(landmarks.landmark[30], landmarks.landmark[26])
        if left_angle is not None and right_angle is not None:
            asymmetry.append(right_angle - left_angle)
        else:
            asymmetry.append(None)
    return asymmetry


def calculate_squat_form(landmarks_list: list) -> dict:
    """Calculates squat form metrics from pose landmarks. Returns per-frame angles and asymmetry."""
    torso_angles_per_frame = calculate_torso_angle_per_frame(landmarks_list)
    quad_angles_per_frame = calculate_quad_angle_per_frame(landmarks_list)
    ankle_angles_per_frame = calculate_ankle_angle_per_frame(landmarks_list)
    torso_asymmetry_per_frame = calculate_torso_asymmetry_per_frame(landmarks_list)
    quad_asymmetry_per_frame = calculate_quad_asymmetry_per_frame(landmarks_list)
    ankle_asymmetry_per_frame = calculate_ankle_asymmetry_per_frame(landmarks_list)
    return {
        "exercise": 1,
        "angles_per_frame": {
            "torso_angle": torso_angles_per_frame,
            "quad_angle": quad_angles_per_frame,
            "ankle_angle": ankle_angles_per_frame
        },
        "asymmetry_per_frame": {
            "torso_asymmetry": torso_asymmetry_per_frame,
            "quad_asymmetry": quad_asymmetry_per_frame,
            "ankle_asymmetry": ankle_asymmetry_per_frame
        }
    }
