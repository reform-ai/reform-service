"""
Per-frame status calculation for visualization.
Determines good/warning/poor status for each frame based on angles and asymmetry.
Usable by both upload and livestream modes.
"""


def _determine_torso_status_per_frame(angle: float) -> str:
    """
    Determines torso angle status for a single frame.
    Uses same thresholds as aggregate analysis but applied per-frame.
    
    Args:
        angle: Torso angle in degrees for this frame
        
    Returns:
        "good", "warning", or "poor"
    """
    if angle is None:
        return None
    
    if angle <= 43:
        return "good"
    elif angle <= 45:
        return "warning"
    else:
        return "poor"


def _determine_quad_status_per_frame(angle: float) -> str:
    """
    Determines quad angle (depth) status for a single frame.
    Uses same thresholds as aggregate analysis but applied per-frame.
    
    Args:
        angle: Quad angle in degrees for this frame
        
    Returns:
        "good", "warning", or "poor"
    """
    if angle is None:
        return None
    
    if angle >= 70:
        return "good"
    elif angle >= 60:
        return "warning"
    else:
        return "poor"


def _determine_ankle_status_per_frame(angle: float) -> str:
    """
    Determines ankle angle (mobility) status for a single frame.
    Uses same thresholds as aggregate analysis but applied per-frame.
    
    Args:
        angle: Ankle angle in degrees for this frame
        
    Returns:
        "good", "warning", or "poor"
    """
    if angle is None:
        return None
    
    if angle <= 60:
        return "good"
    elif angle <= 70:
        return "warning"
    else:
        return "poor"


def _determine_asymmetry_status_per_frame(asymmetry: float) -> str:
    """
    Determines asymmetry status for a single frame.
    Uses same thresholds as aggregate analysis but applied per-frame.
    
    Args:
        asymmetry: Asymmetry value in degrees for this frame (can be negative)
        
    Returns:
        "good", "warning", or "poor"
    """
    if asymmetry is None:
        return None
    
    abs_asymmetry = abs(asymmetry)
    if abs_asymmetry < 5:
        return "good"
    elif abs_asymmetry < 10:
        return "warning"
    else:
        return "poor"


def calculate_single_frame_status(
    angles: dict,
    asymmetry: dict = None,
    glute_dominance_status: str = None
) -> dict:
    """
    Calculates status for a single frame for real-time visualization color coding.
    Optimized for livestream mode where frames are processed one at a time.
    
    Args:
        angles: Dict with keys:
            - "torso_angle": float (single value, not a list)
            - "quad_angle": float (single value, not a list)
            - "ankle_angle": float (single value, not a list)
        asymmetry: Optional dict with keys:
            - "torso_asymmetry": float (single value, not a list)
            - "quad_asymmetry": float (single value, not a list)
            - "ankle_asymmetry": float (single value, not a list)
        glute_dominance_status: Optional overall glute dominance status ("good", "warning", "poor")
            Note: Movement pattern is rep-based, so we use overall status for the frame
    
    Returns:
        Single status dict (not frame-indexed):
        {
            "torso_angle": "good",
            "quad_angle": "warning",
            "ankle_angle": "good",
            "torso_asymmetry": "good",
            "quad_asymmetry": "poor",
            "ankle_asymmetry": "good",
            "movement": "good"  # if glute_dominance_status provided
        }
    """
    if not angles:
        return {}
    
    frame_status = {}
    frame_status["torso_angle"] = _determine_torso_status_per_frame(angles.get("torso_angle"))
    frame_status["quad_angle"] = _determine_quad_status_per_frame(angles.get("quad_angle"))
    frame_status["ankle_angle"] = _determine_ankle_status_per_frame(angles.get("ankle_angle"))
    
    if asymmetry:
        frame_status["torso_asymmetry"] = _determine_asymmetry_status_per_frame(asymmetry.get("torso_asymmetry"))
        frame_status["quad_asymmetry"] = _determine_asymmetry_status_per_frame(asymmetry.get("quad_asymmetry"))
        frame_status["ankle_asymmetry"] = _determine_asymmetry_status_per_frame(asymmetry.get("ankle_asymmetry"))
    else:
        frame_status["torso_asymmetry"] = None
        frame_status["quad_asymmetry"] = None
        frame_status["ankle_asymmetry"] = None
    
    frame_status["movement"] = glute_dominance_status if glute_dominance_status else None
    
    return frame_status


def smooth_per_frame_status(per_frame_status: dict, fps: float, window_duration_seconds: float = 0.2) -> dict:
    """
    Applies temporal smoothing to per-frame status to reduce flickering.
    Uses majority vote over a time window.
    
    Args:
        per_frame_status: Dict mapping frame index to status dict
        fps: Frames per second
        window_duration_seconds: Duration of smoothing window in seconds (default: 0.2s)
    
    Returns:
        Smoothed per-frame status dict with same structure
    """
    if not per_frame_status:
        return {}
    
    frame_count = max(per_frame_status.keys()) + 1 if per_frame_status else 0
    if frame_count == 0:
        return {}
    
    window_size = max(1, int(fps * window_duration_seconds))
    smoothed_status = {}
    status_priority = {"poor": 3, "warning": 2, "good": 1, None: 0}
    
    for frame_idx in range(frame_count):
        window_start = max(0, frame_idx - window_size + 1)
        window_frames = list(range(window_start, frame_idx + 1))
        frame_status = per_frame_status.get(frame_idx, {})
        smoothed_frame_status = {}
        
        all_metrics = set()
        for win_frame_idx in window_frames:
            win_status = per_frame_status.get(win_frame_idx, {})
            all_metrics.update(win_status.keys())
        
        for metric in ["torso_angle", "quad_angle", "ankle_angle", 
                      "torso_asymmetry", "quad_asymmetry", "ankle_asymmetry", "movement"]:
            if metric not in all_metrics:
                continue
            
            statuses_in_window = []
            for win_frame_idx in window_frames:
                win_status = per_frame_status.get(win_frame_idx, {})
                if metric in win_status:
                    statuses_in_window.append(win_status[metric])
            
            if not statuses_in_window:
                smoothed_frame_status[metric] = None
                continue
            
            status_counts = {}
            for status in statuses_in_window:
                status_counts[status] = status_counts.get(status, 0) + 1
            
            max_count = max(status_counts.values())
            candidates = [s for s, count in status_counts.items() if count == max_count]
            smoothed_frame_status[metric] = max(candidates, key=lambda s: status_priority.get(s, 0))
        
        smoothed_status[frame_idx] = smoothed_frame_status
    
    return smoothed_status


def calculate_per_frame_status(
    angles_per_frame: dict,
    asymmetry_per_frame: dict = None,
    glute_dominance_status: str = None
) -> dict:
    """
    Calculates per-frame status for visualization color coding.
    
    Args:
        angles_per_frame: Dict with keys:
            - "torso_angle": list of torso angles per frame
            - "quad_angle": list of quad angles per frame
            - "ankle_angle": list of ankle angles per frame
        asymmetry_per_frame: Optional dict with keys:
            - "torso_asymmetry": list of torso asymmetry per frame
            - "quad_asymmetry": list of quad asymmetry per frame
            - "ankle_asymmetry": list of ankle asymmetry per frame
        glute_dominance_status: Optional overall glute dominance status ("good", "warning", "poor")
            Note: Movement pattern is rep-based, so we use overall status for all frames
    
    Returns:
        Dict mapping frame index to status dict:
        {
            0: {
                "torso_angle": "good",
                "quad_angle": "warning",
                "ankle_angle": "good",
                "torso_asymmetry": "good",
                "quad_asymmetry": "poor",
                "ankle_asymmetry": "good",
                "movement": "good"  # if glute_dominance_status provided
            },
            1: {...},
            ...
        }
    """
    if not angles_per_frame:
        return {}
    
    frame_count = 0
    for angle_list in angles_per_frame.values():
        if angle_list:
            frame_count = max(frame_count, len(angle_list))
    
    if frame_count == 0:
        return {}
    
    per_frame_status = {}
    torso_angles = angles_per_frame.get("torso_angle", [])
    quad_angles = angles_per_frame.get("quad_angle", [])
    ankle_angles = angles_per_frame.get("ankle_angle", [])
    torso_asymmetry = asymmetry_per_frame.get("torso_asymmetry", []) if asymmetry_per_frame else []
    quad_asymmetry = asymmetry_per_frame.get("quad_asymmetry", []) if asymmetry_per_frame else []
    ankle_asymmetry = asymmetry_per_frame.get("ankle_asymmetry", []) if asymmetry_per_frame else []
    
    for frame_idx in range(frame_count):
        frame_status = {}
        
        if frame_idx < len(torso_angles):
            frame_status["torso_angle"] = _determine_torso_status_per_frame(torso_angles[frame_idx])
        else:
            frame_status["torso_angle"] = None
        
        if frame_idx < len(quad_angles):
            frame_status["quad_angle"] = _determine_quad_status_per_frame(quad_angles[frame_idx])
        else:
            frame_status["quad_angle"] = None
        
        if frame_idx < len(ankle_angles):
            frame_status["ankle_angle"] = _determine_ankle_status_per_frame(ankle_angles[frame_idx])
        else:
            frame_status["ankle_angle"] = None
        
        if frame_idx < len(torso_asymmetry):
            frame_status["torso_asymmetry"] = _determine_asymmetry_status_per_frame(torso_asymmetry[frame_idx])
        else:
            frame_status["torso_asymmetry"] = None
        
        if frame_idx < len(quad_asymmetry):
            frame_status["quad_asymmetry"] = _determine_asymmetry_status_per_frame(quad_asymmetry[frame_idx])
        else:
            frame_status["quad_asymmetry"] = None
        
        if frame_idx < len(ankle_asymmetry):
            frame_status["ankle_asymmetry"] = _determine_asymmetry_status_per_frame(ankle_asymmetry[frame_idx])
        else:
            frame_status["ankle_asymmetry"] = None
        
        frame_status["movement"] = glute_dominance_status if glute_dominance_status else None
        
        per_frame_status[frame_idx] = frame_status
    
    return per_frame_status

