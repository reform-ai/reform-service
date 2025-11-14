"""Exercise 1 form analysis component."""


def _is_local_max(i: int, angles_with_indices: list, min_height: float) -> bool:
    """Checks if index i is a local maximum above min_height."""
    if i < 2 or i >= len(angles_with_indices) - 2:
        return False
    prev2, prev1 = angles_with_indices[i-2][1], angles_with_indices[i-1][1]
    curr = angles_with_indices[i][1]
    next1, next2 = angles_with_indices[i+1][1], angles_with_indices[i+2][1]
    return curr > prev1 and curr > next1 and curr >= min_height and curr > prev2 and curr > next2


def _filter_peaks_by_distance(candidates: list, min_distance: int) -> list:
    """Filters candidate peaks by minimum distance between them."""
    if not candidates:
        return []
    peaks = [candidates[0]]
    for idx, peak_data in candidates[1:]:
        frame_diff = idx - peaks[-1][0]
        if frame_diff >= min_distance:
            peaks.append((idx, peak_data))
        elif peak_data[1] > peaks[-1][1][1]:
            peaks[-1] = (idx, peak_data)
    return [p[1] for p in peaks]


def _find_peaks(angles_with_indices: list, min_height: float, min_distance: int = 30) -> list:
    """Finds local maxima (peaks) representing bottom of squat."""
    candidates = [(i, angles_with_indices[i]) for i in range(2, len(angles_with_indices) - 2)
                  if _is_local_max(i, angles_with_indices, min_height)]
    return _filter_peaks_by_distance(candidates, min_distance)


def _filter_bounce_reps(peaks_with_indices: list, angles_with_indices: list, bounce_threshold: int = 60) -> list:
    """Filters out bounce patterns (two close peaks = one rep)."""
    if len(peaks_with_indices) < 2:
        return peaks_with_indices
    filtered = [peaks_with_indices[0]]
    for i in range(1, len(peaks_with_indices)):
        prev_idx, prev_peak_data = peaks_with_indices[i-1]
        curr_idx, curr_peak_data = peaks_with_indices[i]
        if curr_idx - prev_idx < bounce_threshold and curr_peak_data[1] >= prev_peak_data[1] * 0.9:
            filtered[-1] = (curr_idx, curr_peak_data)
        else:
            filtered.append((curr_idx, curr_peak_data))
    return filtered


def _find_rep_start_end(angles_with_indices: list, peak_frame: int, baseline: float, threshold: float) -> tuple:
    """Finds start and end frames for a single rep around a peak."""
    peak_idx = next(i for i, (f, _) in enumerate(angles_with_indices) if f == peak_frame)
    start_frame = None
    for i in range(peak_idx - 1, -1, -1):
        if angles_with_indices[i][1] < threshold:
            start_frame = angles_with_indices[i+1][0] if i+1 <= peak_idx else peak_frame
            break
    if start_frame is None:
        start_frame = angles_with_indices[0][0]
    end_frame = None
    for i in range(peak_idx + 1, len(angles_with_indices)):
        if angles_with_indices[i][1] < threshold:
            end_frame = angles_with_indices[i-1][0] if i-1 >= peak_idx else peak_frame
            break
    if end_frame is None:
        end_frame = angles_with_indices[-1][0]
    return start_frame, end_frame


def _calculate_baseline(valid_angles: list) -> float:
    """Calculates baseline angle from first/last frames."""
    if len(valid_angles) > 20:
        return min([a for _, a in valid_angles[:10]] + [a for _, a in valid_angles[-10:]])
    return min([a for _, a in valid_angles])


def _build_reps_from_peaks(filtered_peaks: list, valid_angles: list, baseline: float, threshold: float) -> list:
    """Builds rep list from filtered peaks."""
    reps = []
    for peak_idx, (peak_frame, peak_angle) in filtered_peaks:
        start_frame, end_frame = _find_rep_start_end(valid_angles, peak_frame, baseline, threshold)
        if start_frame is not None and end_frame is not None and start_frame < end_frame:
            reps.append({"start_frame": start_frame, "bottom_frame": peak_frame, "end_frame": end_frame})
    return reps


def detect_squat_phases(quad_angles_per_frame: list, fps: float = 30.0) -> dict:
    """Detects all squat reps. Filters bounce patterns."""
    if not quad_angles_per_frame or all(a is None for a in quad_angles_per_frame):
        return {"reps": []}
    valid_angles = [(i, a) for i, a in enumerate(quad_angles_per_frame) if a is not None]
    if not valid_angles:
        return {"reps": []}
    baseline = _calculate_baseline(valid_angles)
    squat_threshold = baseline + 20
    peaks = _find_peaks(valid_angles, squat_threshold)
    if not peaks:
        return {"reps": []}
    bounce_threshold_frames = int(fps * 1.0)
    peaks_with_indices = [(next(i for i, (f, _) in enumerate(valid_angles) if f == p[0]), p) for p in peaks]
    filtered_peaks = _filter_bounce_reps(peaks_with_indices, valid_angles, bounce_threshold_frames)
    return {"reps": _build_reps_from_peaks(filtered_peaks, valid_angles, baseline, squat_threshold)}


def _filter_to_active_phases(torso_angles_per_frame: list, quad_angles_per_frame: list) -> list:
    """Filters torso angles to only active squat phases."""
    phases = detect_squat_phases(quad_angles_per_frame)
    if not phases.get("reps"):
        return torso_angles_per_frame
    active_frames = []
    for rep in phases["reps"]:
        active_frames.extend(range(rep["start_frame"], rep["end_frame"] + 1))
    if not active_frames:
        return torso_angles_per_frame
    return [torso_angles_per_frame[i] if i < len(torso_angles_per_frame) else None for i in active_frames]


def _calculate_torso_metrics(valid_angles: list) -> tuple:
    """Calculates max, avg, and range from valid angles."""
    max_angle = max(valid_angles)
    avg_angle = sum(valid_angles) / len(valid_angles)
    angle_range = max(valid_angles) - min(valid_angles)
    return max_angle, avg_angle, angle_range


def _determine_torso_status(max_angle: float, avg_angle: float) -> tuple:
    """Determines status, score, and message based on torso angle metrics."""
    if max_angle <= 43:
        if 35 <= avg_angle <= 43:
            return "good", 100, f"Excellent torso position. Average forward lean: {avg_angle:.1f}° (within research-based optimal range: 35-43°)."
        elif avg_angle < 35:
            return "good", 95, f"Good torso position. Average forward lean: {avg_angle:.1f}° (below optimal 35-43° range, but acceptable)."
        else:
            return "good", 90, f"Good torso position. Average forward lean: {avg_angle:.1f}° (within acceptable range, optimal: 35-43°)."
    elif max_angle <= 45:
        return "warning", 75, f"Moderate forward lean detected. Max angle: {max_angle:.1f}° (slightly above optimal 35-43° range). Maintain upright posture to optimize performance."
    else:
        return "poor", 50, f"Excessive forward lean detected. Max angle: {max_angle:.1f}° (>45°). Research indicates this exceeds recommended range and may reduce squat effectiveness."


def analyze_torso_angle(torso_angles_per_frame: list, quad_angles_per_frame: list = None) -> dict:
    """Analyzes torso angle for squat form using evidence-based thresholds."""
    if not torso_angles_per_frame or all(a is None for a in torso_angles_per_frame):
        return {"status": "error", "message": "No torso angle data available"}
    if quad_angles_per_frame:
        torso_angles_per_frame = _filter_to_active_phases(torso_angles_per_frame, quad_angles_per_frame)
    valid_angles = [a for a in torso_angles_per_frame if a is not None]
    if not valid_angles:
        return {"status": "error", "message": "No valid torso angle data"}
    max_angle, avg_angle, angle_range = _calculate_torso_metrics(valid_angles)
    status, score, message = _determine_torso_status(max_angle, avg_angle)
    return {"status": status, "score": score, "message": message, "max_angle": round(max_angle, 1),
            "avg_angle": round(avg_angle, 1), "angle_range": round(angle_range, 1)}

