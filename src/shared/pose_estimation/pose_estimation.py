"""
Pose estimation component - shared across all exercises
Detects body parts and keypoints from video frames
"""

import mediapipe as mp
import cv2


def process_frames_with_pose(frames: list) -> list:
    """
    Processes frames with MediaPipe Pose to extract keypoints
    Returns list of pose landmarks for each frame
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    results = []
    
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)
        results.append(result.pose_landmarks)
    
    pose.close()
    return results


def draw_landmarks_on_frames(frames: list, landmarks_list: list, 
                             landmark_indices: list) -> list:
    """
    Draws specified landmarks on frames
    Returns frames with landmarks visualized
    """
    annotated_frames = []
    for frame, landmarks in zip(frames, landmarks_list):
        annotated = frame.copy()
        if landmarks:
            h, w, _ = frame.shape
            for idx in landmark_indices:
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
        annotated_frames.append(annotated)
    return annotated_frames

