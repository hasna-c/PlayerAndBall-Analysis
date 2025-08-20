from typing import Dict
import cv2
import numpy as np
from supervision.draw.color import Color
from supervision.geometry.dataclasses import Point, Rect
from supervision.tools.detections import Detections

# ------------------------------
# Touch Counter for Assignment
# ------------------------------
class TouchCounter:
    def __init__(self):
        self.right_leg_count = 0
        self.left_leg_count = 0
        self.prev_ball_center = None
        self.prev_player_center = None
        self.ball_rotation = "None"
        self.player_velocity = 0.0

    def update(self, keypoints, ball_bboxes):
        if keypoints is None or len(keypoints) < 17:
            return
        if len(ball_bboxes) == 0:
            return

        # Ball center (use first detection if multiple)
        x1, y1, x2, y2 = ball_bboxes[0]
        ball_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Keypoints (YOLOv8 format)
        right_foot = np.array(keypoints[16])  # Right foot
        left_foot = np.array(keypoints[15])   # Left foot

        # Distance to ball
        dist_right = np.linalg.norm(ball_center - right_foot)
        dist_left = np.linalg.norm(ball_center - left_foot)

        # Threshold for touch
        touch_threshold = 30  # pixels, adjust per video scale
        if dist_right < touch_threshold:
            self.right_leg_count += 1
        if dist_left < touch_threshold:
            self.left_leg_count += 1

        # Ball rotation (forward/backward)
        if self.prev_ball_center is not None:
            delta_y = ball_center[1] - self.prev_ball_center[1]
            self.ball_rotation = "Forward" if delta_y < 0 else "Backward"
        self.prev_ball_center = ball_center

        # Player velocity (use hips midpoint)
        hip_center = (np.array(keypoints[11]) + np.array(keypoints[12])) / 2
        if self.prev_player_center is not None:
            self.player_velocity = np.linalg.norm(hip_center - self.prev_player_center)
        self.prev_player_center = hip_center


# ------------------------------
# Annotator for Overlay
# ------------------------------
class AssignmentAnnotator:
    def __init__(self, text_scale=1, text_thickness=2):
        self.text_scale = text_scale
        self.text_thickness = text_thickness

    def annotate(self, frame, touch_counter: TouchCounter):
        # Overlay Right/Left leg touches
        cv2.putText(frame, f"Right Leg Touches: {touch_counter.right_leg_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, (0, 0, 255), self.text_thickness)
        cv2.putText(frame, f"Left Leg Touches: {touch_counter.left_leg_count}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, (0, 255, 0), self.text_thickness)

        # Overlay Ball Rotation
        cv2.putText(frame, f"Ball Rotation: {touch_counter.ball_rotation}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, (255, 0, 0), self.text_thickness)

        # Overlay Player Velocity
        cv2.putText(frame, f"Player Velocity: {touch_counter.player_velocity:.2f} px/frame",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, (255, 255, 0), self.text_thickness)

        return frame
