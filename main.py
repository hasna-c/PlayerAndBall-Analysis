from supervision.draw.color import ColorPalette
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from tqdm import tqdm
import numpy as np
import cv2
from ultralytics import YOLO
from utils import TouchCounter, AssignmentAnnotator

# ------------------------------
# File paths for YOLO models
# ------------------------------
MODEL_POSE_PATH = "yolov8n-pose.pt"
MODEL_OBJECT_PATH = "yolov8n.pt"

# Initialize YOLO models
model_pose = YOLO(MODEL_POSE_PATH)
model_object = YOLO(MODEL_OBJECT_PATH)

# Class names dictionary from the object detection model
CLASS_NAMES_DICT = model_object.model.names

# Class ID for the "ball" (check your dataset, 32 ~ sports ball in COCO)
CLASS_ID_BALL = [32]

SOURCE_VIDEO_PATH = "dataset/Toe Taps.mp4"
TARGET_VIDEO_PATH = "dataset/toetaps_result.mp4"

# Create VideoInfo instance from the source video
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
frame_generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# Annotators
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1,
                             text_thickness=1, text_scale=0.5)
touch_counter = TouchCounter()
assignment_annotator = AssignmentAnnotator()

# ------------------------------
# Process video
# ------------------------------
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    window_name = "Assignment Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, video_info.width, video_info.height)

    for frame in tqdm(frame_generator, total=video_info.total_frames):
        # Pose estimation
        results_pose = model_pose.track(frame, persist=True)
        annotated_frame = results_pose[0].plot()
        keypoints = None
        if results_pose[0].keypoints is not None:
            keypoints = results_pose[0].keypoints.xy.int().cpu().tolist()[0]

        # Object detection (ball)
        results_object = model_object.track(frame, persist=True, conf=0.25)
        tracker_ids = results_object[0].boxes.id.int().cpu().numpy() if results_object[0].boxes.id is not None else None
        detections = Detections(
            xyxy=results_object[0].boxes.xyxy.cpu().numpy(),
            confidence=results_object[0].boxes.conf.cpu().numpy(),
            class_id=results_object[0].boxes.cls.cpu().numpy().astype(int),
            tracker_id=tracker_ids
        )

        # Filter only balls
        mask = np.array([cls in CLASS_ID_BALL for cls in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        ball_bboxes = detections.xyxy

        # Update touch counter
        if keypoints is not None:
            touch_counter.update(keypoints, ball_bboxes)

        # Annotate bounding boxes
        labels = [
            f"id:{track_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, track_id in detections
        ]
        annotated_frame = box_annotator.annotate(
            frame=annotated_frame, detections=detections, labels=labels
        )

        # Annotate assignment metrics
        annotated_frame = assignment_annotator.annotate(annotated_frame, touch_counter)

        # Display + save
        if annotated_frame.shape[0] > 0 and annotated_frame.shape[1] > 0:
            cv2.imshow(window_name, annotated_frame)
            sink.write_frame(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
