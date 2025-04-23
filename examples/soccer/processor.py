from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from gamevision3d.annotators.soccer import draw_pitch, draw_points_on_pitch
from gamevision3d.common.ball import BallTracker, BallAnnotator
from gamevision3d.common.team import TeamClassifier
from gamevision3d.common.view import ViewTransformer
from gamevision3d.configs.soccer import SoccerPitchConfiguration

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF6347', '#0000FF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


import cv2
import numpy as np
import onnxruntime as ort

class ViTPose:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"]
        )
        self.input_shape = self.session.get_inputs()[0].shape  # e.g., [1, 3, 256, 192]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self.valid_keypoint_indices = [5,6,7,8,9,10,11,12,13,14,15,16]

    def preprocess(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        x, y, w, h = bbox
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (self.input_width, self.input_height))
        resized = resized.astype(np.float32) / 255.0
        normalized = (resized - self.mean) / self.std
        chw = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(chw, axis=0)

    def postprocess(self, output: np.ndarray, bbox: tuple) -> list:
        x, y, w, h = bbox
        width_ratio = self.input_width / w
        height_ratio = self.input_height / h

        results = []
        for i in range(0, output.shape[1]):
            px = x + 4 * output[0, i,0] / width_ratio
            py = y + 4 * output[0, i, 1] / height_ratio
            results.append([px, py])

        return [results[i] for i in self.valid_keypoint_indices]

    def detect_pose(self, image: np.ndarray, bbox: tuple) -> list:
        input_tensor = self.preprocess(image, bbox)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        return self.postprocess(outputs[0], bbox)
    
pose_detector = ViTPose("D:/AnasAZ/SourceCode/golf-cv/training/onnx_models/pose_2D_vit_small_moveai29.onnx")

class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'
    ALL = 'ALL'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    
    source=keypoints.xy[0][mask].astype(np.float32)
    target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    if source.shape[0] < 4:
        return None
    transformer = ViewTransformer(
        source=source,
        target=target
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        #overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def draw_skeleton(frame, keypoints, color=(0, 255, 0), radius=1, thickness=2, offset = (0,0), zoom = 1):
    # COCO-style skeleton connection pairs
    skeleton = [
        (1, 3), (3, 5),   # Left arm
        (0, 2), (2, 4),  # Right arm
        (7, 9), (9, 11),  # Left leg
        (6, 8), (8, 10),  # Right leg
        (0, 1), (1, 7),  # Shoulders and hips
        (7, 6), (6, 0),  # Torso connections
    ]
    # Draw joints
    i = 0
    for x, y in keypoints:
        if x > 0 and y > 0:  # Ignore invalid keypoints
            cv2.circle(frame,  (zoom *(int(x) - offset[0]), zoom *(int(y) - offset[1])), radius, (255, 165, 0), -1)
            #cv2.putText(frame,str(i),(int(x), int(y)),1,2,color,4)
            #i = i+1

    # Draw lines
    for i, j in skeleton:
        if i < len(keypoints) and j < len(keypoints):
            x1, y1 = keypoints[i]
            x2, y2 = keypoints[j]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(frame,(zoom *(int(x1)- offset[0]), zoom *(int(y1)- offset[1])), (zoom *(int(x2)- offset[0]), zoom *(int(y2)- offset[1])), color, thickness)

    return frame

def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    first_player_idx = None
    first_player_track_id = None
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        if first_player_idx is None:
            if detections.tracker_id.shape[0] != 0:            
                first_player_idx = np.where(detections.tracker_id == detections.tracker_id[0])[0]
                first_player_track_id = detections.tracker_id[0]
        else:
            if not (first_player_track_id in detections.tracker_id):
                first_player_idx = np.where(detections.tracker_id == detections.tracker_id[0])[0]
                first_player_track_id = detections.tracker_id[0]
        
        if first_player_track_id is not None:
            first_player_idx = np.where(detections.tracker_id == first_player_track_id)[0]
                
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        
        annotated_frame = frame.copy()

        zoom_overlay_done = False  # To handle only one overlay
        zoomed_crop = None
        for i in range(detections.class_id.shape[0]):
            if detections.class_id[i] == 2:  # Assuming class_id 2 is player
                box = detections.xyxy[i]
                x1, y1, x2, y2 = map(int, box)
                w = x2 - x1
                h = y2 - y1

                # Ensure box is inside image bounds
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, frame.shape[1])
                y2 = min(y2, frame.shape[0])
                keypoints = pose_detector.detect_pose(frame, [x1, y1, w, h])
                if first_player_idx is not None and not zoom_overlay_done and i == first_player_idx:
                    player_crop = frame[y1:y2, x1:x2].copy()
                    # Resize (zoom in) the cropped player image
                    zoom_factor = 5
                    # Pose estimation on original crop (before draw)
                    
                    zoomed_crop = cv2.resize(player_crop, (w * zoom_factor, h * zoom_factor), interpolation=cv2.INTER_LINEAR)
                    zoomed_crop = draw_skeleton(zoomed_crop, keypoints, color=(255, 0, 0), thickness=2,offset = (x1,y1), zoom = zoom_factor,radius=5)
                    zh, zw = zoomed_crop.shape[:2]
                    frame_h, frame_w = annotated_frame.shape[:2]
                    # Make sure it fits the frame
                    zoom_overlay_done = True

                # Draw skeleton on the full frame too
                annotated_frame = draw_skeleton(annotated_frame, keypoints, color=(255, 0, 0), thickness=1,radius=2)

        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections, idx = first_player_idx)
        if zoomed_crop is not None:
            annotated_frame[0:zh, frame_w - zw:frame_w] = zoomed_crop
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    sorted_players_tracks = np.array([])
    sorted_players_teams_id = np.array([])
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        
        sorted_by_tracks = np.argsort(players.tracker_id)
        if np.array_equal(sorted_players_tracks, np.sort(players.tracker_id)):
            players_team_id = sorted_players_teams_id.copy()
            players_team_id[sorted_by_tracks] = sorted_players_teams_id.copy()
            
        else:
            crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(crops)
            sorted_players_tracks = np.sort(players.tracker_id)
            sorted_players_teams_id = players_team_id[sorted_by_tracks]        
        
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        # annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
        #     annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_all(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        #overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )
    
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        ball_detections = slicer(frame).with_nms(threshold=0.1)
        ball_detections = ball_tracker.update(ball_detections)

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)
        annotated_frame = ball_annotator.annotate(annotated_frame, ball_detections)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)

        if radar is None:
            yield annotated_frame
        else:
            # Resize radar to match the height of the original frame
            radar_h = h
            radar_w = int(radar.shape[1] * (h / radar.shape[0]))
            radar_resized = cv2.resize(radar, (radar_w, radar_h))

            # Optionally pad if needed to match heights exactly
            if radar_resized.shape[0] != h:
                pad_h = h - radar_resized.shape[0]
                radar_resized = cv2.copyMakeBorder(
                    radar_resized, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

            # Concatenate images side by side
            combined = np.hstack((annotated_frame, radar_resized))            
            combined = sv.resize_image(combined, (annotated_frame.shape[1], h))
            yield combined


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        scaled_width = int(CONFIG.width * 0.1)
        scaled_length = int(CONFIG.length * 0.1)
        new_frame = np.zeros((h + scaled_width, w, 3), dtype=np.uint8)
        new_frame[:h, :, :] = annotated_frame
        if radar is None:
            yield new_frame
        else:
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape       

            # Center radar image horizontally
            radar_x = w // 2 - radar_w // 2
            radar_y = h  # Below the original image

            # Paste radar on the new canvas
            new_frame[radar_y:radar_y + radar_h, radar_x:radar_x + radar_w] = (
                cv2.addWeighted(
                    new_frame[radar_y:radar_y + radar_h, radar_x:radar_x + radar_w],
                    1 - 0.5,
                    radar,
                    0.5,
                    0
                )
            )
            yield new_frame

def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.ALL:
        frame_generator = run_all(
            source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)

    # Create a resizable window BEFORE the loop
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()