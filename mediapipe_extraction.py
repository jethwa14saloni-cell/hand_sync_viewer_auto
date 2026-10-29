"""Video to pipeline-compatible XLSX; inference runs locally."""
from pathlib import Path
import math
import cv2
import numpy as np
import pandas as pd

NAMES = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
         'index_mcp', 'index_pip', 'index_dip', 'index_tip',
         'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
         'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
         'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
         (15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]

MAX_HANDS = 4
MIN_CONFIDENCE = 0.3
MAX_JUMP = 0.25


class Cancelled(Exception):
    pass


def _pick_hand(found, previous):
    if previous is not None:
        def distance(c):
            wrist = c[1][0]
            return math.hypot(wrist.x - previous[0], wrist.y - previous[1])
        near = [c for c in found if distance(c) <= MAX_JUMP]
        if near:
            return min(near, key=distance)
    return max(found, key=lambda c: c[0])


def extract_video(video_path, model_path, output_path, swap=False,
                  progress=lambda *args: None, cancelled=lambda: False):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    if not Path(model_path).is_file():
        raise ValueError('Select a valid hand_landmarker.task model file.')
    cap = cv2.VideoCapture(str(video_path))
    rows, overlay = [], []
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no valid FPS; convert to constant frame rate first.')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_CONFIDENCE,
            min_hand_presence_confidence=MIN_CONFIDENCE,
            min_tracking_confidence=MIN_CONFIDENCE)
        previous_ms = -1
        last_wrist = {}
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                if cancelled():
                    raise Cancelled()
                ok, frame = cap.read()
                if not ok:
                    break
                i = len(rows)
                seconds = i / fps
                timestamp = max(previous_ms + 1, round(seconds * 1000))
                previous_ms = timestamp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                row = {'frame_index': i, 'time_seconds': seconds}
                frame_points = {}
                for side in ('left','right'):
                    row[f'{side}_hand_detected'] = False
                    row[f'{side}_hand_score'] = np.nan
                    for name in NAMES:
                        for axis in 'xyz':
                            row[f'{side}_hand_{name}_{axis}'] = np.nan
                candidates = {'left': [], 'right': []}
                for categories, image_pts, world_pts in zip(
                        result.handedness, result.hand_landmarks, result.hand_world_landmarks):
                    category = categories[0]
                    side = category.category_name.lower()
                    if side not in candidates:
                        continue
                    if swap:
                        side = 'right' if side == 'left' else 'left'
                    candidates[side].append((category.score, image_pts, world_pts))
                for side, found in candidates.items():
                    if not found:
                        continue
                    chosen = _pick_hand(found, last_wrist.get(side))
                    score, image_pts, world_pts = chosen
                    last_wrist[side] = (image_pts[0].x, image_pts[0].y)
                    row[f'{side}_hand_detected'] = True
                    row[f'{side}_hand_score'] = score
                    for name, point in zip(NAMES, world_pts):
                        for axis in 'xyz':
                            row[f'{side}_hand_{name}_{axis}'] = getattr(point, axis)
                    frame_points[side] = [(p.x,p.y) for p in image_pts]
                rows.append(row)
                overlay.append(frame_points)
                if i % 10 == 0:
                    progress(i+1, total)
        if not rows:
            raise ValueError('Video contains no readable frames.')
        if cancelled():
            raise Cancelled()
        progress(len(rows), total)
        output = Path(output_path)
        temporary = output.with_name(output.stem + '.partial.xlsx')
        try:
            with pd.ExcelWriter(temporary) as writer:
                pd.DataFrame(rows).to_excel(writer, index=False, sheet_name='landmarks')
                pd.DataFrame([{'video': str(Path(video_path).resolve()), 'fps': fps,
                    'coordinate_system': 'MediaPipe world landmarks, meters',
                    'swap_hands': swap, 'timebase': 'frame_index / fps (CFR assumed)'}]).to_excel(
                        writer, index=False, sheet_name='metadata')
            if cancelled():
                raise Cancelled()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return str(output), fps, overlay
    finally:
        cap.release()


def draw_overlay(frame, points):
    image = frame.copy()
    height, width = image.shape[:2]
    for side, landmarks in points.items():
        color = (70,210,100) if side == 'right' else (230,170,70)
        pixels = [(int(x*width),int(y*height)) for x,y in landmarks]
        for a,b in EDGES:
            cv2.line(image,pixels[a],pixels[b],color,2)
        for point in pixels:
            cv2.circle(image,point,3,color,-1)
        cv2.putText(image,side,pixels[0],cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
    return image


def extraction_worker(video, model, output, swap, messages, cancel):
    """Spawn entry point; never touches Tk or Matplotlib."""
    import traceback
    try:
        result = extract_video(video, model, output, swap=swap,
            progress=lambda n,total: messages.put(("progress", (n,total))),
            cancelled=cancel.is_set)
        messages.put(("done", result))
    except Cancelled:
        messages.put(("cancelled", None))
    except Exception:
        messages.put(("error", traceback.format_exc()))