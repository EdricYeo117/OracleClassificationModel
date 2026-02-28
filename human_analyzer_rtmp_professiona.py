# human_analyzer_rtmp_professional.py
# PROFESSIONAL FREE HUMAN ANALYZER (RTMP)
# - RTMP livestream input (PyAV/FFmpeg)
# - Face detection (MediaPipe)
# - Age & Gender (FairFace ResNet34 checkpoint with 18 logits)
# - Height estimation (pose-based, single-person)
# - Distance estimation (approx, face bbox width)
# - Simple person ID tracking
# - CSV logging
# - CPU/GPU support
#
# Usage:
#   .\.venv\Scripts\python.exe human_analyzer_rtmp_professional.py --rtmp rtmp://192.168.1.49/live/android-controller-01
#
# Press Q to quit.

import os
import time
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
import mediapipe as mp
import torchvision.models as models

import av  # PyAV (FFmpeg bindings)


# -----------------------
# SETTINGS (tweak as needed)
# -----------------------
REFERENCE_HEIGHT_CM = 170
KNOWN_FACE_WIDTH_CM = 16
FOCAL_LENGTH = 600

DEFAULT_CSV_FILE = "professional_results.csv"
DEFAULT_FAIRFACE_CKPT = "fairface_alldata_4race_20191111.pt"

AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


# -----------------------
# DEVICE
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -----------------------
# MEDIAPIPE
# -----------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

mp_draw = mp.solutions.drawing_utils


# -----------------------
# FAIRFACE MODEL (ResNet34 + 18 outputs)
# Layout assumed: [7 race | 2 gender | 9 age] = 18 logits
# -----------------------
def load_fairface_model(ckpt_path: str) -> torch.nn.Module:
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 18)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Missing FairFace checkpoint: {ckpt_path}\n"
            f"Place it in the same folder as this script, or pass --ckpt.\n"
            f"Current working dir: {os.getcwd()}"
        )

    state_dict = torch.load(ckpt_path, map_location=device)

    # Some checkpoints store {"state_dict": ...}
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip "module." prefix if present
    if isinstance(state_dict, dict):
        fixed = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "")
            fixed[nk] = v
        state_dict = fixed

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# -----------------------
# TRACKING (simple centroid matching)
# -----------------------
person_id = 0
tracked = {}  # pid -> (x, y)


def get_person_id(x: int, y: int) -> int:
    global person_id
    for pid, (px, py) in tracked.items():
        if abs(x - px) < 50 and abs(y - py) < 50:
            tracked[pid] = (x, y)
            return pid
    person_id += 1
    tracked[person_id] = (x, y)
    return person_id


# -----------------------
# HELPERS
# -----------------------
def clamp_bbox(x, y, w, h, img_w, img_h):
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    return x, y, w, h


def predict_age_gender(model: torch.nn.Module, face_bgr: np.ndarray):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    face_rgb = face_rgb.astype(np.float32) / 255.0

    x = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)[0]  # [18]

    gender_logits = logits[7:9]     # 2
    age_logits = logits[9:18]       # 9

    gender_idx = int(torch.argmax(gender_logits).item())
    age_idx = int(torch.argmax(age_logits).item())

    gender = "Male" if gender_idx == 0 else "Female"
    age = AGES[age_idx]
    return age, gender


def estimate_height(pose_landmarks, frame_h: int) -> float:
    # Very rough: uses head landmark 0 and ankle landmark 28
    head = pose_landmarks.landmark[0].y * frame_h
    foot = pose_landmarks.landmark[28].y * frame_h
    pixel = abs(foot - head)
    return round((pixel / frame_h) * REFERENCE_HEIGHT_CM, 1)


def estimate_distance(face_width_pixels: int) -> float:
    face_width_pixels = max(1, int(face_width_pixels))
    return round((KNOWN_FACE_WIDTH_CM * FOCAL_LENGTH) / face_width_pixels, 1)


def save_csv_row(csv_file: str, row: dict):
    write_header = not os.path.exists(csv_file)
    df = pd.DataFrame([row])
    df.to_csv(csv_file, mode="a", header=write_header, index=False)


# -----------------------
# RTMP FRAME SOURCE (PyAV)
# -----------------------
def frames_from_rtmp(url: str, low_latency: bool = True):
    """
    Yields BGR frames (numpy arrays) from an RTMP URL.

    low_latency=True tries to reduce buffering, but may increase frame drops on bad networks.
    """
    options = {}
    if low_latency:
        # These map to FFmpeg demuxer options. Not all builds honor all options.
        options = {
            "fflags": "nobuffer",
            "flags": "low_delay",
            "rtmp_live": "live",
        }

    container = av.open(url, options=options)

    # Choose first video stream
    vstream = None
    for s in container.streams:
        if s.type == "video":
            vstream = s
            break
    if vstream is None:
        raise RuntimeError(f"No video stream found in RTMP source: {url}")

    vstream.thread_type = "AUTO"

    for packet in container.demux(vstream):
        for frame in packet.decode():
            img = frame.to_ndarray(format="bgr24")
            yield img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtmp", required=True, help="RTMP URL, e.g. rtmp://192.168.1.49/live/android-controller-01")
    parser.add_argument("--csv", default=DEFAULT_CSV_FILE, help="CSV output file")
    parser.add_argument("--ckpt", default=DEFAULT_FAIRFACE_CKPT, help="FairFace checkpoint .pt path")
    parser.add_argument("--no-low-latency", action="store_true", help="Disable low-latency demux options")
    parser.add_argument("--reconnect", action="store_true", help="Reconnect if stream drops")
    parser.add_argument("--reconnect-wait", type=float, default=2.0, help="Seconds to wait before reconnect")
    args = parser.parse_args()

    model = load_fairface_model(args.ckpt)

    print("Professional Analyzer (RTMP) Running...")
    print("RTMP:", args.rtmp)
    print("Press Q")

    def run_once():
        for frame in frames_from_rtmp(args.rtmp, low_latency=(not args.no_low_latency)):
            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face detection
            face_results = face_detection.process(rgb)
            faces = []
            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * frame_w)
                    y = int(bbox.ymin * frame_h)
                    w = int(bbox.width * frame_w)
                    h = int(bbox.height * frame_h)
                    x, y, w, h = clamp_bbox(x, y, w, h, frame_w, frame_h)
                    faces.append((x, y, w, h))

            # Pose (single-person)
            pose_result = pose.process(rgb)
            height_cm = None
            if pose_result.pose_landmarks:
                mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                height_cm = estimate_height(pose_result.pose_landmarks, frame_h)

            # Process faces
            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                if face_crop.size == 0:
                    continue

                pid = get_person_id(x, y)

                try:
                    age, gender = predict_age_gender(model, face_crop)
                except Exception:
                    age, gender = "?", "?"

                distance_cm = estimate_distance(w)

                # Draw overlays
                text = f"ID:{pid} {gender} {age}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, max(20, y - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                htxt = "?" if height_cm is None else f"{height_cm}"
                cv2.putText(frame, f"H:{htxt}cm D:{distance_cm}cm", (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Log CSV
                save_csv_row(args.csv, {
                    "time": time.time(),
                    "id": pid,
                    "gender": gender,
                    "age": age,
                    "height_cm": height_cm,
                    "distance_cm": distance_cm,
                })

            cv2.imshow("PROFESSIONAL HUMAN ANALYZER (RTMP)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False  # stop
        return True  # stream ended

    try:
        if args.reconnect:
            while True:
                try:
                    ended = run_once()
                    if not ended:
                        break
                    print("Stream ended. Reconnecting...")
                except Exception as e:
                    print(f"[RTMP] error: {e}")
                time.sleep(args.reconnect_wait)
        else:
            run_once()
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()