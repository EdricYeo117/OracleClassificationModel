# human_analyzer_webcam_professional.py
# PROFESSIONAL FREE HUMAN ANALYZER (fixed)
# - Face detection (MediaPipe)
# - Age & Gender (FairFace ResNet34 checkpoint with 18 logits)
# - Height estimation (pose-based, single-person)
# - Distance estimation (approx, face bbox width)
# - Simple person ID tracking
# - CSV logging
# - CPU/GPU support

import cv2
import mediapipe as mp
import torch
import numpy as np
import pandas as pd
import time
import os

# -----------------------
# SETTINGS
# -----------------------
REFERENCE_HEIGHT_CM = 170
KNOWN_FACE_WIDTH_CM = 16
FOCAL_LENGTH = 600
CSV_FILE = "professional_results.csv"
FAIRFACE_CKPT = "fairface_alldata_4race_20191111.pt"  # put this file in the same folder

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
import torchvision.models as models

model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 18)

if not os.path.exists(FAIRFACE_CKPT):
    raise FileNotFoundError(
        f"Missing FairFace checkpoint: {FAIRFACE_CKPT}\n"
        f"Place it in the same folder as this script: {os.getcwd()}"
    )

state_dict = torch.load(FAIRFACE_CKPT, map_location=device)

# Some checkpoints may store {"state_dict": ...}
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# If keys are prefixed (e.g. "module."), strip them
if isinstance(state_dict, dict):
    fixed = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        fixed[nk] = v
    state_dict = fixed

model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

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

def predict_age_gender(face_bgr: np.ndarray):
    # BGR -> RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    face_rgb = face_rgb.astype(np.float32) / 255.0

    x = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)          # [1, 18]
        logits = logits[0]         # [18]

    # Slice logits
    gender_logits = logits[7:9]   # 2
    age_logits = logits[9:18]     # 9

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

def save_csv_row(row: dict):
    write_header = not os.path.exists(CSV_FILE)
    df = pd.DataFrame([row])
    df.to_csv(CSV_FILE, mode="a", header=write_header, index=False)

# -----------------------
# WEBCAM
# -----------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (VideoCapture(0)). Check camera permissions / device.")

print("Professional Analyzer Running... Press Q")

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
            age, gender = predict_age_gender(face_crop)
        except Exception:
            age, gender = "?", "?"
        distance_cm = estimate_distance(w)

        text = f"ID:{pid} {gender} {age}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, max(20, y - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        htxt = "?" if height_cm is None else f"{height_cm}"
        cv2.putText(
            frame,
            f"H:{htxt}cm D:{distance_cm}cm",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        save_csv_row({
            "time": time.time(),
            "id": pid,
            "gender": gender,
            "age": age,
            "height_cm": height_cm,
            "distance_cm": distance_cm
        })

    cv2.imshow("PROFESSIONAL HUMAN ANALYZER", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()