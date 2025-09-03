# app.py
import os
import cv2
import uuid
import threading
import webbrowser
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
from datetime import timedelta

# ---------- config ----------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO + detection params (tweak if needed)
IOU_THRESHOLD = 0.35        # overlap threshold => collision
SPEED_DROP_THRESHOLD = 2.5  # pixels/frame sudden drop threshold (tweak for your videos)
CONFIDENCE = 0.25           # YOLO confidence threshold
FRAME_MAX = None            # None => full video, or set to small int for quick tests

# Vehicle COCO class ids (COCO: 2=car,3=motorbike,5=bus,7=truck)
VEHICLE_CLASSES = {2, 3, 5, 7}

# ---------- app ----------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model once
print("Loading YOLO model (this may take a few seconds)...")
model = YOLO("yolov8n.pt")
print("YOLO loaded.")

# Utility: IoU
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

# Core detector: reads entire video and attempts to find collision
def detect_accident(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"detected": False, "message": "Cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    prev_centers = {}  # map tracked id -> center (we will use simple indexing)
    prev_boxes = {}    # map idx -> box
    # We will use YOLO outputs per frame and simple nearest mapping between frames,
    # since we don't have tracker. This is a simple heuristic good for many crash clips.

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if FRAME_MAX and frame_idx > FRAME_MAX:
                break

            # Run YOLO on the frame
            results = model(frame, imgsz=640, conf=CONFIDENCE, verbose=False)[0]

            # Collect vehicle boxes this frame
            current_boxes = []
            for b in results.boxes:
                try:
                    cls_tensor = b.cls
                    # cls_tensor might be a tensor; extract int robustly:
                    if hasattr(cls_tensor, "cpu"):
                        cls_id = int(cls_tensor.cpu().numpy()[0])
                    else:
                        cls_id = int(cls_tensor[0])
                except Exception:
                    continue
                if cls_id in VEHICLE_CLASSES:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    current_boxes.append((x1, y1, x2, y2))

            # If fewer than 2 vehicles, skip overlap check (no collision among vehicles)
            # But still check speed drop for single vehicle
            # Map current boxes to previous boxes by nearest center (simple)
            current_centers = []
            for b in current_boxes:
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0
                current_centers.append((cx, cy))

            # 1) Check IoU between any pair in current frame (immediate overlap/collision)
            n = len(current_boxes)
            for i in range(n):
                for j in range(i + 1, n):
                    iou = compute_iou(current_boxes[i], current_boxes[j])
                    if iou >= IOU_THRESHOLD:
                        # collision detected
                        t_seconds = frame_idx / fps
                        cap.release()
                        return {
                            "detected": True,
                            "time_seconds": t_seconds,
                            "time_str": str(timedelta(seconds=int(t_seconds))),
                            "reason": "bounding_box_overlap",
                            "iou": round(iou, 3),
                        }

            # 2) Speed drop detection:
            # match current centers to prev_centers by nearest euclidean distance
            # build lists to compare
            for ci, center in enumerate(current_centers):
                # find nearest previous center
                best_prev_idx = None
                best_dist = None
                for pid, pcenter in prev_centers.items():
                    dist = ((center[0] - pcenter[0]) ** 2 + (center[1] - pcenter[1]) ** 2) ** 0.5
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_prev_idx = pid
                if best_prev_idx is not None:
                    # compute previous speed ~ previous dist (we only have one-step speed)
                    prev_center = prev_centers[best_prev_idx]
                    speed = best_dist  # pixels per frame (approx)
                    # If speed is very small compared to previous speed (or below threshold),
                    # and previously it was moving, we can consider a sudden stop
                    # We'll estimate previous speed by comparing with prev_boxes positions if available
                    prev_speed = None
                    if best_prev_idx in prev_boxes:
                        prev_box = prev_boxes[best_prev_idx]
                        # estimate previous center from prev_box
                        prev_cx = (prev_box[0] + prev_box[2]) / 2.0
                        prev_cy = (prev_box[1] + prev_box[3]) / 2.0
                        prev_speed = ((prev_center[0] - prev_cx) ** 2 + (prev_center[1] - prev_cy) ** 2) ** 0.5
                    # If it was previously moving (prev_speed exists and > threshold*1.2)
                    # and now speed is much smaller -> sudden stop
                    if prev_speed is not None and prev_speed > (SPEED_DROP_THRESHOLD * 1.2) and speed < SPEED_DROP_THRESHOLD:
                        t_seconds = frame_idx / fps
                        cap.release()
                        return {
                            "detected": True,
                            "time_seconds": t_seconds,
                            "time_str": str(timedelta(seconds=int(t_seconds))),
                            "reason": "sudden_speed_drop",
                            "prev_speed": round(prev_speed, 2),
                            "curr_speed": round(speed, 2),
                        }

            # update prev centers/boxes for next iteration
            prev_centers = {i: c for i, c in enumerate(current_centers)}
            prev_boxes = {i: b for i, b in enumerate(current_boxes)}

    finally:
        cap.release()

    return {"detected": False, "message": "No accident found in video."}


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # expecting form key 'file'
    if "file" not in request.files:
        return jsonify({"error": True, "message": "No file uploaded (key 'file' missing)."}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": True, "message": "No file selected."}), 400

    # Save with unique name
    filename = f"{uuid.uuid4().hex}_{f.filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(path)

    # Run detector (synchronous). For large videos, this may take time.
    print(f"[UPLOAD] saved to {path} - starting detection")
    result = detect_accident(path)
    print("[UPLOAD] detection result:", result)

    # Build response
    if result.get("detected"):
        response = {
            "error": False,
            "detected": True,
            "time_seconds": result.get("time_seconds"),
            "time_str": result.get("time_str"),
            "reason": result.get("reason"),
            "details": {k: v for k, v in result.items() if k not in ("detected", "time_seconds", "time_str", "reason")},
            "message": f"Accident detected at {result.get('time_str')}.",
            "filename": filename,
        }
    else:
        response = {
            "error": False,
            "detected": False,
            "message": result.get("message", "No accident detected"),
            "filename": filename,
        }

    return jsonify(response)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)



# Auto-open browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(debug=False, host='127.0.0.1',port=5000)