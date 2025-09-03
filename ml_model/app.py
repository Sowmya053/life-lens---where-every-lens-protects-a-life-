from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("yolov8n.pt")

HTML_PAGE = """
<html>
<head><title>YOLO Collision Detection</title></head>
<body>
    <h2>{{ status }}</h2>
    <img src="{{ url_for('video_feed') }}">
</body>
</html>
"""

def detect_collision(frame):
    results = model(frame)[0]
    humans = 0
    cars = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            humans += 1
        if model.names[cls] == "car":
            xyxy = box.xyxy[0].cpu().numpy()
            # Calculate center of car box
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            cars.append((x_center, y_center, xyxy))
    # Check for collision: two cars close together
    collision = False
    min_distance = 100  # pixels, adjust as needed
    if len(cars) >= 2:
        for i in range(len(cars)):
            for j in range(i+1, len(cars)):
                x1, y1, _ = cars[i]
                x2, y2, _ = cars[j]
                distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                if distance < min_distance:
                    collision = True
    # Prioritize collision message
    if collision:
        status = "Collision detected!"
    elif humans > 0:
        status = "No collision: Humans detected."
    else:
        status = "No collision detected."
    return status, results

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        status, results = detect_collision(frame)
        annotated_frame = results.plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    cap.release()
    if success:
        status, _ = detect_collision(frame)
    else:
        status = "Camera not available."
    return render_template('index.html', status=status)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)