from ultralytics import YOLO
import cv2
import time
from collections import deque

model = YOLO('yolov8n.pt')

# Only detect relevant classes (COCO class IDs)
CLASSES = {
    0:  'person',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    5:  'bus',
    7:  'truck',
    9:  'traffic light',
    11: 'stop sign',
}

WINDOW_NAME = 'YOLOv8 Detection'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.createTrackbar('Width',  WINDOW_NAME, 640,  1920, lambda x: None)
cv2.createTrackbar('imgsz',  WINDOW_NAME, 640,  1280, lambda x: None)  # live inference size
cv2.createTrackbar('Conf %', WINDOW_NAME, 35,   95,   lambda x: None)  # conf as integer 0–95

# Smoothed FPS using a rolling average (less flickery than instantaneous)
fps_buffer = deque(maxlen=30)
prev_time  = time.time()

for result in model(
    source=0,
    stream=True,
    conf=0.35,
    imgsz=640,                        # start conservative, adjust via trackbar
    classes=list(CLASSES.keys()),     # ignore all other COCO classes
):
    frame = result.plot()

    # --- Smoothed FPS ---
    curr_time = time.time()
    fps_buffer.append(1.0 / (curr_time - prev_time + 1e-9))
    prev_time = curr_time
    avg_fps = sum(fps_buffer) / len(fps_buffer)

    cv2.putText(
        frame, f'{avg_fps:.1f} FPS',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2, cv2.LINE_AA
    )

    # --- Aspect-ratio-preserving resize ---
    display_width  = max(cv2.getTrackbarPos('Width', WINDOW_NAME), 100)
    h, w           = frame.shape[:2]
    display_height = int(display_width * (h / w))
    small          = cv2.resize(frame, (display_width, display_height))

    cv2.imshow(WINDOW_NAME, small)

    # --- Apply trackbar values to next inference call ---
    new_imgsz = cv2.getTrackbarPos('imgsz', WINDOW_NAME)
    new_imgsz = max(new_imgsz - (new_imgsz % 32), 32)  # must be multiple of 32
    new_conf  = cv2.getTrackbarPos('Conf %', WINDOW_NAME) / 100.0

    model.predictor.args.imgsz = new_imgsz
    model.predictor.args.conf  = new_conf

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()