from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

#results = model(source='https://www.youtube.com/watch?v=X8dVS1z-m58', show=True, conf=0.4, save=True)  # predict on an image

for result in model(source='https://www.youtube.com/watch?v=X8dVS1z-m58', stream=True, conf=0.4, save=True):
    frame = result.plot()                          # get annotated frame
    small = cv2.resize(frame, (640, 360))          # resize display only (width, height)
    cv2.imshow('YOLOv8 Detection', small)
    if cv2.waitKey(1) & 0xFF == ord('q'):          # press 'q' to quit
        break

cv2.destroyAllWindows()