from ultralytics import YOLO
import cv2
import os

model = YOLO('yolov8n.pt')

source = 'assets/stopsign1.png'
test_image_name = os.path.splitext(os.path.basename(source))[0]  # extracts image name

for result in model(source=source, stream=True, conf=0.4, save=True):
    frame = result.plot()
    small = cv2.resize(frame, (640, 360))
    cv2.imshow('YOLOv8 Detection', small)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('e'):
        save_path = f'results/{test_image_name}-res.png'
        cv2.imwrite(save_path, frame)  # saves full-res annotated frame
        print(f'Saved: {save_path}')
        break

cv2.destroyAllWindows()