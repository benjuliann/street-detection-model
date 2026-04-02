from ultralytics import YOLO
import cv2
import time
from collections import deque

model = YOLO('best.pt')

# Only detect relevant classes (COCO class IDs)
CLASSES = {
    # COCO
    0:  'person',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    4:  'bus',
    5:  'truck',
    6:  'traffic light',
    7:  'stop sign',
    # Canadian signs (8–135)
    8:  'Airport',
    9:  'All-Movements-Permitted',
    10: 'Bicycle-Crossing',
    11: 'Bicycle-Route',
    12: 'Bicycles-Excepted',
    13: 'Bus-Station',
    14: 'Buses-Excepted',
    15: 'Carpool-Parking',
    16: 'Cattle-Crossing',
    17: 'Checkerboard',
    18: 'Chevron-Arrow',
    19: 'Construction-Ahead',
    20: 'Construction-Begins',
    21: 'Construction-Ends',
    22: 'Cross-At-Crossover',
    23: 'Cross-On-Green',
    24: 'Cross-On-Signal',
    25: 'Cross-Other-Side',
    26: 'Deer-Crossing',
    27: 'Detour-Ahead',
    28: 'Detour-Marker',
    29: 'Divided-Road-Begins',
    30: 'Divided-Road-Ends',
    31: 'Do-Not-Enter',
    32: 'Do-Not-Turn-Left',
    33: 'Do-Not-Turn-Right',
    34: 'Drawbridge',
    35: 'Except-Authorized-Vehicles',
    36: 'Fallen-Rock',
    37: 'Ferry',
    38: 'Fire-Hydrant',
    39: 'Firetruck-Entrance',
    40: 'HOV',
    41: 'Hiker-Crossing',
    42: 'Horse-Drawn-Vehicle-Crossing',
    43: 'Horseback-Crossing',
    44: 'Hospital',
    45: 'Intersection',
    46: 'Island-Split',
    47: 'Keep-Right',
    48: 'Keep-Right-Of-Island',
    49: 'Lane-Closure',
    50: 'Left-Lane-Ends',
    51: 'Left-Turn-Signal',
    52: 'Left-or-Right-Turn-Only',
    53: 'Low-Clearance',
    54: 'Moose-Crossing',
    55: 'Narrow-Bridge',
    56: 'No-Bicycles',
    57: 'No-Left-On-Red',
    58: 'No-Parking',
    59: 'No-Passing',
    60: 'No-Passing-Bicycles',
    61: 'No-Pedestrians',
    62: 'No-Pedestrians-Or-Bicycles',
    63: 'No-Right-On-Red',
    64: 'No-Snowmobiles',
    65: 'No-Standing',
    66: 'No-Stopping',
    67: 'No-Straight-Through',
    68: 'No-Tractors',
    69: 'No-Trucks-This-Lane',
    70: 'No-U-Turn',
    71: 'OPP-Station',
    72: 'Object-Marker',
    73: 'One-Way',
    74: 'Parking-Allowed',
    75: 'Pass-With-Care',
    76: 'Passenger-Railway-Station',
    77: 'Passing-Allowed',
    78: 'Paved-Surface-Ends',
    79: 'Pavement-Milled',
    80: 'Pavement-Narrows',
    81: 'Pedestrian-Crossing',
    82: 'Pedestrian-and-Bicycle-Crossing',
    83: 'Playground-Ahead',
    84: 'Private-Road',
    85: 'Public-Telephone',
    86: 'Railway-Crossing',
    87: 'Reverse-Curve',
    88: 'Right-Lane-Ends',
    89: 'Road-Closed',
    90: 'Road-Fork-To-Left',
    91: 'Road-Fork-To-Right',
    92: 'Road-Work',
    93: 'Roundabout-Ahead',
    94: 'School-Bus-Entrance',
    95: 'School-Bus-Loading-Zone',
    96: 'School-Bus-Stop-Ahead',
    97: 'School-Crossing',
    98: 'Share-Road',
    99: 'Shared-Pathway',
    100: 'Sharp-Bend',
    101: 'Sharp-Reverse-Curve',
    102: 'Sharp-Turn',
    103: 'Slight-Bend',
    104: 'Slippery-Road',
    105: 'Snowmobile-Crossing',
    106: 'Snowmobile-Route',
    107: 'Speed-Hump',
    108: 'Speed-Limit-10',
    109: 'Speed-Limit-100',
    110: 'Speed-Limit-110',
    111: 'Speed-Limit-20',
    112: 'Speed-Limit-30',
    113: 'Speed-Limit-40',
    114: 'Speed-Limit-50',
    115: 'Speed-Limit-60',
    116: 'Speed-Limit-70',
    117: 'Speed-Limit-80',
    118: 'Speed-Limit-90',
    119: 'Steep-Hill',
    120: 'Stop',
    121: 'Stop-Ahead',
    122: 'Stop-For-Pedestrians',
    123: 'Stop-For-School-Bus',
    124: 'Temporary-Merge',
    125: 'Traffic-Control-Person',
    126: 'Traffic-Light-Ahead',
    127: 'Truck-Entrance',
    128: 'Truck-Route',
    129: 'Two-way-Left-Turn',
    130: 'U-Turn',
    131: 'Uneven-Pavement',
    132: 'Water-Over-Road',
    133: 'Winding-Road',
    134: 'Yield',
    135: 'Yield-Ahead',
}

WINDOW_NAME = 'YOLO11 Upgraded Detection'
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