import torch
import cv2

model = torch.hub.load('.', 'custom', source='local', path='yolov5s.pt')
cap = cv2.VideoCapture(0)
assert cap.isOpened()
try:
    while True:
        res, frame = cap.read()
        if not res:
            print('Warning: could not read frame')
            continue
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size=640)
        results.print()
finally:
    cap.release()
