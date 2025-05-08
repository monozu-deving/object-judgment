import cv2
import numpy as np
from ultralytics import YOLO
from configs.settings import CONFIDENCE_THRESHOLD, MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_objects(frame):
    results = model.predict(frame, imgsz=640, conf=CONFIDENCE_THRESHOLD)
    result = results[0]
    h_frame, w_frame = frame.shape[:2]
    cx_frame, cy_frame = w_frame // 2, h_frame // 2

    objects = []
    if hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for mask in masks:
            mask_bin = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 5:
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), (w, h), angle = rect
                    dx = int(cx - cx_frame)
                    dy = int(cy - cy_frame)
                    dist = abs(dx) + abs(dy)
                    box = cv2.boxPoints(rect).astype(np.int32)
                    objects.append({
                        'box': box, 'cx': cx, 'cy': cy,
                        'w': w, 'h': h, 'angle': angle,
                        'dx': dx, 'dy': dy, 'dist': dist
                    })
    return objects, (cx_frame, cy_frame)
