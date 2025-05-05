import cv2
import time
from ultralytics import YOLO

# 사전학습된 YOLOv8 모델 로드 (기둥 탐지가 아니라 일반 객체용)
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

last_time = 0
interval_sec = 2  # 2초마다 한 번 탐지

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_time >= interval_sec:
        last_time = now

        results = model.predict(frame, imgsz=640, conf=0.5)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            print(f"Detected object size: {w}px x {h}px")

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{w}x{h}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 프레임 표시
    cv2.imshow("Object Detection (Every 2s)", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
