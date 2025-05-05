import cv2
from ultralytics import YOLO

# 학습된 기둥 탐지 모델 불러오기 (파일명은 반드시 존재해야 함)
model = YOLO("best.pt")  # 나중에 학습한 기둥 탐지 모델로 바꿔야 함

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.5)
    annotated = results[0].plot()

    cv2.imshow("Column Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
