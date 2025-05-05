import cv2
import time
import csv
import os
from ultralytics import YOLO

# 모델 불러오기
model = YOLO("yolov8n.pt")

# 저장 폴더 생성
os.makedirs("captures", exist_ok=True)

# CSV 파일 준비
csv_file = open("captures/detection_log.csv", mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "x1", "y1", "x2", "y2", "width", "height"])

cap = cv2.VideoCapture(0)
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=640, conf=0.5)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{w}x{h}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Press 's' to save", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 저장 키
        if boxes:
            filename = f"captures/capture_{saved_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[✓] 저장됨: {filename}")

            # 여러 객체가 있을 수 있으니 반복
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                csv_writer.writerow([os.path.basename(filename), x1, y1, x2, y2, w, h])

            saved_count += 1
        else:
            print("[!] 객체가 감지되지 않았습니다. 저장 안 됨.")
    elif key == ord('q'):  # 종료 키
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
