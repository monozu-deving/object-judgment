import cv2
import time
import csv
import os
import shutil
from ultralytics import YOLO

# ▶ 모델 로드
model = YOLO("yolov8n.pt")

# ▶ 캡처 폴더 초기화
if os.path.exists("captures"):
    shutil.rmtree("captures")
os.makedirs("captures", exist_ok=True)

# ▶ CSV 로그 파일 생성
csv_file = open("captures/detection_log.csv", mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["filename", "x1", "y1", "x2", "y2", "width", "height", "dx", "dy"])

# ▶ 웹캠 실행
cap = cv2.VideoCapture(0)
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    cx_frame, cy_frame = w_frame // 2, h_frame // 2  # 중심 좌표 (0,0 기준)

    # 예측 수행
    results = model.predict(frame, imgsz=640, conf=0.5)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1

        # 객체 중심 좌표
        cx_obj = (x1 + x2) // 2
        cy_obj = (y1 + y2) // 2

        # 중심 기준 상대 좌표 (dx, dy)
        dx = cx_obj - cx_frame
        dy = cy_obj - cy_frame

        # 시각화
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{w}x{h}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"dx={dx}, dy={dy}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 중심선 표시
    cv2.line(frame, (cx_frame, 0), (cx_frame, h_frame), (100, 100, 100), 1)
    cv2.line(frame, (0, cy_frame), (w_frame, cy_frame), (100, 100, 100), 1)

    # 화면 출력
    cv2.imshow("Object Detection with Relative Coordinates", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if boxes:
            filename = f"captures/capture_{saved_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[✓] 저장됨: {filename}")

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                cx_obj = (x1 + x2) // 2
                cy_obj = (y1 + y2) // 2
                dx = cx_obj - cx_frame
                dy = cy_obj - cy_frame
                csv_writer.writerow([
                    os.path.basename(filename), x1, y1, x2, y2, w, h, dx, dy
                ])

            saved_count += 1
        else:
            print("[!] 객체가 감지되지 않았습니다. 저장 안 됨.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
