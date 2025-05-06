import os
import cv2
import csv
from configs.settings import CAPTURE_DIR, CSV_NAME

def init_capture_dir():
    if os.path.exists(CAPTURE_DIR):
        import shutil
        shutil.rmtree(CAPTURE_DIR)
    os.makedirs(CAPTURE_DIR, exist_ok=True)

def open_csv():
    csv_path = os.path.join(CAPTURE_DIR, CSV_NAME)
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["filename", "x", "y", "width", "height", "angle", "dx", "dy"])
    return csv_file, writer

def save_frame_and_log(frame, objects, csv_writer, count):
    filename = f"capture_{count}.jpg"
    filepath = os.path.join(CAPTURE_DIR, filename)
    cv2.imwrite(filepath, frame)
    for obj in objects:
        csv_writer.writerow([
            filename,
            int(obj['cx']), int(obj['cy']),
            int(obj['w']), int(obj['h']),
            round(obj['angle'], 2),
            obj['dx'], obj['dy']
        ])
