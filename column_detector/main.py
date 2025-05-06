import cv2
from detection.detector import detect_objects
from detection.utils import assign_color, draw_angle_line
from capture.saver import init_capture_dir, open_csv, save_frame_and_log


def main():
    init_capture_dir()
    csv_file, csv_writer = open_csv()
    cap = cv2.VideoCapture(0)
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects, (cx_frame, cy_frame) = detect_objects(frame)
        ref_obj = min(objects, key=lambda o: o['dist']) if objects else None

        for obj in objects:
            color = assign_color(obj, ref_obj)
            cv2.drawContours(frame, [obj['box']], 0, color, 2)
            cx, cy, dx, dy, angle = obj['cx'], obj['cy'], obj['dx'], obj['dy'], obj['angle']

            cv2.putText(frame, f"Angle:{angle:.1f}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"dx:{dx} dy:{dy}", (int(cx), int(cy)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            draw_angle_line(frame, cx, cy, angle, color)

        cv2.line(frame, (cx_frame, 0), (cx_frame, frame.shape[0]), (120, 120, 120), 1)
        cv2.line(frame, (0, cy_frame), (frame.shape[1], cy_frame), (120, 120, 120), 1)

        cv2.imshow("YOLOv8 Seg Modular", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and objects:
            save_frame_and_log(frame, objects, csv_writer, saved_count)
            saved_count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()
