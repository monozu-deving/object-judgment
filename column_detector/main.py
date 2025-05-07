import cv2
from detection.detector import detect_objects
from detection.utils import assign_color, draw_angle_line
from capture.saver import init_capture_dir, open_csv, save_frame_and_log

def main():
    # 저장 폴더 및 CSV 초기화
    init_capture_dir()
    csv_file, csv_writer = open_csv()

    cap = cv2.VideoCapture(0)
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 객체 감지 + 프레임 중심 좌표
        objects_info, (cx_frame, cy_frame) = detect_objects(frame)

        # 중심 기준 객체 선택
        ref_obj = min(objects_info, key=lambda o: o['dist']) if objects_info else None

        for obj in objects_info:
            box = obj['box']
            cx = obj['cx']
            cy = obj['cy']
            angle = obj['angle']
            dx = obj['dx']
            dy = obj['dy']

            color = assign_color(obj, ref_obj)

            # 바운딩 박스와 상대 좌표는 모두 시각화
            cv2.drawContours(frame, [box], 0, color, 2)
            cv2.putText(frame, f"dx:{dx} dy:{dy}", (int(cx), int(cy) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 기준 객체일 때만 각도 및 회전선 시각화
            if obj is ref_obj:
                cv2.putText(frame, f"Angle:{angle:.1f}", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                draw_angle_line(frame, cx, cy, angle, color)

        # 중심 기준 십자선 표시
        cv2.line(frame, (cx_frame, 0), (cx_frame, frame.shape[0]), (120, 120, 120), 1)
        cv2.line(frame, (0, cy_frame), (frame.shape[1], cy_frame), (120, 120, 120), 1)

        # 프레임 표시
        cv2.imshow("YOLOv8 Seg + Angle View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and objects_info:
            save_frame_and_log(frame, objects_info, csv_writer, saved_count)
            saved_count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

if __name__ == "__main__":
    main()
