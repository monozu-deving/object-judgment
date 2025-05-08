import cv2
from detection.midas_filter import refine_yolo_with_midas
from detection.cvdetector import detect_objects
from detection.utils import assign_color, draw_angle_line
from detection.midas_depth import estimate_depth_map, normalize_depth_map
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

        # MiDaS depth 예측
        depth_image = estimate_depth_map(frame)
        

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

            depth_map_raw = estimate_depth_map(frame)
            depth_map = normalize_depth_map(depth_map_raw)  # 0~1

            objects_info = refine_yolo_with_midas(objects_info, depth_map)

        # 중심 기준 십자선 표시
        cv2.line(frame, (cx_frame, 0), (cx_frame, frame.shape[0]), (120, 120, 120), 1)
        cv2.line(frame, (0, cy_frame), (frame.shape[1], cy_frame), (120, 120, 120), 1)
        cv2.putText(frame, f"Detected Objects: {len(objects_info)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 두 영상 나란히 보여주기
        frame_resized = cv2.resize(frame, (640, 480))
        depth_resized = cv2.resize(depth_image, (640, 480))

        # 타입 맞추기: frame은 BGR (uint8, 3채널), depth는 아마 Gray나 컬러맵 (uint8)
        if frame_resized.dtype != depth_resized.dtype:
            depth_resized = depth_resized.astype(frame_resized.dtype)

        if len(depth_resized.shape) == 2:  # Grayscale이면 BGR로 변환
            depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)

        # 최종 연결
        combined = cv2.hconcat([frame_resized, depth_resized])
        cv2.imshow("YOLO + MiDaS", combined)
        

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
