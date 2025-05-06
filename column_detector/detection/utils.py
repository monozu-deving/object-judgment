import cv2
import math

def draw_angle_line(frame, cx, cy, angle, color, length=40):
    """
    중심 좌표(cx, cy)를 기준으로 angle(도) 방향의 직선을 그린다.
    """
    rad = math.radians(angle)
    dx = int(length * math.cos(rad))
    dy = int(length * math.sin(rad))

    pt1 = (int(cx - dx), int(cy - dy))
    pt2 = (int(cx + dx), int(cy + dy))

    cv2.line(frame, pt1, pt2, color, 2)

def assign_color(obj, ref_obj):
    if obj is ref_obj:
        return (0, 255, 0)  # 기준 → 초록
    elif obj['cx'] < ref_obj['cx']:
        return (255, 0, 0)  # 왼쪽 → 파랑
    else:
        return (0, 0, 255)  # 오른쪽 → 빨강
