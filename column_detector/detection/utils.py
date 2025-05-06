def assign_color(obj, ref_obj):
    if obj is ref_obj:
        return (0, 255, 0)  # 기준 → 초록
    elif obj['cx'] < ref_obj['cx']:
        return (255, 0, 0)  # 왼쪽 → 파랑
    else:
        return (0, 0, 255)  # 오른쪽 → 빨강
