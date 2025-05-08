import numpy as np

def refine_yolo_with_midas(objects, depth_map, variance_thresh=0.1, merge_thresh=0.05, verbose=True):
    """
    MiDaS 기반 YOLO 후처리:
    - depth 편차 큰 객체 제거
    - 유사 depth끼리 그룹화
    - verbose=True → 로그 출력
    """
    initial_count = len(objects)
    kept = []
    dropped = []

    for obj in objects:
        x1 = int(min(pt[0] for pt in obj['box']))
        y1 = int(min(pt[1] for pt in obj['box']))
        x2 = int(max(pt[0] for pt in obj['box']))
        y2 = int(max(pt[1] for pt in obj['box']))

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            dropped.append(("empty ROI", obj))
            continue

        mean_depth = np.mean(roi)
        std_depth = np.std(roi)

        if std_depth < variance_thresh:
            obj['mean_depth'] = mean_depth
            kept.append(obj)
        else:
            dropped.append((f"std={std_depth:.2f} > {variance_thresh}", obj))

    # 그룹핑
    grouped = []
    used = [False] * len(kept)
    merged_count = 0

    for i, obj_i in enumerate(kept):
        if used[i]:
            continue
        group = [obj_i]
        used[i] = True
        for j in range(i + 1, len(kept)):
            if used[j]:
                continue
            if abs(obj_i['mean_depth'] - kept[j]['mean_depth']) < merge_thresh:
                group.append(kept[j])
                used[j] = True
        if len(group) > 1:
            merged_count += len(group) - 1
        merged = max(group, key=lambda o: o['w'] * o['h'])
        grouped.append(merged)

    if verbose:
        print(f"[YOLO Postprocess] initial={initial_count}, kept={len(kept)}, grouped={len(grouped)}")
        for reason, obj in dropped:
            print(f"  - Dropped: {reason} (box w={obj['w']:.1f}, h={obj['h']:.1f})")
        if merged_count > 0:
            print(f"  - Merged {merged_count} objects into {len(grouped)} groups")

    return grouped

