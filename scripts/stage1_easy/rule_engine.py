#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Rough constants under 1024x1024
DD_PIXEL_ESTIMATE = 90
CSME_THRESHOLD_PX = 35
HE_QUADRANT_SEVERE = 20
MA_MAX_DIAMETER_PX = 45


def validate_od_fovea(
    od_center: tuple[int, int],
    od_radius: int,
    fovea_center: tuple[int, int],
    image_size: int = 1024,
    dd_estimate_px: float = DD_PIXEL_ESTIMATE,
) -> bool:
    """
    Sanity check for OD / fovea geometry on preprocessed 1024x1024 fundus.

    Heuristics:
      - centers inside image bounds
      - OD and fovea not unrealistically close/far in DD units
      - OD not too close to image boundary (avoid picking a spurious circle on the edge)
    """
    od_x, od_y = od_center
    fov_x, fov_y = fovea_center
    if not (0 <= od_x < image_size and 0 <= od_y < image_size and 0 <= fov_x < image_size and 0 <= fov_y < image_size):
        return False

    # Keep OD away from hard boundary; true OD is usually well inside the circular field.
    margin = max(8, int(od_radius * 0.3))
    if od_x < margin or od_x > image_size - 1 - margin or od_y < margin or od_y > image_size - 1 - margin:
        return False

    dist_px = float(np.hypot(float(od_x - fov_x), float(od_y - fov_y)))
    dist_dd = dist_px / float(dd_estimate_px)
    # Typical OD–fovea distance is ~2.5DD; allow a wide range for different crops.
    if dist_dd < 1.5 or dist_dd > 5.0:
        return False
    return True


def get_quadrant(cx: int, cy: int, od_center: tuple[int, int], fovea_center: tuple[int, int]) -> str:
    od_x, od_y = od_center
    fov_x, fov_y = fovea_center
    # Determine laterality from OD↔fovea geometry.
    #
    # Convention in image coordinates: x grows to the right.
    # - Right eye: optic disc is nasal (left), fovea is temporal (right) → fov_x > od_x
    # - Left eye:  optic disc is nasal (right), fovea is temporal (left) → fov_x < od_x
    #
    # Use the OD–fovea midline to decide temporal vs nasal for a lesion location.
    mid_x = (od_x + fov_x) / 2.0
    is_right_eye = fov_x > od_x
    if is_right_eye:
        # Temporal is to the right of the OD–fovea midline.
        is_temporal = cx > mid_x
    else:
        # Temporal is to the left of the OD–fovea midline.
        is_temporal = cx < mid_x

    # Superior vs inferior: use the OD–fovea midline in y.
    is_superior = cy < (od_y + fov_y) / 2.0
    if is_temporal and is_superior:
        return "TS"
    if is_temporal and not is_superior:
        return "TI"
    if (not is_temporal) and is_superior:
        return "NS"
    return "NI"


def dist_to_fovea_dd(cx: int, cy: int, fovea_center: tuple[int, int], dd_px: float = DD_PIXEL_ESTIMATE) -> float:
    fx, fy = fovea_center
    dist_px = float(np.hypot(cx - fx, cy - fy))
    return round(dist_px / float(dd_px), 1)


def dist_band(dist_dd: float) -> str:
    if dist_dd <= 1.0:
        return "黄斑区（≤1DD）"
    if dist_dd <= 3.0:
        return "后极部（1-3DD）"
    if dist_dd <= 6.0:
        return "中周部（3-6DD）"
    return "周边部（>6DD）"


@dataclass(frozen=True)
class CCProp:
    area: int
    cx: int
    cy: int
    bbox: tuple[int, int, int, int]  # x1,y1,x2,y2
    major_axis: float
    minor_axis: float


def _connected_components_props(mask: np.ndarray) -> list[CCProp]:
    """
    Fast component stats extraction using connectedComponentsWithStats.

    This intentionally avoids contour tracing / perimeter / ellipse fitting for speed.
    """
    m = (mask > 0).astype(np.uint8)
    if cv2.countNonZero(m) == 0:
        return []
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    props: list[CCProp] = []
    for i in range(1, num):  # 0 is background
        x, y, w, h, area = stats[i].tolist()
        if area <= 0:
            continue
        cx, cy = centroids[i]
        major, minor = float(max(w, h)), float(min(w, h))
        props.append(
            CCProp(
                area=int(area),
                cx=int(round(float(cx))),
                cy=int(round(float(cy))),
                bbox=(int(x), int(y), int(x + w), int(y + h)),
                major_axis=major,
                minor_axis=minor,
            )
        )
    return props


def analyze_ma(mask: np.ndarray, fovea_center: tuple[int, int], od_center: tuple[int, int]) -> tuple[dict, str]:
    props = _connected_components_props(mask)
    if not props:
        return {"present": False}, "未见典型红色点状微动脉瘤，MA（-）。"

    max_area = float((MA_MAX_DIAMETER_PX**2) * np.pi / 4.0)
    props = [p for p in props if p.area < max_area]
    if not props:
        return {"present": False}, "未见明确微动脉瘤，MA（-）。"

    count = len(props)
    diameters = [2.0 * np.sqrt(p.area / np.pi) for p in props]
    d_min, d_max = int(min(diameters)), int(max(diameters))

    quadrant_counts = {"TS": 0, "TI": 0, "NS": 0, "NI": 0}
    dist_dds: list[float] = []
    for p in props:
        q = get_quadrant(p.cx, p.cy, od_center, fovea_center)
        quadrant_counts[q] += 1
        dist_dds.append(dist_to_fovea_dd(p.cx, p.cy, fovea_center))
    main_quad = max(quadrant_counts, key=quadrant_counts.get)
    min_dist = min(dist_dds)
    # MA is typically dot-like; keep wording stable while using fast stats.
    shape_word = "规则圆点"

    desc = (
        f"后极部可见约{count}处红色至暗红色{shape_word}，边界清晰锐利，直径约{d_min}-{d_max}px。"
        f"主要分布于{main_quad}象限，最近病灶距黄斑中心约{min_dist}DD（{dist_band(min_dist)}）。"
        f"边界规则，区别于点状出血的不规则边缘。"
    )
    info = {
        "present": True,
        "count_estimate": count,
        "diameter_range_px": [d_min, d_max],
        "main_quadrant": main_quad,
        "min_dist_to_fovea_dd": min_dist,
        "morphology": shape_word,
    }
    return info, desc


def analyze_he(mask: np.ndarray, fovea_center: tuple[int, int], od_center: tuple[int, int]) -> tuple[dict, str]:
    props = _connected_components_props(mask)
    if not props:
        return {"present": False}, "未见暗红色斑块状出血，HE（-）。"

    count = len(props)
    max_width = int(max((p.bbox[2] - p.bbox[0]) for p in props))
    # HE tends to be larger and irregular; use bbox aspect ratio heuristics.
    ratios = [float(p.major_axis / (p.minor_axis + 1e-5)) for p in props]
    shape_word = "不规则斑块状" if float(np.mean(ratios)) > 1.5 else "类圆点状"

    quadrant_counts = {"TS": 0, "TI": 0, "NS": 0, "NI": 0}
    for p in props:
        quadrant_counts[get_quadrant(p.cx, p.cy, od_center, fovea_center)] += 1

    severe_flag = ""
    for quad, cnt in quadrant_counts.items():
        if cnt >= HE_QUADRANT_SEVERE:
            severe_flag = f"【注意：{quad}象限出血{cnt}处，达重度NPDR的4-2-1规则阈值，建议紧急复查。】"
            break

    main_quads = [q for q, c in quadrant_counts.items() if c > 0]
    location_desc = f"分布于{'、'.join(main_quads)}象限"
    desc = (
        f"可见{count}处暗红色至黑红色{shape_word}出血，最大病灶横径约{max_width}px，{location_desc}。"
        f"面积明显大于微动脉瘤。{severe_flag}"
    )
    info = {
        "present": True,
        "count_estimate": count,
        "max_width_px": max_width,
        "quadrant_distribution": quadrant_counts,
        "morphology": shape_word,
        "4_2_1_alert": bool(severe_flag),
    }
    return info, desc


def analyze_ex(mask: np.ndarray, fovea_center: tuple[int, int], od_center: tuple[int, int]) -> tuple[dict, str]:
    props = _connected_components_props(mask)
    if not props:
        return {"present": False}, "未见黄白色蜡样沉积物，EX（-）。"

    total_area = int(np.sum(mask > 0))
    dist_dds = [dist_to_fovea_dd(p.cx, p.cy, fovea_center) for p in props]
    min_dist = float(min(dist_dds))
    csme_risk = (min_dist * DD_PIXEL_ESTIMATE) < CSME_THRESHOLD_PX

    # Detect circinate (ring-like) EX pattern via enclosed hollow area.
    # Practical issue: circinate rings may be "broken" (gaps) in masks → interior connects to exterior.
    # We therefore apply a light morphological closing before hole estimation.
    binary = (mask > 0).astype(np.uint8)
    mask_area = int(binary.sum())
    hollow_area = 0

    def _hole_area_after_close(src: np.ndarray, k: int) -> int:
        bb = src
        if k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            bb = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=1)
        h, w = bb.shape[:2]
        inv = (1 - bb).astype(np.uint8)
        ff = inv.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(ff, flood_mask, (0, 0), 0)
        return int((ff > 0).sum())

    # Multi-scale closing: broken rings may require a larger kernel to become enclosed.
    # Use the scale that yields the largest hollow ratio.
    best_k = 0
    best_hollow = 0
    if mask_area > 0:
        for k in (0, 9, 25):
            ha = _hole_area_after_close(binary, k)
            if ha > best_hollow:
                best_hollow = ha
                best_k = k
        hollow_area = best_hollow

    if mask_area > 0 and hollow_area > int(mask_area * 0.20):
        pattern = "环形（circinate）分布"
    else:
        pattern = "多灶散在分布" if len(props) >= 10 else "散在簇状"
    csme_desc = ""
    if csme_risk:
        csme_desc = f"【注意：最近病灶距黄斑中心仅{min_dist}DD（<500μm），达CSME诊断阈值，中心视力受累风险高。】"

    desc = (
        f"可见黄白色蜡样沉积物，边界清晰锐利，呈{pattern}，总掩码面积约{total_area}px²。"
        f"最近病灶距黄斑中心约{min_dist}DD。{csme_desc}"
        f"边界明确，区别于棉绒斑的边缘模糊；不遮挡其下血管。"
    )
    info = {
        "present": True,
        "total_area_px2": total_area,
        "pattern": pattern,
        "hollow_area_px2": hollow_area,
        "hollow_ratio": round(float(hollow_area) / float(mask_area + 1e-6), 3),
        "min_dist_to_fovea_dd": min_dist,
        "csme_risk": csme_risk,
    }
    return info, desc


def analyze_se(mask: np.ndarray, fovea_center: tuple[int, int], od_center: tuple[int, int], dd_px: float = DD_PIXEL_ESTIMATE) -> tuple[dict, str]:
    props = _connected_components_props(mask)
    if not props:
        return {"present": False}, "未见灰白色棉絮状斑块，SE（-）。"

    count = len(props)
    od_area = float(np.pi * (dd_px / 2.0) ** 2)
    size_descs: list[str] = []
    for p in sorted(props, key=lambda x: -x.area):
        ratio = float(p.area) / od_area
        size_descs.append(f"约{ratio:.2f}DD²" + ("（较大）" if ratio >= 0.083 else ""))

    dist_dds = [dist_to_fovea_dd(p.cx, p.cy, fovea_center) for p in props]
    location_desc = f"位于后极部，距黄斑中心约{min(dist_dds):.1f}DD"
    desc = (
        f"可见{count}处灰白色棉絮状斑块，边界模糊不清，"
        f"大小{size_descs[0] if len(size_descs)==1 else '分别为' + '、'.join(size_descs[:3])}，{location_desc}。"
        f"边界欠清，区别于硬性渗出的清晰边界；可能遮挡局部血管走行。"
    )
    info = {
        "present": True,
        "count": count,
        "min_dist_to_fovea_dd": round(float(min(dist_dds)), 1),
        "location": location_desc,
    }
    return info, desc


def analyze_nv(mask: np.ndarray, od_center: tuple[int, int], od_radius: int) -> tuple[dict, str]:
    props = _connected_components_props(mask)
    if not props:
        return {"present": False}, "未见视网膜表面新生血管，NV（-）。"

    count = len(props)
    nvd, nve = 0, 0
    for p in props:
        dist_od = float(np.hypot(p.cx - od_center[0], p.cy - od_center[1]))
        if dist_od <= float(od_radius) * 2.0:
            nvd += 1
        else:
            nve += 1
    nv_type_desc: list[str] = []
    if nvd:
        nv_type_desc.append(f"NVD（视盘旁{nvd}处）")
    if nve:
        nv_type_desc.append(f"NVE（视网膜表面{nve}处）")
    desc = (
        f"视网膜{'及视盘' if nvd else ''}表面可见细小新生血管网，走行迂曲紊乱，形态不规则，共{count}处（{'、'.join(nv_type_desc)}）。"
        f"【提示增殖期DR，建议紧急干预。】"
    )
    info = {"present": True, "total_count": count, "nvd_count": nvd, "nve_count": nve}
    return info, desc


def estimate_od_fovea(preprocessed_rgb: np.ndarray) -> tuple[tuple[int, int], int, tuple[int, int]]:
    gray = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=40,
        maxRadius=120,
    )

    def _fallback() -> tuple[tuple[int, int], int, tuple[int, int]]:
        od_x0, od_y0, od_r0 = int(1024 * 0.65), 512, 70
        dd0 = od_r0 * 2
        fovea_x0 = int(od_x0 - 2.5 * dd0)
        fovea_y0 = int(od_y0)
        fovea_x0 = max(0, min(1023, fovea_x0))
        fovea_y0 = max(0, min(1023, fovea_y0))
        return (od_x0, od_y0), od_r0, (fovea_x0, fovea_y0)

    # Pick the first circle candidate that yields a plausible OD–fovea geometry.
    if circles is not None and len(circles) > 0:
        cand = circles[0].tolist()
        # Sort by radius descending (OD tends to be among larger circles in this range).
        cand.sort(key=lambda x: float(x[2]), reverse=True)
        for x, y, r in cand[:5]:
            od_x, od_y, od_r = int(round(x)), int(round(y)), int(round(r))
            dd = od_r * 2
            fovea_x = int(round(od_x - 2.5 * dd))  # assume right-eye orientation (most FGADR are right-eye)
            fovea_y = int(od_y)
            fovea_x = max(0, min(1023, fovea_x))
            fovea_y = max(0, min(1023, fovea_y))
            if validate_od_fovea((od_x, od_y), od_r, (fovea_x, fovea_y)):
                return (od_x, od_y), od_r, (fovea_x, fovea_y)

    # If Hough fails or yields implausible geometry, fall back to a stable prior.
    return _fallback()


def load_mask_1024(path: Path | None, crop_box_xyxy: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """
    Load a binary/gray mask and align it to the same crop+resize used by image preprocessing.

    Note:
      - IDRiD/FGADR masks are typically in the ORIGINAL image coordinate system.
      - Our training images are cropped (fundus circle bbox) then resized to 1024.
      - Therefore, the mask must apply the SAME crop before resizing, otherwise the lesion locations
        and derived geometry (quadrants, distances) will be inconsistent.
    """
    if path is None or not path.exists():
        return np.zeros((1024, 1024), dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros((1024, 1024), dtype=np.uint8)

    if crop_box_xyxy is not None:
        x1, y1, x2, y2 = crop_box_xyxy
        x1 = max(0, min(int(x1), m.shape[1]))
        x2 = max(0, min(int(x2), m.shape[1]))
        y1 = max(0, min(int(y1), m.shape[0]))
        y2 = max(0, min(int(y2), m.shape[0]))
        if x2 > x1 and y2 > y1:
            m = m[y1:y2, x1:x2]

    if m.shape[0] != 1024 or m.shape[1] != 1024:
        m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    return m


def generate_cot(
    image_id: str,
    mask_paths: dict[str, Path | None],
    fovea_center: tuple[int, int],
    od_center: tuple[int, int],
    od_radius: int,
    has_nv: bool = False,
    crop_box_xyxy: tuple[int, int, int, int] | None = None,
) -> tuple[str, dict]:
    ma = load_mask_1024(mask_paths.get("MA"), crop_box_xyxy=crop_box_xyxy)
    he = load_mask_1024(mask_paths.get("HE"), crop_box_xyxy=crop_box_xyxy)
    ex = load_mask_1024(mask_paths.get("EX"), crop_box_xyxy=crop_box_xyxy)
    se = load_mask_1024(mask_paths.get("SE"), crop_box_xyxy=crop_box_xyxy)
    nv = (
        load_mask_1024(mask_paths.get("NV"), crop_box_xyxy=crop_box_xyxy)
        if has_nv
        else np.zeros((1024, 1024), dtype=np.uint8)
    )

    ma_info, ma_desc = analyze_ma(ma, fovea_center, od_center)
    he_info, he_desc = analyze_he(he, fovea_center, od_center)
    ex_info, ex_desc = analyze_ex(ex, fovea_center, od_center)
    se_info, se_desc = analyze_se(se, fovea_center, od_center)
    nv_info, nv_desc = analyze_nv(nv, od_center, od_radius) if has_nv else ({"present": False}, "未见新生血管，NV（-）。")

    analysis = f"【MA】{ma_desc}\n【HE】{he_desc}\n【EX】{ex_desc}\n【SE】{se_desc}\n【NV】{nv_desc}"
    out = {"MA": ma_info, "HE": he_info, "EX": ex_info, "SE": se_info, "NV": nv_info}
    return analysis, out


def format_assistant_output(analysis_text: str, output_json: dict) -> str:
    json_str = json.dumps(output_json, ensure_ascii=False, indent=2)
    return f"## Analysis\n{analysis_text}\n\n## Output\n{json_str}"

