from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _as_number(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _as_xy(x: Any) -> tuple[float, float] | None:
    if isinstance(x, (list, tuple)) and len(x) >= 2:
        a = _as_number(x[0])
        b = _as_number(x[1])
        if a is None or b is None:
            return None
        return (a, b)
    if isinstance(x, dict):
        a = _as_number(x.get("x"))
        b = _as_number(x.get("y"))
        if a is None or b is None:
            return None
        return (a, b)
    return None


def _deep_get(obj: Any, keys: list[str]) -> Any:
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _find_first_key(obj: Any, wanted: set[str]) -> Any:
    """
    Recursive search: return value of the first matching key in DFS order.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in wanted:
                return v
        for v in obj.values():
            found = _find_first_key(v, wanted)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_first_key(v, wanted)
            if found is not None:
                return found
    return None


def _find_lesion_block(root: dict[str, Any], lesion: str) -> dict[str, Any] | None:
    """
    Try common RetSAM patterns and fall back to DFS search for a dict keyed by lesion name.
    """
    lesion = lesion.upper()
    # Common: root["lesions"][lesion]
    for path in (["lesions", lesion], ["lesion", lesion], ["lesion_analysis", lesion], ["lesion_quantification", lesion]):
        blk = _deep_get(root, path)
        if isinstance(blk, dict):
            return blk
    # Sometimes keys are lower-case.
    for path in (["lesions", lesion.lower()], ["lesion_analysis", lesion.lower()]):
        blk = _deep_get(root, path)
        if isinstance(blk, dict):
            return blk

    # Fallback: DFS to find dict where this lesion exists as a key.
    def dfs(o: Any) -> dict[str, Any] | None:
        if isinstance(o, dict):
            if lesion in o and isinstance(o[lesion], dict):
                return o[lesion]
            if lesion.lower() in o and isinstance(o[lesion.lower()], dict):
                return o[lesion.lower()]
            for v in o.values():
                r = dfs(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = dfs(v)
                if r is not None:
                    return r
        return None

    return dfs(root)


@dataclass(frozen=True)
class LesionMetrics:
    count: int
    total_area: float
    quadrant_distribution: dict[str, int]  # TS/TI/NS/NI
    centroids: list[tuple[float, float]]  # (x,y) pixels in RetSAM coordinate space


def parse_lesion_metrics(q: dict[str, Any], lesion: str) -> LesionMetrics | None:
    lesion = lesion.upper()
    blk = _find_lesion_block(q, lesion)

    # Schema v2.x: measurements.lesions.lesion_dr.categories.{hemorrhage,exudate,cotton_wool_spot}
    if blk is None:
        cat_map = {"HE": "hemorrhage", "EX": "exudate", "SE": "cotton_wool_spot"}
        cat = cat_map.get(lesion)
        if cat:
            cats = _deep_get(q, ["measurements", "lesions", "lesion_dr", "categories"])
            if isinstance(cats, dict) and isinstance(cats.get(cat), dict):
                blk = cats[cat]
    if blk is None:
        return None

    count = _as_int(_find_first_key(blk, {"count", "num", "n", "number"})) or 0
    total_area = _as_number(_find_first_key(blk, {"total_area", "area_total", "area", "sum_area", "area_px", "total_area_px"})) or 0.0

    qd_raw = _find_first_key(blk, {"quadrant_distribution", "quadrant", "quadrants"})
    qd: dict[str, int] = {"TS": 0, "TI": 0, "NS": 0, "NI": 0}
    if isinstance(qd_raw, dict):
        # Schema v2.x uses superior_temporal/superior_nasal/inferior_temporal/inferior_nasal.
        st = _as_int(qd_raw.get("superior_temporal")) or 0
        sn = _as_int(qd_raw.get("superior_nasal")) or 0
        it = _as_int(qd_raw.get("inferior_temporal")) or 0
        inn = _as_int(qd_raw.get("inferior_nasal")) or 0
        if any([st, sn, it, inn]):
            qd["TS"], qd["NS"], qd["TI"], qd["NI"] = st, sn, it, inn
        else:
            for k, v in qd_raw.items():
                kk = str(k).upper()
                if kk in qd:
                    qd[kk] = _as_int(v) or 0

    cents_raw = _find_first_key(blk, {"centroids", "centroid_list", "lesion_centroids", "centroid"})
    centroids: list[tuple[float, float]] = []
    if isinstance(cents_raw, list):
        for it in cents_raw:
            xy = _as_xy(it)
            if xy is not None:
                centroids.append(xy)
    else:
        xy = _as_xy(cents_raw)
        if xy is not None:
            centroids.append(xy)

    # Some schemas don't report centroids; leave empty.
    return LesionMetrics(count=count, total_area=float(total_area), quadrant_distribution=qd, centroids=centroids)


@dataclass(frozen=True)
class ODInfo:
    od_center: tuple[float, float] | None
    od_area: float | None

    @property
    def od_radius_px(self) -> float | None:
        if self.od_area is None or self.od_area <= 0:
            return None
        return math.sqrt(float(self.od_area) / math.pi)


def parse_fovea_xy(q: dict[str, Any]) -> tuple[float, float] | None:
    return _as_xy(_find_first_key(q, {"foveal_coordinates", "fovea_coordinates", "fovea", "macula_center"}))


def parse_od_info(q: dict[str, Any]) -> ODInfo:
    od_center = _as_xy(_find_first_key(q, {"od_center", "optic_disc_center", "optic_disc_coordinates"}))
    od_area = _as_number(_find_first_key(q, {"od_area", "optic_disc_area", "disc_area"}))
    return ODInfo(od_center=od_center, od_area=od_area)

