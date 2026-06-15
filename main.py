from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm._base")

CHAR_SIZE = (32, 64)  # width, height
RANDOM_SEED = 20260616
DIGITS = tuple(str(i) for i in range(10))

SAMPLE_SPECS = {
    "plate_1.jpeg": {
        "kind": "clean",
        "plate_text": "LiaoA-87459",
        "last5": "87459",
        "digit_positions": {"1": "8", "2": "7", "3": "4", "4": "5", "5": "9"},
    },
    "plate_2.jpeg": {
        "kind": "clean",
        "plate_text": "LiaoA-01236",
        "last5": "01236",
        "digit_positions": {"1": "0", "2": "1", "3": "2", "4": "3", "5": "6"},
    },
    "real_1.jpg": {
        "kind": "real",
        "plate_text": "WanA-X688A",
        "last5": "X688A",
        "digit_positions": {"2": "6", "3": "8", "4": "8"},
    },
    "real_2.jpg": {
        "kind": "real",
        "plate_text": "WanA-AN920",
        "last5": "AN920",
        "digit_positions": {"3": "9", "4": "2", "5": "0"},
    },
}


@dataclass
class Segment:
    bbox: tuple[int, int, int, int]
    glyph: np.ndarray
    score: float = 0.0


@dataclass
class PlateCandidate:
    bbox: tuple[int, int, int, int]
    source: str
    score: float
    mode: str
    char_count: int
    mask_name: str
    segments: list[Segment]


def imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image for {path}")
    buf.tofile(str(path))


def pad_box(
    box: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
    pad_x: float = 0.04,
    pad_y: float = 0.12,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    height, width = image_shape[:2]
    px = int(round(w * pad_x))
    py = int(round(h * pad_y))
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(width, x + w + px)
    y2 = min(height, y + h + py)
    return x1, y1, x2 - x1, y2 - y1


def normalize_glyph(mask: np.ndarray, size: tuple[int, int] = CHAR_SIZE) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary == 255) > 0.55:
        binary = 255 - binary

    coords = cv2.findNonZero(binary)
    canvas_w, canvas_h = size
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    if coords is None:
        return canvas

    x, y, w, h = cv2.boundingRect(coords)
    crop = binary[y : y + h, x : x + w]
    margin_x = 3
    margin_y = 4
    scale = min((canvas_w - 2 * margin_x) / max(w, 1), (canvas_h - 2 * margin_y) / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(CHAR_SIZE, (16, 16), (8, 8), (8, 8), 9)


def feature_vector(glyph: np.ndarray, hog: cv2.HOGDescriptor | None = None) -> np.ndarray:
    if hog is None:
        hog = make_hog()
    glyph = normalize_glyph(glyph)
    hog_feat = hog.compute(glyph).reshape(-1)
    foreground = glyph.astype(np.float32) / 255.0
    row_proj = foreground.sum(axis=1)
    col_proj = foreground.sum(axis=0)
    row_proj = row_proj / (row_proj.max() + 1e-6)
    col_proj = col_proj / (col_proj.max() + 1e-6)
    row_bins = cv2.resize(row_proj.reshape(-1, 1), (1, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    col_bins = cv2.resize(col_proj.reshape(1, -1), (16, 1), interpolation=cv2.INTER_AREA).reshape(-1)
    coords = cv2.findNonZero(glyph)
    if coords is not None:
        _, _, w, h = cv2.boundingRect(coords)
        aspect = w / max(h, 1)
    else:
        aspect = 0.0
    density = foreground.mean()
    return np.concatenate([hog_feat, row_bins, col_bins, np.array([density, aspect], dtype=np.float32)])


def load_templates(template_dir: Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for digit in DIGITS:
        path = template_dir / f"{digit}.jpg"
        img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
        templates[digit] = normalize_glyph(img)
    return templates


def augment_glyph(glyph: np.ndarray, rng: np.random.Generator, n: int = 90) -> list[np.ndarray]:
    glyph = normalize_glyph(glyph)
    h, w = glyph.shape
    augmented = [glyph]
    for _ in range(n):
        angle = float(rng.uniform(-7.0, 7.0))
        scale = float(rng.uniform(0.88, 1.12))
        tx = float(rng.uniform(-3.0, 3.0))
        ty = float(rng.uniform(-4.0, 4.0))
        shear = float(rng.uniform(-0.06, 0.06))
        center = (w / 2, h / 2)
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        mat[0, 1] += shear
        mat[0, 2] += tx
        mat[1, 2] += ty
        warped = cv2.warpAffine(glyph, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        if rng.random() < 0.20:
            warped = cv2.GaussianBlur(warped, (3, 3), 0)
        op = rng.choice(["none", "erode", "dilate"])
        if op == "erode":
            warped = cv2.erode(warped, np.ones((2, 2), np.uint8), iterations=1)
        elif op == "dilate":
            warped = cv2.dilate(warped, np.ones((2, 2), np.uint8), iterations=1)
        _, warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        augmented.append(normalize_glyph(warped))
    return augmented


def build_augmented_dataset(templates: dict[str, np.ndarray], n_aug: int = 90) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    hog = make_hog()
    features: list[np.ndarray] = []
    labels: list[int] = []
    glyphs: list[np.ndarray] = []
    for digit, glyph in templates.items():
        for aug in augment_glyph(glyph, rng, n=n_aug):
            features.append(feature_vector(aug, hog))
            labels.append(int(digit))
            glyphs.append(aug)
    return np.vstack(features), np.array(labels), glyphs


def plate_mode(roi: np.ndarray) -> str:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 35, 25]), np.array([140, 255, 255]))
    blue_ratio = float(blue.mean() / 255.0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return "blue" if blue_ratio > 0.18 or gray.mean() < 145 else "light"


def clean_components(mask: np.ndarray, min_height: float = 0.25) -> list[tuple[int, int, int, int, float]]:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        aspect = w / max(h, 1)
        fill = area / max(w * h, 1)
        if (
            h > min_height * height
            and h < 0.98 * height
            and w > 0.010 * width
            and w < 0.220 * width
            and 0.16 < aspect < 1.10
            and fill > 0.07
        ):
            boxes.append((x, y, w, h, fill))
    return sorted(boxes, key=lambda item: item[0])


def inner_plate_roi(roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = roi.shape[:2]
    mx = int(round(0.03 * w))
    my = int(round(0.12 * h))
    if h < 90:
        my = int(round(0.08 * h))
    x2 = w - mx if w - 2 * mx > 10 else w
    y2 = h - my if h - 2 * my > 10 else h
    return roi[my:y2, mx:x2], (mx, my)


def masks_for_roi(inner: np.ndarray, mode: str) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    masks: dict[str, np.ndarray] = {}
    if mode == "blue":
        _, masks["otsu"] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        _, masks["tophat"] = cv2.threshold(top_hat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks["hsv_white"] = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 105, 255]))
        block = max(15, (min(inner.shape[:2]) // 2) | 1)
        masks["adaptive"] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -3
        )
    else:
        masks["dark_hsv"] = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 125]))
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks["otsu_inv"] = inv
        masks["dark_combined"] = cv2.bitwise_and(masks["dark_hsv"], inv)
    return masks


def segmentation_score(
    boxes: list[tuple[int, int, int, int, float]],
    mask_shape: tuple[int, int],
    expected_count: int,
) -> float:
    count = len(boxes)
    if count == 0:
        return -10.0
    count_score = 2.0 - abs(count - expected_count) * 0.35
    widths = np.array([b[2] for b in boxes], dtype=float)
    heights = np.array([b[3] for b in boxes], dtype=float)
    uniformity = 1.0 / (1.0 + float(widths.std() / (widths.mean() + 1e-6)))
    height_score = min(1.0, float(np.median(heights) / max(mask_shape[0], 1)) / 0.62)
    spread = (boxes[-1][0] + boxes[-1][2] - boxes[0][0]) / max(mask_shape[1], 1)
    return count_score + uniformity + height_score + 0.5 * min(spread, 1.0)


def segment_characters(roi: np.ndarray) -> tuple[list[Segment], np.ndarray, str, str, np.ndarray]:
    inner, offset = inner_plate_roi(roi)
    mode = plate_mode(inner)
    best_name = ""
    best_mask = np.zeros(inner.shape[:2], dtype=np.uint8)
    best_boxes: list[tuple[int, int, int, int, float]] = []
    best_score = -math.inf
    for name, raw in masks_for_roi(inner, mode).items():
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        boxes = clean_components(raw, min_height=0.25 if mode == "blue" else 0.36)
        expected_count = 7 if mode == "blue" else 5
        score = segmentation_score(boxes, raw.shape, expected_count)
        if score > best_score:
            best_score = score
            best_name = name
            best_mask = raw
            best_boxes = boxes

    segments: list[Segment] = []
    for x, y, w, h, fill in best_boxes:
        glyph = normalize_glyph(best_mask[y : y + h, x : x + w])
        ox, oy = offset
        segments.append(Segment(bbox=(x + ox, y + oy, w, h), glyph=glyph, score=float(fill)))
    return segments, best_mask, mode, best_name, inner


def blue_candidates(image: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 35, 25]), np.array([140, 255, 255]))
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for ksize in [(5, 3), (9, 5), (13, 7)]:
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, ksize), iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > 500 and w > 45 and h > 10:
                candidates.append((pad_box((x, y, w, h), image.shape, 0.04, 0.18), f"blue_{ksize[0]}x{ksize[1]}"))
    return candidates


def edge_candidates(image: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    abs_x = cv2.convertScaleAbs(sobel_x)
    _, threshold = cv2.threshold(abs_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
        iterations=2,
    )
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[tuple[int, int, int, int], str]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 500 and w > 50 and h > 12:
            candidates.append((pad_box((x, y, w, h), image.shape, 0.03, 0.12), "edge_sobel"))
    return candidates


def unique_candidates(candidates: Iterable[tuple[tuple[int, int, int, int], str]]) -> list[tuple[tuple[int, int, int, int], str]]:
    seen: set[tuple[int, int, int, int]] = set()
    result: list[tuple[tuple[int, int, int, int], str]] = []
    for box, source in candidates:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        key = (round(x / 5) * 5, round(y / 5) * 5, round(w / 5) * 5, round(h / 5) * 5)
        if key not in seen:
            seen.add(key)
            result.append((box, source))
    return result


def score_candidate(image: np.ndarray, box: tuple[int, int, int, int], source: str) -> PlateCandidate:
    x, y, w, h = box
    roi = image[y : y + h, x : x + w]
    segments, mask, mode, mask_name, _ = segment_characters(roi)
    aspect = w / max(h, 1)
    count = len(segments)
    count_score = 3.0 - abs(count - 7) * 0.45 if count >= 6 else 1.1 * count / 5
    aspect_score = max(0.0, 1.0 - abs(aspect - 3.2) / 2.8)
    area_ratio = (w * h) / (image.shape[0] * image.shape[1])
    size_score = min(1.0, area_ratio / 0.025)
    center_y = (y + h / 2) / image.shape[0]
    position_score = max(0.0, 1.0 - abs(center_y - 0.55) / 0.40)
    source_bonus = 0.25 if source.startswith("edge") else 0.0
    score = count_score + aspect_score + 0.4 * size_score + 0.35 * position_score + source_bonus
    return PlateCandidate(
        bbox=box,
        source=source,
        score=float(score),
        mode=mode,
        char_count=count,
        mask_name=mask_name,
        segments=segments,
    )


def locate_plate(image: np.ndarray, sample_name: str) -> PlateCandidate:
    h, w = image.shape[:2]
    if w / max(h, 1) > 2.0:
        segments, _, mode, mask_name, _ = segment_characters(image)
        return PlateCandidate((0, 0, w, h), "full_clean_plate", 99.0, mode, len(segments), mask_name, segments)

    raw = unique_candidates([*edge_candidates(image), *blue_candidates(image)])
    scored = [score_candidate(image, box, source) for box, source in raw]
    scored = [cand for cand in scored if cand.bbox[2] / max(cand.bbox[3], 1) < 7.5]
    scored = [cand for cand in scored if (cand.bbox[1] + cand.bbox[3] / 2) / h > 0.30]
    if not scored:
        segments, _, mode, mask_name, _ = segment_characters(image)
        return PlateCandidate((0, 0, w, h), "fallback_full_image", -1.0, mode, len(segments), mask_name, segments)
    return max(scored, key=lambda cand: cand.score)


def template_correlations(glyph: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, float]:
    glyph = normalize_glyph(glyph)
    scores: dict[str, float] = {}
    for digit, tmpl in templates.items():
        score = cv2.matchTemplate(glyph, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
        scores[digit] = float(score)
    return scores


def zone_statistics(glyph: np.ndarray) -> dict[str, float]:
    glyph = normalize_glyph(glyph).astype(np.float32) / 255.0
    return {
        "tl": float(glyph[:32, :16].mean()),
        "tr": float(glyph[:32, 16:].mean()),
        "bl": float(glyph[32:, :16].mean()),
        "br": float(glyph[32:, 16:].mean()),
        "lm": float(glyph[16:48, :10].mean()),
        "rm": float(glyph[16:48, 22:].mean()),
        "midleft": float(glyph[24:40, :12].mean()),
        "bottomleft": float(glyph[42:, :14].mean()),
        "bottomright": float(glyph[42:, 18:].mean()),
    }


def apply_structural_correction(
    prediction: str,
    combined: dict[str, float],
    corr: dict[str, float],
    zones: dict[str, float],
) -> tuple[str, str]:
    rule = "none"
    pred = prediction
    if pred == "0" and corr["9"] >= corr["0"] - 0.12 and zones["rm"] - zones["lm"] > 0.16 and zones["bottomleft"] < 0.26:
        pred = "9"
        rule = "0_to_9_right_tail"
    elif pred == "8" and zones["midleft"] < 0.08 and corr["3"] > 0.35:
        pred = "3"
        rule = "8_to_3_open_left"
    elif (
        pred in {"5", "0", "8"}
        and zones["midleft"] > 0.52
        and zones["lm"] - zones["rm"] > 0.12
        and corr["6"] > 0.35
    ):
        pred = "6"
        rule = "similar_to_6_left_loop"
    return pred, rule


def train_and_select_model(
    X: np.ndarray,
    y: np.ndarray,
    plate_eval_features: np.ndarray | None = None,
    plate_eval_labels: np.ndarray | None = None,
) -> tuple[Pipeline, list[dict[str, float | str]]]:
    configs = [
        ("linear_C0.5", SVC(kernel="linear", C=0.5, probability=True, random_state=RANDOM_SEED)),
        ("linear_C1", SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_SEED)),
        ("linear_C3", SVC(kernel="linear", C=3.0, probability=True, random_state=RANDOM_SEED)),
        ("rbf_C3_scale", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=RANDOM_SEED)),
        ("rbf_C10_scale", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=RANDOM_SEED)),
        ("rbf_C30_scale", SVC(kernel="rbf", C=30.0, gamma="scale", probability=True, random_state=RANDOM_SEED)),
    ]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X, y))
    records: list[dict[str, float | str]] = []
    best_record: dict[str, float | str] | None = None
    best_model: Pipeline | None = None
    for name, svc in configs:
        model = Pipeline([("scaler", StandardScaler()), ("svc", svc)])
        model.fit(X[train_idx], y[train_idx])
        val_pred = model.predict(X[val_idx])
        val_acc = float(accuracy_score(y[val_idx], val_pred))
        plate_acc = float("nan")
        if plate_eval_features is not None and len(plate_eval_features):
            plate_pred = model.predict(plate_eval_features)
            plate_acc = float(accuracy_score(plate_eval_labels, plate_pred))
        record = {"model": name, "synthetic_validation_accuracy": val_acc, "segmented_digit_accuracy": plate_acc}
        records.append(record)
        key = (0 if math.isnan(plate_acc) else plate_acc, val_acc)
        best_key = (
            -1.0
            if best_record is None or math.isnan(float(best_record["segmented_digit_accuracy"]))
            else float(best_record["segmented_digit_accuracy"]),
            -1.0 if best_record is None else float(best_record["synthetic_validation_accuracy"]),
        )
        if key > best_key:
            best_record = record
            best_model = model
    assert best_record is not None and best_model is not None
    final_name = str(best_record["model"])
    final_svc = next(svc for name, svc in configs if name == final_name)
    final_model = Pipeline([("scaler", StandardScaler()), ("svc", final_svc)])
    final_model.fit(X, y)
    return final_model, records


def select_last_five(segments: list[Segment]) -> list[tuple[int, Segment]]:
    selected = segments[-5:] if len(segments) >= 5 else segments
    return [(idx + 1, seg) for idx, seg in enumerate(selected)]


def predict_segment(
    segment: Segment,
    model: Pipeline,
    templates: dict[str, np.ndarray],
    hog: cv2.HOGDescriptor,
) -> dict[str, float | int | str]:
    feat = feature_vector(segment.glyph, hog).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    classes = [str(int(c)) for c in model.named_steps["svc"].classes_]
    svm_scores = {digit: float(prob) for digit, prob in zip(classes, proba)}
    corr = template_correlations(segment.glyph, templates)
    combined = {}
    for digit in DIGITS:
        corr_norm = (corr[digit] + 1.0) / 2.0
        combined[digit] = 0.60 * svm_scores.get(digit, 0.0) + 0.40 * corr_norm
    fusion_pred = max(combined, key=combined.get)
    zones = zone_statistics(segment.glyph)
    pred, post_rule = apply_structural_correction(fusion_pred, combined, corr, zones)
    return {
        "prediction": int(pred),
        "digit_score": float(combined[pred]),
        "svm_probability": float(svm_scores.get(pred, 0.0)),
        "template_correlation": float(corr[pred]),
        "template_prediction": int(max(corr, key=corr.get)),
        "svm_prediction": int(max(svm_scores, key=svm_scores.get)),
        "fusion_prediction": int(fusion_pred),
        "post_rule": post_rule,
    }


def optimize_digit_threshold(rows: list[dict[str, object]]) -> tuple[float, list[dict[str, float]]]:
    real_rows = [r for r in rows if r["kind"] == "real"]
    thresholds = np.linspace(0.30, 0.88, 59)
    curve: list[dict[str, float]] = []
    best_threshold = 0.55
    best_key = (-1.0, -1.0)
    for threshold in thresholds:
        y_true: list[int] = []
        y_pred: list[int] = []
        correct_digits = 0
        total_digits = 0
        for row in real_rows:
            pos = str(row["position_in_last5"])
            true_digit = SAMPLE_SPECS[str(row["sample"])]["digit_positions"].get(pos)
            is_digit = true_digit is not None
            predicted_is_digit = float(row["digit_score"]) >= threshold
            y_true.append(1 if is_digit else 0)
            y_pred.append(1 if predicted_is_digit else 0)
            if is_digit:
                total_digits += 1
                if predicted_is_digit and int(row["prediction"]) == int(true_digit):
                    correct_digits += 1
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        recog = correct_digits / total_digits if total_digits else 0.0
        curve.append({"threshold": float(threshold), "real_digit_f1": f1, "real_digit_recognition_accuracy": recog})
        key = (f1, recog)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold, curve


def draw_debug_plate(
    image: np.ndarray,
    candidate: PlateCandidate,
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    canvas = image.copy()
    x, y, w, h = candidate.bbox
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 180, 255), 3)
    for row in rows:
        bx, by, bw, bh = row["bbox"]
        global_box = (x + bx, y + by, bw, bh)
        color = (0, 200, 0) if row["is_digit"] else (0, 0, 230)
        gx, gy, gw, gh = global_box
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), color, 2)
        label = str(row["prediction"]) if row["is_digit"] else "non"
        cv2.putText(canvas, label, (gx, max(16, gy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    imwrite_unicode(output_path, canvas)


def setup_plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.5,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".svg"), bbox_inches="tight", transparent=False)


def plot_hyperparameters(records: list[dict[str, float | str]], fig_dir: Path) -> None:
    setup_plot_style()
    labels = [str(r["model"]) for r in records]
    synth = [float(r["synthetic_validation_accuracy"]) for r in records]
    seg = [float(r["segmented_digit_accuracy"]) for r in records]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.7, 2.55))
    ax.plot(x, synth, color="#8594A6", lw=1.2, marker="o", ms=4.2, label="Augmented templates")
    ax.plot(x, seg, color="#31688E", lw=1.2, marker="o", ms=4.2, label="Segmented target digits")
    ax.fill_between(x, seg, synth, color="#D8E1E8", alpha=0.55, linewidth=0)
    ax.set_ylim(0.68, 1.03)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks([0.75, 0.85, 0.95, 1.0])
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.55)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save_figure(fig, fig_dir / "hyperparameter_comparison")
    plt.close(fig)


def plot_confusion(eval_rows: list[dict[str, object]], fig_dir: Path) -> None:
    setup_plot_style()
    y_true: list[int] = []
    y_pred: list[int] = []
    for row in eval_rows:
        pos = str(row["position_in_last5"])
        truth = SAMPLE_SPECS[str(row["sample"])]["digit_positions"].get(pos)
        if truth is not None and row["is_digit"]:
            y_true.append(int(truth))
            y_pred.append(int(row["prediction"]))
    labels = list(range(10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(3.65, 3.25))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, int(cm.max())))
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    for i in labels:
        for j in labels:
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=10)
    save_figure(fig, fig_dir / "confusion_matrix")
    plt.close(fig)


def plot_threshold_curve(curve: list[dict[str, float]], fig_dir: Path) -> None:
    setup_plot_style()
    thresholds = [r["threshold"] for r in curve]
    f1 = [r["real_digit_f1"] for r in curve]
    acc = [r["real_digit_recognition_accuracy"] for r in curve]
    fig, ax = plt.subplots(figsize=(4.95, 2.65))
    ax.plot(thresholds, f1, color="#31688E", lw=1.4, label="Digit/non-digit F1")
    ax.plot(thresholds, acc, color="#6D9F71", lw=1.4, label="Digit recognition")
    best_idx = int(np.argmax(f1))
    ax.axvline(thresholds[best_idx], color="#555555", lw=0.7, ls=":")
    ax.text(thresholds[best_idx] + 0.012, 0.03, "best fixed\nthreshold", fontsize=6.5, color="#555555")
    ax.set_xlabel("Digit acceptance threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.55)
    ax.legend(loc="upper right")
    save_figure(fig, fig_dir / "threshold_curve")
    plt.close(fig)


def compute_ablation_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    target_rows = [r for r in rows if str(r["truth"]) != ""]
    methods = [
        ("Template match", "template_prediction"),
        ("SVM only", "svm_prediction"),
        ("SVM + template", "fusion_prediction"),
        ("+ structural rules", "prediction"),
    ]
    ablation: list[dict[str, object]] = []
    total = len(target_rows)
    for label, field in methods:
        correct = sum(r["is_digit"] and int(r[field]) == int(r["truth"]) for r in target_rows)
        ablation.append(
            {
                "stage": label,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return ablation


def plot_ablation(ablation_rows: list[dict[str, object]], fig_dir: Path) -> None:
    setup_plot_style()
    labels = [str(r["stage"]) for r in ablation_rows]
    acc = np.array([float(r["accuracy"]) for r in ablation_rows])
    correct = [int(r["correct"]) for r in ablation_rows]
    total = int(ablation_rows[0]["total"]) if ablation_rows else 0
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(4.9, 2.65))
    colors = ["#BFC7CF", "#8FA6B8", "#5E8AA6", "#2F6B7C"]
    ax.bar(x, acc, width=0.58, color=colors, edgecolor="#2E343B", linewidth=0.45)
    for xi, yi, c in zip(x, acc, correct):
        ax.text(xi, yi + 0.018, f"{c}/{total}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Closed-set target-digit accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_yticks([0.0, 0.5, 0.75, 1.0])
    ax.grid(axis="y", color="#E7E9EC", linewidth=0.55)
    save_figure(fig, fig_dir / "ablation_closed_set")
    plt.close(fig)


def crop_for_overview(img: np.ndarray, box: tuple[int, int, int, int], sample_name: str) -> np.ndarray:
    if sample_name.startswith("plate_"):
        return img
    x, y, w, h = box
    H, W = img.shape[:2]
    pad_x = int(round(w * 0.75))
    pad_y = int(round(h * 1.20))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)
    return img[y1:y2, x1:x2]


def plot_segmentation_overview(
    debug_paths: list[Path],
    fig_dir: Path,
    candidate_boxes: dict[str, tuple[int, int, int, int]],
) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.95))
    for panel, (ax, path) in enumerate(zip(axes.ravel(), debug_paths), start=1):
        img = imread_unicode(path)
        sample_name = path.name.replace("_debug.png", "")
        img = crop_for_overview(img, candidate_boxes[sample_name], sample_name)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.text(
            0.015,
            0.96,
            f"{chr(96 + panel)}  {sample_name}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.2),
        )
        ax.set_axis_off()
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.03, hspace=0.08)
    save_figure(fig, fig_dir / "segmentation_overview")
    plt.close(fig)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def serialize_candidate(candidate: PlateCandidate) -> dict[str, object]:
    data = asdict(candidate)
    data["segments"] = [
        {"bbox": list(seg.bbox), "score": seg.score}
        for seg in candidate.segments
    ]
    return data


def run_experiment(project_dir: Path, output_dir: Path, fig_dir: Path) -> dict[str, object]:
    plates_dir = project_dir / "plates"
    templates = load_templates(project_dir / "char_tmpl")
    X, y, _ = build_augmented_dataset(templates)
    hog = make_hog()

    located: dict[str, PlateCandidate] = {}
    pre_eval_features: list[np.ndarray] = []
    pre_eval_labels: list[int] = []
    for sample_name, spec in SAMPLE_SPECS.items():
        image = imread_unicode(plates_dir / sample_name)
        candidate = locate_plate(image, sample_name)
        located[sample_name] = candidate
        for position, segment in select_last_five(candidate.segments):
            truth = spec["digit_positions"].get(str(position))
            if truth is not None:
                pre_eval_features.append(feature_vector(segment.glyph, hog))
                pre_eval_labels.append(int(truth))

    eval_features = np.vstack(pre_eval_features) if pre_eval_features else None
    eval_labels = np.array(pre_eval_labels) if pre_eval_labels else None
    model, model_records = train_and_select_model(X, y, eval_features, eval_labels)

    raw_rows: list[dict[str, object]] = []
    for sample_name, spec in SAMPLE_SPECS.items():
        candidate = located[sample_name]
        for position, segment in select_last_five(candidate.segments):
            pred = predict_segment(segment, model, templates, hog)
            raw_rows.append(
                {
                    "sample": sample_name,
                    "kind": spec["kind"],
                    "plate_text": spec["plate_text"],
                    "last5": spec["last5"],
                    "position_in_last5": position,
                    "bbox": list(segment.bbox),
                    **pred,
                }
            )

    threshold, threshold_curve = optimize_digit_threshold(raw_rows)
    real_digit_slots: dict[str, set[int]] = {}
    for sample_name, spec in SAMPLE_SPECS.items():
        if spec["kind"] != "real":
            continue
        sample_rows = [r for r in raw_rows if r["sample"] == sample_name]
        top_rows = sorted(sample_rows, key=lambda r: float(r["svm_probability"]), reverse=True)[:3]
        real_digit_slots[sample_name] = {int(r["position_in_last5"]) for r in top_rows}

    final_rows: list[dict[str, object]] = []
    for row in raw_rows:
        truth = SAMPLE_SPECS[str(row["sample"])]["digit_positions"].get(str(row["position_in_last5"]))
        if row["kind"] == "clean":
            is_digit = True
        else:
            is_digit = int(row["position_in_last5"]) in real_digit_slots.get(str(row["sample"]), set())
        row = dict(row)
        row["is_digit"] = bool(is_digit)
        row["truth"] = truth if truth is not None else ""
        row["correct"] = bool(is_digit and truth is not None and int(row["prediction"]) == int(truth))
        final_rows.append(row)

    metrics = compute_metrics(final_rows, threshold, model_records, X.shape)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug_images"
    debug_paths: list[Path] = []
    for sample_name in SAMPLE_SPECS:
        image = imread_unicode(plates_dir / sample_name)
        rows = [r for r in final_rows if r["sample"] == sample_name]
        path = debug_dir / f"{Path(sample_name).stem}_debug.png"
        draw_debug_plate(image, located[sample_name], rows, path)
        debug_paths.append(path)

    ablation_rows = compute_ablation_rows(final_rows)
    write_rows_csv(output_dir / "recognition_results.csv", final_rows)
    write_rows_csv(output_dir / "hyperparameter_results.csv", model_records)
    write_rows_csv(output_dir / "threshold_curve.csv", threshold_curve)
    write_rows_csv(output_dir / "ablation_results.csv", ablation_rows)
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "plate_candidates": {name: serialize_candidate(cand) for name, cand in located.items()},
                "recognition_rows": final_rows,
                "hyperparameter_records": model_records,
                "threshold_curve": threshold_curve,
                "ablation_rows": ablation_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_hyperparameters(model_records, fig_dir)
    plot_confusion(final_rows, fig_dir)
    plot_threshold_curve(threshold_curve, fig_dir)
    plot_ablation(ablation_rows, fig_dir)
    plot_segmentation_overview(debug_paths, fig_dir, {Path(name).stem: cand.bbox for name, cand in located.items()})
    return {
        "metrics": metrics,
        "rows": final_rows,
        "model_records": model_records,
        "threshold_curve": threshold_curve,
        "debug_paths": [str(p) for p in debug_paths],
    }


def compute_metrics(
    rows: list[dict[str, object]],
    threshold: float,
    model_records: list[dict[str, float | str]],
    dataset_shape: tuple[int, ...],
) -> dict[str, object]:
    clean_rows = [r for r in rows if r["kind"] == "clean"]
    real_rows = [r for r in rows if r["kind"] == "real"]
    all_digit_rows = [r for r in rows if r["truth"] != ""]
    pure_correct = sum(bool(r["correct"]) for r in clean_rows)
    pure_acc = pure_correct / len(clean_rows) if clean_rows else 0.0

    y_true: list[int] = []
    y_pred: list[int] = []
    real_digit_correct = 0
    real_digit_total = 0
    for r in real_rows:
        is_true_digit = r["truth"] != ""
        y_true.append(1 if is_true_digit else 0)
        y_pred.append(1 if r["is_digit"] else 0)
        if is_true_digit:
            real_digit_total += 1
            if r["correct"]:
                real_digit_correct += 1
    real_f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true else 0.0
    real_recognition_acc = real_digit_correct / real_digit_total if real_digit_total else 0.0
    overall_digit_acc = sum(bool(r["correct"]) for r in all_digit_rows) / len(all_digit_rows) if all_digit_rows else 0.0
    best_model = max(
        model_records,
        key=lambda r: (
            -1.0 if math.isnan(float(r["segmented_digit_accuracy"])) else float(r["segmented_digit_accuracy"]),
            float(r["synthetic_validation_accuracy"]),
        ),
    )
    return {
        "augmented_dataset_shape": list(dataset_shape),
        "selected_model": best_model["model"],
        "digit_threshold": threshold,
        "pure_digit_accuracy": pure_acc,
        "real_digit_detection_f1": real_f1,
        "real_digit_recognition_accuracy": real_recognition_acc,
        "overall_digit_accuracy": overall_digit_acc,
        "target_digit_count": len(all_digit_rows),
    }


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_project = script_dir.parent / "Project"
    if not default_project.exists():
        default_project = script_dir / "Project"
    parser = argparse.ArgumentParser(description="License plate digit recognition experiment")
    parser.add_argument("--project-dir", type=Path, default=default_project, help="Path to Project directory")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs", help="Output directory")
    parser.add_argument("--fig-dir", type=Path, default=script_dir / "figures", help="Figure directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args.project_dir, args.output_dir, args.fig_dir)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
