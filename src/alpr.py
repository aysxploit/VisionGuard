from __future__ import annotations
import cv2
import numpy as np
import pytesseract
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import logging

log = logging.getLogger("alpr")

@dataclass(slots=True)
class ProcConfig:
    min_plate_area: int
    max_plate_area: int
    gaussian_blur: int
    canny_low: int
    canny_high: int
    plate_aspect_low: float
    plate_aspect_high: float
    adaptive_thresh_block: int
    adaptive_thresh_C: int

def _read_proc_config(cfg: ConfigParser) -> ProcConfig:
    p = cfg["processing"]
    return ProcConfig(
        min_plate_area=p.getint("min_plate_area", 4500),
        max_plate_area=p.getint("max_plate_area", 250000),
        gaussian_blur=p.getint("gaussian_blur", 3),
        canny_low=p.getint("canny_low", 50),
        canny_high=p.getint("canny_high", 150),
        plate_aspect_low=p.getfloat("plate_aspect_low", 2.0),
        plate_aspect_high=p.getfloat("plate_aspect_high", 6.0),
        adaptive_thresh_block=p.getint("adaptive_thresh_block", 31),
        adaptive_thresh_C=p.getint("adaptive_thresh_C", 7),
    )

def set_tesseract_cmd(cfg: ConfigParser) -> None:
    cmd = cfg.get("app", "tesseract_cmd", fallback="").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

def preprocess(gray: np.ndarray, pcfg: ProcConfig) -> np.ndarray:
    if pcfg.gaussian_blur % 2 == 0:
        pcfg.gaussian_blur += 1
    blur = cv2.GaussianBlur(gray, (pcfg.gaussian_blur, pcfg.gaussian_blur), 0)
    edges = cv2.Canny(blur, pcfg.canny_low, pcfg.canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges

def find_plate_contours(edges: np.ndarray, pcfg: ProcConfig) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[np.ndarray] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < pcfg.min_plate_area or area > pcfg.max_plate_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if pcfg.plate_aspect_low <= aspect <= pcfg.plate_aspect_high:
            candidates.append(c)
    return candidates

def crop_plate(img: np.ndarray, contour: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    pad = int(0.05 * max(w, h))
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img.shape[1]); y1 = min(y + h + pad, img.shape[0])
    roi = img[y0:y1, x0:x1]
    return roi

def ocr_plate(roi: np.ndarray) -> Tuple[str, float]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    data = pytesseract.image_to_data(thr, config=config, output_type=pytesseract.Output.DICT)
    text = "".join(ch for ch in (data.get("text") and " ".join(data["text"]) or "").upper() if ch.isalnum())
    confs = [float(c) for c in data.get("conf", []) if c not in ("-1", "", None)]
    conf = float(np.mean(confs)) if confs else 0.0
    return text, conf

def detect_and_read(img_bgr: np.ndarray, pcfg: ProcConfig) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = preprocess(gray, pcfg)
    cnts = find_plate_contours(edges, pcfg)

    results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for c in cnts:
        roi = crop_plate(img_bgr, c)
        text, conf = ocr_plate(roi)
        if text:
            x, y, w, h = cv2.boundingRect(c)
            results.append((text, conf, (x, y, w, h)))
    return results

def annotate(img: np.ndarray, results: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
    out = img.copy()
    for text, conf, (x, y, w, h) in results:
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{text} ({conf:.0f})"
        cv2.putText(out, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def load_config(path: Path) -> tuple[ConfigParser, ProcConfig]:
    cfg = ConfigParser()
    cfg.read(path)
    return cfg, _read_proc_config(cfg)