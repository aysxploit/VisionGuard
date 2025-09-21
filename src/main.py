from __future__ import annotations
import argparse
import os
from pathlib import Path
import cv2
from configparser import ConfigParser
from dotenv import load_dotenv
import logging

from .logging_setup import setup_logging
from .db import DB
from .alpr import load_config, set_tesseract_cmd, detect_and_read, annotate
from .gemini_client import GeminiCleaner

log = logging.getLogger("main")

def _init(cfg_path: Path) -> tuple[ConfigParser, GeminiCleaner, DB, Path]:
    load_dotenv()
    cfg, pcfg = load_config(cfg_path)
    set_tesseract_cmd(cfg)
    logs_dir = Path(cfg.get("app", "snapshot_dir", fallback="data/logs")).resolve()
    setup_logging(logs_dir)
    db = DB(Path(cfg.get("app", "db_path", fallback="data/db/alpr.sqlite3")))
    g = GeminiCleaner(
        enabled=cfg.getboolean("gemini", "enable", fallback=False),
        model_name=cfg.get("gemini", "model", fallback="gemini-1.5-flash"),
    )
    return cfg, g, db, logs_dir

def run_image(image_path: Path, cfg_path: Path) -> None:
    cfg, g, db, logs = _init(cfg_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    _, pcfg = load_config(cfg_path)
    results = detect_and_read(img, pcfg)
    cleaned = []
    for text, conf, bbox in results:
        text2 = g.clean_plate(text)
        cleaned.append((text2, conf, bbox))
        db.insert_detection(text2, float(conf), f"image:{image_path.name}")
        log.info("Detection: %s conf=%.1f", text2, conf)
    out = annotate(img, cleaned)
    out_path = logs / f"annotated_{image_path.stem}.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"Detections: {[ (t,c) for t,c,_ in cleaned ]}")
    print(f"Annotated saved: {out_path}")

def run_camera(index: int, cfg_path: Path) -> None:
    cfg, g, db, logs = _init(cfg_path)
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {index}")
    _, pcfg = load_config(cfg_path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = detect_and_read(frame, pcfg)
            cleaned = []
            for text, conf, bbox in results:
                text2 = g.clean_plate(text)
                cleaned.append((text2, conf, bbox))
                if text2:
                    db.insert_detection(text2, float(conf), f"camera:{index}")
            vis = annotate(frame, cleaned)
            cv2.imshow("VisionGuard", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                p = logs / "snapshot.jpg"
                cv2.imwrite(str(p), frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_gui(cfg_path: Path) -> None:
    from .gui import App
    _cfg, _g, db, _logs = _init(cfg_path)
    App(cfg_path, db).run()

def main() -> None:
    ap = argparse.ArgumentParser(description="VisionGuard ALPR (Python 3.13)")
    ap.add_argument("--config", type=Path, default=Path("config.ini"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Path to image file")
    mode.add_argument("--camera", action="store_true", help="Use webcam")
    mode.add_argument("--gui", action="store_true", help="Launch GUI")
    ap.add_argument("--camera-index", type=int, default=0)
    args = ap.parse_args()

    if args.image:
        run_image(args.image, args.config)
    elif args.camera:
        run_camera(args.camera_index, args.config)
    else:
        run_gui(args.config)

if __name__ == "__main__":
    main()