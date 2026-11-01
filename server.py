#!/usr/bin/env python3
"""
Universal Dataset Studio & Polygon Lab Server
Flask-based backend for arbitrary folder selection, multi-dataset exploration,
interactive polygon editing, AI auto-labeling (Cross-Pseudo / YOLO-Seg forward pass),
noise pruning, and dataset repackaging.
"""

import os
import sys
import json
import time
import shutil
import glob
import uuid
import threading
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import yaml
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

# Workspace root & Config
STUDIO_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", str(STUDIO_DIR)))
CONFIG_FILE = Path.home() / ".dataset_studio_config.json"
PACKAGES_DIR = STUDIO_DIR / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)

# Default base dataset (Default to STUDIO_DIR/datasets or WORKSPACE_ROOT)
default_studio_ds = STUDIO_DIR / "datasets"
default_studio_ds.mkdir(parents=True, exist_ok=True)
DEFAULT_DATASET = default_studio_ds

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG", ".WEBP", ".BMP"}

# Known class information & color schemes (COCO 80 + Urban / Accessibility / Driving)
KNOWN_CLASSES = {
    # Urban & Accessibility
    "tactile_line": {"label_ko": "선형 점자", "color": "#FACC15"},
    "tactile_dot": {"label_ko": "점형 점자", "color": "#FB923C"},
    "stairs": {"label_ko": "계단", "color": "#E879F9"},
    "stair": {"label_ko": "계단", "color": "#E879F9"},
    "elevator": {"label_ko": "승강기", "color": "#38BDF8"},
    "crosswalk": {"label_ko": "횡단보도", "color": "#4ADE80"},
    "bollard": {"label_ko": "볼라드", "color": "#F87171"},
    "movable_signage": {"label_ko": "입간판", "color": "#F97316"},
    "signage": {"label_ko": "입간판", "color": "#F97316"},
    "facility": {"label_ko": "시설물", "color": "#A855F7"},
    "curb": {"label_ko": "연석/턱", "color": "#64748B"},
    "sidewalk": {"label_ko": "보도", "color": "#38BDF8"},
    "roadway": {"label_ko": "차도", "color": "#6B7280"},
    "elevator_sign": {"label_ko": "승강기 표지", "color": "#2DD4BF"},
    "toilet_sign": {"label_ko": "화장실 표지", "color": "#818CF8"},

    # Korean Direct Aliases
    "점자블록": {"label_ko": "선형 점자", "color": "#FACC15"},
    "선형점자": {"label_ko": "선형 점자", "color": "#FACC15"},
    "선형 점자": {"label_ko": "선형 점자", "color": "#FACC15"},
    "점형점자": {"label_ko": "점형 점자", "color": "#FB923C"},
    "점형 점자": {"label_ko": "점형 점자", "color": "#FB923C"},
    "계단": {"label_ko": "계단", "color": "#E879F9"},
    "승강기": {"label_ko": "승강기", "color": "#38BDF8"},
    "횡단보도": {"label_ko": "횡단보도", "color": "#4ADE80"},
    "볼라드": {"label_ko": "볼라드", "color": "#F87171"},
    "입간판": {"label_ko": "입간판", "color": "#F97316"},
    "시설물": {"label_ko": "시설물", "color": "#A855F7"},
    "자전거": {"label_ko": "자전거", "color": "#FDE047"},
    "이륜차": {"label_ko": "이륜차", "color": "#FDE047"},
    "오토바이": {"label_ko": "오토바이", "color": "#FB923C"},
    "승용차": {"label_ko": "승용차", "color": "#60A5FA"},
    "차량": {"label_ko": "차량", "color": "#94A3B8"},
    "보행자": {"label_ko": "보행자", "color": "#F472B6"},
    "신호등": {"label_ko": "신호등", "color": "#EF4444"},
    "연석": {"label_ko": "연석/턱", "color": "#64748B"},

    # Vehicles & Traffic
    "person": {"label_ko": "보행자", "color": "#F472B6"},
    "pedestrian": {"label_ko": "보행자", "color": "#F472B6"},
    "bicycle": {"label_ko": "이륜차/자전거", "color": "#FDE047"},
    "car": {"label_ko": "승용차", "color": "#60A5FA"},
    "motorcycle": {"label_ko": "오토바이", "color": "#FB923C"},
    "airplane": {"label_ko": "비행기", "color": "#93C5FD"},
    "bus": {"label_ko": "버스", "color": "#34D399"},
    "train": {"label_ko": "기차", "color": "#A78BFA"},
    "truck": {"label_ko": "트럭", "color": "#F59E0B"},
    "boat": {"label_ko": "보트", "color": "#38BDF8"},
    "vehicle": {"label_ko": "차량", "color": "#94A3B8"},
    "traffic_light": {"label_ko": "신호등", "color": "#EF4444"},
    "traffic light": {"label_ko": "신호등", "color": "#EF4444"},
    "fire hydrant": {"label_ko": "소화전", "color": "#DC2626"},
    "stop_sign": {"label_ko": "정지 표지", "color": "#F87171"},
    "stop sign": {"label_ko": "정지 표지", "color": "#F87171"},
    "parking meter": {"label_ko": "주차요금기", "color": "#9CA3AF"},
    "bench": {"label_ko": "벤치", "color": "#A3E635"},

    # Animals
    "bird": {"label_ko": "새", "color": "#F472B6"},
    "cat": {"label_ko": "고양이", "color": "#FB923C"},
    "dog": {"label_ko": "개", "color": "#FACC15"},
    "horse": {"label_ko": "말", "color": "#A16207"},
    "sheep": {"label_ko": "양", "color": "#E5E7EB"},
    "cow": {"label_ko": "소", "color": "#B45309"},
    "elephant": {"label_ko": "코끼리", "color": "#94A3B8"},
    "bear": {"label_ko": "곰", "color": "#78350F"},
    "zebra": {"label_ko": "얼룩말", "color": "#475569"},
    "giraffe": {"label_ko": "기린", "color": "#D97706"},

    # Accessories & Everyday Objects
    "backpack": {"label_ko": "배낭", "color": "#6366F1"},
    "umbrella": {"label_ko": "우산", "color": "#EC4899"},
    "handbag": {"label_ko": "핸드백", "color": "#D946EF"},
    "tie": {"label_ko": "넥타이", "color": "#8B5CF6"},
    "suitcase": {"label_ko": "여행가방", "color": "#06B6D4"},
    "bottle": {"label_ko": "병", "color": "#14B8A6"},
    "chair": {"label_ko": "의자", "color": "#10B981"},
    "couch": {"label_ko": "소파", "color": "#84CC16"},
    "potted plant": {"label_ko": "화분", "color": "#22C55E"},
    "potted_plant": {"label_ko": "화분", "color": "#22C55E"},
    "bed": {"label_ko": "침대", "color": "#0EA5E9"},
    "dining table": {"label_ko": "식탁", "color": "#3B82F6"},
    "dining_table": {"label_ko": "식탁", "color": "#3B82F6"},
    "toilet": {"label_ko": "변기", "color": "#6366F1"},
    "tv": {"label_ko": "TV/모니터", "color": "#8B5CF6"},
    "laptop": {"label_ko": "노트북", "color": "#A855F7"},
    "mouse": {"label_ko": "마우스", "color": "#D946EF"},
    "remote": {"label_ko": "리모컨", "color": "#EC4899"},
    "keyboard": {"label_ko": "키보드", "color": "#F43F5E"},
    "cell phone": {"label_ko": "스마트폰", "color": "#F97316"},
    "cell_phone": {"label_ko": "스마트폰", "color": "#F97316"},
    "book": {"label_ko": "책", "color": "#EAB308"},
    "clock": {"label_ko": "시계", "color": "#84CC16"},
}

FALLBACK_COLORS = [
    "#38BDF8", "#4ADE80", "#F87171", "#FACC15", "#A855F7",
    "#FB923C", "#F472B6", "#E879F9", "#2DD4BF", "#FDE047",
    "#94A3B8", "#60A5FA", "#34D399", "#F43F5E", "#A78BFA",
]


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "recent_folders": [str(DEFAULT_DATASET)],
        "last_opened": str(DEFAULT_DATASET)
    }


def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Warning] Failed to save config: {e}")


def add_recent_folder(folder_path):
    cfg = load_config()
    p_str = str(Path(folder_path).resolve())
    recents = cfg.get("recent_folders", [])
    if p_str in recents:
        recents.remove(p_str)
    recents.insert(0, p_str)
    cfg["recent_folders"] = recents[:15]
    cfg["last_opened"] = p_str
    save_config(cfg)


class TaskManager:
    """Manages asynchronous background jobs (dataset creation, auto-labeling, etc.)"""
    def __init__(self):
        self.tasks = {}
        self.lock = threading.Lock()

    def create_task(self, name, description=""):
        tid = str(uuid.uuid4())[:8]
        with self.lock:
            self.tasks[tid] = {
                "id": tid,
                "name": name,
                "description": description,
                "status": "pending",
                "progress": 0,
                "total": 0,
                "message": "작업 대기 중...",
                "logs": [],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "result": None,
                "error": None,
            }
        return tid

    def update_task(self, tid, **kwargs):
        with self.lock:
            if tid in self.tasks:
                self.tasks[tid].update(kwargs)

    def log(self, tid, message):
        timestamp = time.strftime("[%H:%M:%S]")
        log_line = f"{timestamp} {message}"
        print(f"[{tid}] {log_line}")
        with self.lock:
            if tid in self.tasks:
                self.tasks[tid]["logs"].append(log_line)
                self.tasks[tid]["message"] = message

    def get_task(self, tid):
        with self.lock:
            return self.tasks.get(tid, None)

    def list_tasks(self):
        with self.lock:
            return sorted(list(self.tasks.values()), key=lambda x: x["created_at"], reverse=True)


task_manager = TaskManager()


class DatasetState:
    """Encapsulates the state of the currently active dataset, accommodating arbitrary folders."""
    def __init__(self, dataset_path=None):
        self.lock = threading.Lock()
        self.dataset_path = None
        self.backup_dir = None
        self.excluded_dir = None
        self.edits_log_file = None
        self.classes = {}
        self.in_memory_history = None
        self.split_dirs = []

        if dataset_path:
            self.set_dataset(dataset_path)

    def set_dataset(self, new_path):
        with self.lock:
            path_obj = Path(new_path).resolve()
            if not path_obj.exists() or not path_obj.is_dir():
                raise ValueError(f"Dataset path does not exist: {new_path}")

            self.dataset_path = path_obj
            self.backup_dir = self.dataset_path / "labels_backup"
            self.excluded_dir = self.dataset_path / "excluded_images"
            self.edits_log_file = self.dataset_path / "edits_history.json"

            # Check directory layout
            img_dir = self.dataset_path / "images"
            lbl_dir = self.dataset_path / "labels"

            self.split_dirs = []
            if img_dir.exists() and (img_dir / "train").exists():
                self.split_dirs = ["train", "val"]
            elif img_dir.exists():
                self.split_dirs = ["."]
            else:
                # Raw folder containing images directly
                self.split_dirs = ["."]

            self.backup_dir.mkdir(parents=True, exist_ok=True)
            self.excluded_dir.mkdir(parents=True, exist_ok=True)
            for s in self.split_dirs:
                if s == ".":
                    (self.backup_dir / "root").mkdir(parents=True, exist_ok=True)
                    (self.excluded_dir / "images").mkdir(parents=True, exist_ok=True)
                    (self.excluded_dir / "labels").mkdir(parents=True, exist_ok=True)
                else:
                    (self.backup_dir / s).mkdir(parents=True, exist_ok=True)
                    (self.excluded_dir / "images" / s).mkdir(parents=True, exist_ok=True)
                    (self.excluded_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

            self.in_memory_history = None
            self._load_classes()
            add_recent_folder(self.dataset_path)

    def _load_classes(self):
        """
        Dynamically infer and load dataset class definitions from:
        1. Custom metadata file (classes_meta.json) if user previously edited classes
        2. YAML dataset definitions (data.yaml, dataset.yaml, data.yml, metadata.yaml)
        3. TXT class lists (classes.txt, labels.txt, names.txt, obj.names)
        4. JSON class lists (classes.json, labels.json, annotations.json)
        5. Actual YOLO label text files in labels/ or root (only used IDs)
        """
        dp = self.dataset_path
        classes = {}

        # 1. Check custom metadata saved from studio
        meta_json = dp / "classes_meta.json"
        if meta_json.exists():
            try:
                with open(meta_json, "r", encoding="utf-8") as f:
                    saved_meta = json.load(f)
                if isinstance(saved_meta, dict):
                    for cid_str, info in saved_meta.items():
                        cid = int(cid_str)
                        cname = info.get("name", f"class_{cid}")
                        known = KNOWN_CLASSES.get(cname, {})
                        classes[cid] = {
                            "name": cname,
                            "label_ko": info.get("label_ko") or known.get("label_ko", cname),
                            "color": info.get("color") or known.get("color", FALLBACK_COLORS[cid % len(FALLBACK_COLORS)])
                        }
            except Exception as e:
                print(f"[Warning] Failed to parse {meta_json}: {e}")

        # 2. Check YAML definitions
        if not classes:
            yaml_cands = [
                dp / "data.yaml", dp / "dataset.yaml", dp / "data.yml", dp / "dataset.yml",
                dp / "metadata.yaml", dp / "labels.yaml",
                dp.parent / "data.yaml", dp.parent / "dataset.yaml",
                dp.parent / "data.yml", dp.parent / "dataset.yml"
            ]
            for yp in yaml_cands:
                if yp.exists():
                    try:
                        with open(yp, "r", encoding="utf-8") as f:
                            ydata = yaml.safe_load(f)
                        if not isinstance(ydata, dict):
                            continue
                        names = ydata.get("names") or ydata.get("classes") or ydata.get("labels") or {}
                        if isinstance(names, list):
                            names = {i: name for i, name in enumerate(names)}
                        elif isinstance(names, dict):
                            names = {int(k): str(v) for k, v in names.items() if str(k).isdigit()}
                        else:
                            names = {}

                        for cid, cname in sorted(names.items()):
                            cname_str = str(cname).strip()
                            known = KNOWN_CLASSES.get(cname_str, {})
                            label_ko = known.get("label_ko", cname_str)
                            color = known.get("color", FALLBACK_COLORS[cid % len(FALLBACK_COLORS)])
                            classes[cid] = {
                                "name": cname_str,
                                "label_ko": label_ko,
                                "color": color
                            }
                        if classes:
                            break
                    except Exception as e:
                        print(f"[Warning] Failed to parse {yp}: {e}")

        # 3. Check TXT class lists
        if not classes:
            txt_cands = [
                dp / "classes.txt", dp / "labels.txt", dp / "names.txt", dp / "obj.names",
                dp / "labels/classes.txt"
            ]
            for tp in txt_cands:
                if tp.exists():
                    try:
                        with open(tp, "r", encoding="utf-8") as f:
                            lines = [line.strip() for line in f if line.strip()]
                        for cid, cname in enumerate(lines):
                            known = KNOWN_CLASSES.get(cname, {})
                            label_ko = known.get("label_ko", cname)
                            color = known.get("color", FALLBACK_COLORS[cid % len(FALLBACK_COLORS)])
                            classes[cid] = {
                                "name": cname,
                                "label_ko": label_ko,
                                "color": color
                            }
                        if classes:
                            break
                    except Exception as e:
                        print(f"[Warning] Failed to parse {tp}: {e}")

        # 4. Check JSON class lists
        if not classes:
            json_cands = [dp / "classes.json", dp / "labels.json", dp / "dataset_info.json"]
            for jp in json_cands:
                if jp.exists():
                    try:
                        with open(jp, "r", encoding="utf-8") as f:
                            jdata = json.load(f)
                        if isinstance(jdata, list):
                            for cid, cname in enumerate(jdata):
                                cname_str = str(cname)
                                known = KNOWN_CLASSES.get(cname_str, {})
                                classes[cid] = {
                                    "name": cname_str,
                                    "label_ko": known.get("label_ko", cname_str),
                                    "color": known.get("color", FALLBACK_COLORS[cid % len(FALLBACK_COLORS)])
                                }
                        elif isinstance(jdata, dict):
                            raw_names = jdata.get("classes") or jdata.get("names") or jdata.get("categories")
                            if isinstance(raw_names, list):
                                for cid, item in enumerate(raw_names):
                                    cname = item.get("name") if isinstance(item, dict) else str(item)
                                    known = KNOWN_CLASSES.get(cname, {})
                                    classes[cid] = {
                                        "name": cname,
                                        "label_ko": known.get("label_ko", cname),
                                        "color": known.get("color", FALLBACK_COLORS[cid % len(FALLBACK_COLORS)])
                                    }
                        if classes:
                            break
                    except Exception as e:
                        print(f"[Warning] Failed to parse {jp}: {e}")

        # 5. Scan actual label files for used class IDs (only if no explicit metadata exists)
        if not classes:
            found_cids = set()
            for s in self.split_dirs:
                lbl_d = dp / ("labels/" + s if s != "." else "labels")
                if not lbl_d.exists():
                    lbl_d = dp
                for lp in list(lbl_d.glob("*.txt"))[:200]:
                    try:
                        with open(lp, "r", encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5 and parts[0].isdigit():
                                    found_cids.add(int(parts[0]))
                    except Exception:
                        pass

            for cid in sorted(found_cids):
                name = f"class_{cid}"
                known = KNOWN_CLASSES.get(name, {})
                classes[cid] = {
                    "name": name,
                    "label_ko": known.get("label_ko", f"클래스 {cid}"),
                    "color": FALLBACK_COLORS[cid % len(FALLBACK_COLORS)]
                }

        self.classes = classes

        self.classes = classes

    def load_edits_history(self):
        with self.lock:
            if self.edits_log_file.exists():
                for _ in range(3):
                    try:
                        with open(self.edits_log_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                self.in_memory_history = data
                                return data
                    except Exception:
                        time.sleep(0.05)
                bak = self.edits_log_file.with_suffix(".bak")
                if bak.exists():
                    try:
                        with open(bak, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                self.in_memory_history = data
                                return data
                    except Exception:
                        pass
            if self.in_memory_history is not None:
                return self.in_memory_history.copy()
            return {}

    def save_edits_history(self, history):
        with self.lock:
            self.in_memory_history = history.copy()
            tmp_file = self.edits_log_file.with_suffix(".tmp")
            bak_file = self.edits_log_file.with_suffix(".bak")
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                if self.edits_log_file.exists() and self.edits_log_file.stat().st_size > 0:
                    shutil.copy2(self.edits_log_file, bak_file)
                os.replace(tmp_file, self.edits_log_file)
            except Exception as e:
                print(f"[Error saving edits history] {e}")


state = DatasetState()


# =========================================================================
# File System Browser & Folder Management APIs
# =========================================================================
@app.route("/api/browse_fs")
def browse_fs():
    """Browse arbitrary folders on the host machine to select or inspect datasets."""
    req_path = request.args.get("path", "").strip()
    if not req_path:
        curr = WORKSPACE_ROOT if WORKSPACE_ROOT.exists() else Path.home()
    else:
        curr = Path(req_path).resolve()

    if not curr.exists() or not curr.is_dir():
        curr = WORKSPACE_ROOT if WORKSPACE_ROOT.exists() else Path.home()

    subdirs = []
    try:
        for p in sorted(curr.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if p.name.startswith(".") or p.name in ("__pycache__", "node_modules", ".git"):
                continue
            if p.is_dir():
                has_imgs = (p / "images").exists() or any(p.glob("*.jpg")) or any(p.glob("*.png"))
                has_lbls = (p / "labels").exists() or any(p.glob("*.txt"))
                has_yaml = (p / "data.yaml").exists()
                subdirs.append({
                    "name": p.name,
                    "path": str(p),
                    "is_yolo_dataset": (has_imgs and has_lbls),
                    "has_yaml": has_yaml,
                    "has_images": has_imgs,
                })
    except PermissionError:
        pass

    # Inspect current folder details
    img_count = 0
    lbl_count = 0
    img_dir = curr / "images"
    lbl_dir = curr / "labels"
    if img_dir.exists():
        for f in img_dir.rglob("*"):
            if f.suffix in IMAGE_EXTS:
                img_count += 1
    else:
        for f in curr.glob("*"):
            if f.suffix in IMAGE_EXTS:
                img_count += 1

    if lbl_dir.exists():
        lbl_count = len(list(lbl_dir.rglob("*.txt")))
    else:
        lbl_count = len(list(curr.glob("*.txt")))

    is_yolo = (curr / "data.yaml").exists() or (img_dir.exists() and lbl_dir.exists())

    return jsonify({
        "current_path": str(curr),
        "parent_path": str(curr.parent) if curr != curr.parent else None,
        "is_yolo_dataset": is_yolo,
        "raw_images_count": img_count,
        "labels_count": lbl_count,
        "has_yaml": (curr / "data.yaml").exists(),
        "subdirs": subdirs,
    })


@app.route("/api/open_folder", methods=["POST"])
def open_folder():
    """Open an arbitrary folder as a dataset, initializing it if it's a raw image folder."""
    data = request.get_json() or {}
    folder_path = data.get("path")
    init_if_raw = data.get("init_if_raw", False)

    if not folder_path:
        return jsonify({"error": "Folder path is required"}), 400

    p = Path(folder_path).resolve()
    if not p.exists() or not p.is_dir():
        return jsonify({"error": f"Folder not found: {folder_path}"}), 404

    # If raw images only and user requested init
    img_files = [f for f in p.glob("*") if f.suffix in IMAGE_EXTS]
    has_layout = (p / "images").exists() and (p / "labels").exists()

    if not has_layout and img_files and init_if_raw:
        # Organize into YOLO standard structure (8:2 train/val)
        train_img_dir = p / "images/train"
        val_img_dir = p / "images/val"
        train_lbl_dir = p / "labels/train"
        val_lbl_dir = p / "labels/val"

        for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
            d.mkdir(parents=True, exist_ok=True)

        total = len(img_files)
        val_count = max(1, int(total * 0.2))

        for idx, im in enumerate(img_files):
            split = "val" if idx < val_count else "train"
            dest_img = p / "images" / split / im.name
            dest_lbl = p / "labels" / split / f"{im.stem}.txt"
            shutil.move(str(im), str(dest_img))
            if not dest_lbl.exists():
                dest_lbl.touch()

        # Create basic data.yaml
        yaml_cfg = {
            "path": str(p),
            "train": "images/train",
            "val": "images/val",
            "nc": 11,
            "names": {i: KNOWN_CLASSES.get(k, {}).get("label_ko", k) for i, k in enumerate(KNOWN_CLASSES.keys()) if i < 11}
        }
        with open(p / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_cfg, f, sort_keys=False, allow_unicode=True)

    try:
        state.set_dataset(p)
        return jsonify({
            "success": True,
            "current": str(state.dataset_path),
            "classes": state.classes,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/recent_folders")
def get_recent_folders():
    cfg = load_config()
    recents = cfg.get("recent_folders", [])
    valid_recents = []
    for r in recents:
        rp = Path(r)
        if rp.exists() and rp.is_dir():
            valid_recents.append({
                "path": str(rp),
                "name": rp.name,
                "is_current": (str(rp) == str(state.dataset_path))
            })
    return jsonify({"recent_folders": valid_recents})


# =========================================================================
# Dataset Discovery & Utilities
# =========================================================================
def discover_all_datasets():
    """Scan workspace and recent paths for all valid YOLO dataset directories."""
    candidates = []
    search_dirs = [
        STUDIO_DIR / "datasets",
        WORKSPACE_ROOT,
    ]

    seen_paths = set()

    # Include recent folders
    cfg = load_config()
    for r in cfg.get("recent_folders", []):
        rp = Path(r)
        if rp.exists() and rp.is_dir():
            search_dirs.append(rp)

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        if (s_dir / "images").exists() and (s_dir / "labels").exists():
            dirs_to_check = [s_dir]
        else:
            dirs_to_check = [p for p in s_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]

        for p in dirs_to_check:
            p_resolved = p.resolve()
            if p_resolved in seen_paths:
                continue
            seen_paths.add(p_resolved)

            img_dir = p / "images"
            lbl_dir = p / "labels"
            if img_dir.exists() and lbl_dir.exists():
                train_imgs = sum(1 for f in (img_dir / "train").glob("*") if f.suffix in IMAGE_EXTS) if (img_dir / "train").exists() else 0
                val_imgs = sum(1 for f in (img_dir / "val").glob("*") if f.suffix in IMAGE_EXTS) if (img_dir / "val").exists() else 0
                train_lbls = len(list((lbl_dir / "train").glob("*.txt"))) if (lbl_dir / "train").exists() else 0
                val_lbls = len(list((lbl_dir / "val").glob("*.txt"))) if (lbl_dir / "val").exists() else 0

                yaml_path = p / "data.yaml"
                nc = 0
                class_names = []
                if yaml_path.exists():
                    try:
                        with open(yaml_path, "r", encoding="utf-8") as f:
                            yd = yaml.safe_load(f)
                            nc = yd.get("nc", 0)
                            names = yd.get("names", {})
                            if isinstance(names, list):
                                class_names = names
                            elif isinstance(names, dict):
                                class_names = [names[k] for k in sorted(names.keys())]
                    except Exception:
                        pass

                candidates.append({
                    "path": str(p),
                    "name": p.name,
                    "parent": p.parent.name,
                    "train_images": train_imgs,
                    "val_images": val_imgs,
                    "total_images": train_imgs + val_imgs,
                    "train_labels": train_lbls,
                    "val_labels": val_lbls,
                    "num_classes": nc,
                    "class_names": class_names,
                    "is_active": (p_resolved == state.dataset_path),
                    "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
                })

    candidates.sort(key=lambda x: (not x["is_active"], -x["total_images"]))
    return candidates


WEIGHT_METADATA_CACHE = {}


def get_weight_metadata(weight_path):
    """Safely inspect YOLO .pt checkpoint metadata with mtime-based caching."""
    p = Path(weight_path)
    if not p.exists():
        return {"names": [], "nc": 0, "fitness": None, "epoch": None}

    mtime = p.stat().st_mtime
    cached = WEIGHT_METADATA_CACHE.get(str(p))
    if cached and cached.get("mtime") == mtime:
        return cached

    names_list = []
    fitness = None
    epoch = None
    try:
        import torch
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            m = ckpt.get("model") or ckpt.get("ema")
            names = getattr(m, "names", None) if m else ckpt.get("names")
            raw_fitness = ckpt.get("best_fitness")
            epoch = ckpt.get("epoch")

            if raw_fitness is not None:
                try:
                    fitness = round(float(raw_fitness), 4)
                except Exception:
                    fitness = None

            if isinstance(names, dict):
                names_list = [str(v).strip() for k, v in sorted(names.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))]
            elif isinstance(names, list):
                names_list = [str(x).strip() for x in names]
    except Exception:
        pass

    data = {
        "mtime": mtime,
        "names": names_list,
        "nc": len(names_list),
        "fitness": fitness,
        "epoch": epoch
    }
    WEIGHT_METADATA_CACHE[str(p)] = data
    return data


def discover_available_weights(target_dataset_classes=None):
    """
    Scan workspace for available .pt weights, extract model metadata,
    and rank models using a recommendation score based on class overlap and performance metrics.
    """
    target_names = set()
    if target_dataset_classes:
        for info in target_dataset_classes.values():
            if isinstance(info, dict):
                if info.get("name"):
                    target_names.add(info["name"].strip().lower())
                if info.get("label_ko"):
                    target_names.add(info["label_ko"].strip().lower())

    results = []
    seen_paths = set()
    
    search_dirs = [STUDIO_DIR, STUDIO_DIR.parent]
    if state.dataset_path:
        search_dirs.extend([state.dataset_path, state.dataset_path.parent])

    weight_files = []
    seen_wf = set()
    for s_dir in search_dirs:
        if not s_dir or not s_dir.exists():
            continue
        try:
            for wf in s_dir.glob("**/*.pt"):
                if ".git" in str(wf) or ".cache" in str(wf):
                    continue
                r = wf.resolve()
                if r not in seen_wf:
                    seen_wf.add(r)
                    weight_files.append(wf)
        except Exception:
            pass

    for wf in weight_files:
        if ".git" in str(wf) or ".cache" in str(wf):
            continue
        p_resolved = wf.resolve()
        if p_resolved in seen_paths:
            continue
        seen_paths.add(p_resolved)

        size_mb = round(wf.stat().st_size / (1024 * 1024), 1)
        try:
            rel_path = str(wf.relative_to(STUDIO_DIR.parent))
        except ValueError:
            try:
                rel_path = str(wf.relative_to(WORKSPACE_ROOT))
            except ValueError:
                rel_path = wf.name

        meta = get_weight_metadata(wf)
        model_classes = meta["names"]
        fitness = meta["fitness"]
        epoch = meta["epoch"]

        # Calculate class overlap
        matched = []
        if target_names and model_classes:
            for mc in model_classes:
                mc_lower = mc.lower()
                known_ko = KNOWN_CLASSES.get(mc_lower, {}).get("label_ko", "").lower()
                if mc_lower in target_names or (known_ko and known_ko in target_names):
                    matched.append(mc)

        match_count = len(matched)
        target_count = len(target_names) if target_names else 1
        match_ratio = round((match_count / max(1, target_count)) * 100, 1) if target_names else 0

        score = 0.0
        if match_count > 0:
            score += match_ratio * 1.5
            if fitness is not None:
                score += min(35.0, fitness * 35.0)
            if "active" in str(wf).lower() or "best" in wf.name.lower():
                score += 15.0
            if "seg" in wf.name.lower():
                score += 10.0
        elif not target_names:
            if "seg" in wf.name.lower():
                score += 20.0
            if fitness is not None:
                score += fitness * 20.0

        results.append({
            "path": str(wf),
            "filename": wf.name,
            "relative_path": rel_path,
            "size_mb": size_mb,
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(wf.stat().st_mtime)),
            "num_classes": meta["nc"],
            "classes": model_classes,
            "fitness": fitness,
            "epoch": epoch,
            "matched_classes": matched,
            "matched_count": match_count,
            "match_ratio": match_ratio,
            "score": round(score, 1),
        })

    results.sort(key=lambda x: (-x["score"], -(x["fitness"] or 0), -x["size_mb"]))

    for rank, item in enumerate(results):
        if rank == 0 and item["score"] > 0 and item["matched_count"] > 0:
            item["is_recommended"] = True
            fit_str = f" | mAP {item['fitness']}" if item["fitness"] else ""
            item["recommend_badge"] = f"🎯 추천 ({item['match_ratio']}% 일치{fit_str})"
            item["recommend_reason"] = f"현재 데이터셋 클래스 중 {item['matched_count']}개를 지원하며 성능 지표({item['fitness'] or '우수'})가 가장 높습니다."
        elif item["matched_count"] > 0 and item["match_ratio"] >= 40:
            item["is_recommended"] = False
            item["recommend_badge"] = f"⭐ 호환 ({item['match_ratio']}% 일치)"
            item["recommend_reason"] = f"현재 데이터셋 클래스 {item['matched_count']}개 호환"
        else:
            item["is_recommended"] = False
            item["recommend_badge"] = ""
            item["recommend_reason"] = ""

    return results


# =========================================================================
# Web UI Routes
# =========================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/benchmark")
def benchmark():
    bench_html = Path(__file__).parent / "benchmark_20.html"
    if bench_html.exists():
        with open(bench_html, "r", encoding="utf-8") as f:
            return f.read()
    return "Benchmark gallery not found", 404


# =========================================================================
# Dataset Management APIs
# =========================================================================
@app.route("/api/datasets")
def list_datasets():
    datasets = discover_all_datasets()
    return jsonify({
        "current": str(state.dataset_path) if state.dataset_path else None,
        "total": len(datasets),
        "datasets": datasets,
    })


@app.route("/api/dataset/current")
def get_current_dataset_info():
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({
            "path": None,
            "name": "선택된 디렉토리 없음",
            "classes": {},
            "train_images": 0,
            "val_images": 0,
            "total_images": 0,
            "is_empty": True
        })

    p = state.dataset_path
    img_dir = p / "images"

    train_imgs = sum(1 for f in (img_dir / "train").glob("*") if f.suffix in IMAGE_EXTS) if (img_dir / "train").exists() else 0
    val_imgs = sum(1 for f in (img_dir / "val").glob("*") if f.suffix in IMAGE_EXTS) if (img_dir / "val").exists() else 0
    if train_imgs == 0 and val_imgs == 0:
        # Check direct images
        train_imgs = sum(1 for f in p.glob("*") if f.suffix in IMAGE_EXTS)

    history = state.load_edits_history()
    active_history = {k: v for k, v in history.items() if v.get("status") != "excluded"}

    return jsonify({
        "path": str(p),
        "name": p.name,
        "classes": state.classes,
        "train_images": train_imgs,
        "val_images": val_imgs,
        "total_images": train_imgs + val_imgs,
        "edited_count": len(active_history),
    })


@app.route("/api/dataset/select", methods=["POST"])
def select_dataset():
    data = request.get_json() or {}
    path = data.get("path")
    if not path:
        return jsonify({"error": "Missing dataset path"}), 400

    try:
        state.set_dataset(path)
        return jsonify({
            "success": True,
            "current": str(state.dataset_path),
            "classes": state.classes,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/available_models")
def get_available_models():
    weights = discover_available_weights(target_dataset_classes=state.classes)
    best_model = None
    for w in weights:
        if w.get("is_recommended"):
            best_model = w
            break

    return jsonify({
        "total": len(weights),
        "models": weights,
        "recommended_model": best_model,
        "active_dataset_classes_count": len(state.classes)
    })


# =========================================================================
# Asynchronous Dataset Creation Jobs
# =========================================================================
@app.route("/api/dataset/create", methods=["POST"])
def create_dataset_job():
    data = request.get_json() or {}
    mode = data.get("mode")
    name = data.get("name", "").strip()
    target_base = data.get("target_base", "").strip()

    if not name:
        return jsonify({"error": "데이터셋 이름을 입력해주세요."}), 400

    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)

    if target_base and Path(target_base).exists():
        out_dir = Path(target_base) / safe_name
    elif state.dataset_path and state.dataset_path.exists():
        out_dir = state.dataset_path.parent / safe_name
    else:
        out_dir = STUDIO_DIR / "datasets" / safe_name

    if out_dir.exists():
        return jsonify({"error": f"이미 존재하는 데이터셋 경로입니다: {out_dir}"}), 400

    tid = task_manager.create_task(f"데이터셋 생성: {safe_name} ({mode})")

    if mode == "ai_auto_label":
        thread = threading.Thread(target=_worker_ai_auto_label, args=(tid, data, out_dir))
    elif mode == "filter_clone":
        thread = threading.Thread(target=_worker_filter_clone, args=(tid, data, out_dir))
    elif mode == "empty":
        thread = threading.Thread(target=_worker_create_empty, args=(tid, data, out_dir))
    else:
        return jsonify({"error": f"Unknown mode: {mode}"}), 400

    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "task_id": tid, "target_dir": str(out_dir)})


def _worker_create_empty(tid, data, out_dir):
    try:
        task_manager.update_task(tid, status="running", progress=10)
        task_manager.log(tid, f"빈 데이터셋 디렉토리 생성 중: {out_dir}")

        for s in ("train", "val"):
            (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

        class_names = data.get("classes", [])
        if not class_names:
            class_names = [f"class_{i}" for i in range(11)]

        yaml_cfg = {
            "path": str(out_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": {i: name for i, name in enumerate(class_names)},
        }
        with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_cfg, f, sort_keys=False, allow_unicode=True)

        task_manager.log(tid, f"✅ data.yaml 생성 완료 (총 {len(class_names)}개 클래스)")
        task_manager.update_task(tid, status="success", progress=100, result={"path": str(out_dir)})
    except Exception as e:
        task_manager.log(tid, f"❌ 오류 발생: {str(e)}")
        task_manager.update_task(tid, status="failed", error=str(e))


def _worker_filter_clone(tid, data, out_dir):
    try:
        source_dir = Path(data.get("source_dataset", str(state.dataset_path)))
        keep_classes = data.get("keep_classes", None)
        prune_small_poly = data.get("prune_20px", True)
        min_pixels = data.get("min_pixels", 20.0)

        task_manager.update_task(tid, status="running", progress=5)
        task_manager.log(tid, f"원본 데이터셋 복제 및 정제 시작: {source_dir} -> {out_dir}")

        for s in ("train", "val"):
            (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

        if (source_dir / "data.yaml").exists():
            shutil.copy2(source_dir / "data.yaml", out_dir / "data.yaml")

        img_files = []
        for s in ("train", "val"):
            src_img_dir = source_dir / "images" / s
            if src_img_dir.exists():
                for ip in sorted(src_img_dir.glob("*")):
                    if ip.suffix in IMAGE_EXTS:
                        img_files.append((s, ip))

        total = len(img_files)
        task_manager.update_task(tid, total=total)
        task_manager.log(tid, f"총 처리할 이미지: {total}장")

        filtered_poly_count = 0
        pruned_noise_count = 0
        kept_poly_count = 0

        for idx, (split, img_p) in enumerate(img_files):
            stem = img_p.stem
            lbl_p = source_dir / "labels" / split / f"{stem}.txt"

            dest_img = out_dir / "images" / split / img_p.name
            dest_lbl = out_dir / "labels" / split / f"{stem}.txt"

            shutil.copy2(img_p, dest_img)

            new_lines = []
            if lbl_p.exists():
                im_w, im_h = 640, 640
                if prune_small_poly:
                    try:
                        with Image.open(img_p) as im:
                            im_w, im_h = im.size
                    except Exception:
                        pass

                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 7:
                            continue
                        cid = int(parts[0])
                        if keep_classes is not None and cid not in keep_classes:
                            filtered_poly_count += 1
                            continue

                        coords = [float(x) for x in parts[1:]]
                        if prune_small_poly:
                            pts_px = np.array([[coords[i] * im_w, coords[i+1] * im_h] for i in range(0, len(coords)-1, 2)], dtype=np.float32)
                            area = cv2.contourArea(pts_px)
                            if area < min_pixels:
                                pruned_noise_count += 1
                                continue

                        new_lines.append(line.strip())
                        kept_poly_count += 1

            with open(dest_lbl, "w", encoding="utf-8") as f:
                if new_lines:
                    f.write("\n".join(new_lines) + "\n")
                else:
                    f.write("")

            if idx % 200 == 0 or idx == total - 1:
                prog = int(10 + (idx / max(1, total)) * 85)
                task_manager.update_task(tid, progress=prog)
                task_manager.log(tid, f"진행 중: {idx}/{total}장 완료... (보존: {kept_poly_count}, 제거: {pruned_noise_count})")

        task_manager.log(tid, f"✅ 정제 복제 완료! 보존: {kept_poly_count}, 제거: {pruned_noise_count}")
        task_manager.update_task(tid, status="success", progress=100, result={"path": str(out_dir)})
    except Exception as e:
        task_manager.log(tid, f"❌ 오류 발생: {str(e)}")
        task_manager.update_task(tid, status="failed", error=str(e))


def _worker_ai_auto_label(tid, data, out_dir):
    try:
        from ultralytics import YOLO

        source_dataset = Path(data.get("source_dataset", str(state.dataset_path)))
        weight_path = Path(data.get("weight_path"))
        conf_thres = float(data.get("conf_threshold", 0.50))
        batch_size = int(data.get("batch_size", 16))
        device = data.get("device", "0")

        task_manager.update_task(tid, status="running", progress=5)
        task_manager.log(tid, f"AI 자동 의사라벨 생성 시작 (모델: {weight_path.name}, Conf: {conf_thres})")

        for s in ("train", "val"):
            (out_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

        task_manager.log(tid, f"YOLO 가중치 로드 중: {weight_path}")
        model = YOLO(str(weight_path))

        img_records = []
        for s in ("train", "val"):
            src_img_dir = source_dataset / "images" / s
            if src_img_dir.exists():
                for ip in sorted(src_img_dir.glob("*")):
                    if ip.suffix in IMAGE_EXTS:
                        img_records.append((s, ip))
        if not img_records:
            # Check direct images
            for ip in sorted(source_dataset.glob("*")):
                if ip.suffix in IMAGE_EXTS:
                    img_records.append(("train", ip))

        total = len(img_records)
        task_manager.update_task(tid, total=total)
        task_manager.log(tid, f"대상 이미지: 총 {total}장 배치 추론 시작...")

        generated_poly_count = 0

        for b_start in range(0, total, batch_size):
            batch = img_records[b_start : b_start + batch_size]
            b_paths = [str(r[1]) for r in batch]

            preds = model.predict(
                b_paths,
                imgsz=640,
                conf=conf_thres,
                device=device,
                retina_masks=False,
                verbose=False,
            )

            for (split, src_ip), res in zip(batch, preds):
                dst_img = out_dir / "images" / split / src_ip.name
                dst_lbl = out_dir / "labels" / split / f"{src_ip.stem}.txt"

                shutil.copy2(src_ip, dst_img)

                lines = []
                if res.masks is not None and res.boxes is not None:
                    boxes_cls = res.boxes.cls.cpu().numpy().astype(int)
                    boxes_conf = res.boxes.conf.cpu().numpy().astype(float)
                    masks_xyn = res.masks.xyn

                    for cid, conf, xyn in zip(boxes_cls, boxes_conf, masks_xyn):
                        if len(xyn) < 3:
                            continue
                        pts = (xyn * 1000.0).astype(np.float32)
                        peri = cv2.arcLength(pts, True)
                        approx = cv2.approxPolyDP(pts, 0.0015 * peri, True)
                        if len(approx) < 3:
                            approx = pts
                        norm_pts = np.clip(approx.reshape(-1, 2) / 1000.0, 0.0, 1.0)
                        coords = norm_pts.flatten().tolist()
                        coord_str = " ".join(f"{v:.6f}" for v in coords)
                        lines.append(f"{cid} {coord_str}")
                        generated_poly_count += 1

                with open(dst_lbl, "w", encoding="utf-8") as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")
                    else:
                        f.write("")

            prog = int(10 + (min(total, b_start + batch_size) / max(1, total)) * 85)
            task_manager.update_task(tid, progress=prog)
            if b_start % (batch_size * 5) == 0 or (b_start + batch_size) >= total:
                task_manager.log(tid, f"추론 진행: {min(total, b_start + batch_size)}/{total}장 완료 (생성 폴리곤: {generated_poly_count}개)")

        model_names = model.names if hasattr(model, "names") else {}
        yaml_cfg = {
            "path": str(out_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(model_names),
            "names": model_names,
        }
        with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_cfg, f, sort_keys=False, allow_unicode=True)

        task_manager.log(tid, f"✅ AI 자동 라벨링 완료: 총 {generated_poly_count}개 폴리곤 생성 및 저장됨")
        task_manager.update_task(tid, status="success", progress=100, result={"path": str(out_dir)})
    except Exception as e:
        task_manager.log(tid, f"❌ 추론 중 오류 발생: {str(e)}")
        task_manager.update_task(tid, status="failed", error=str(e))


@app.route("/api/tasks/<tid>")
def get_task_status(tid):
    t = task_manager.get_task(tid)
    if not t:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(t)


# =========================================================================
# Noise Pruning API (20px Micro Noise Filter)
# =========================================================================
@app.route("/api/dataset/prune_noise", methods=["POST"])
def prune_noise():
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    min_pixels = float(request.args.get("min_pixels", 20.0))
    ts = int(time.time())

    pruned_poly_count = 0
    modified_files = 0
    total_files = 0

    for split in state.split_dirs:
        lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
        img_dir = state.dataset_path / ("images/" + split if split != "." else "images")
        if not lbl_dir.exists():
            lbl_dir = state.dataset_path
        if not img_dir.exists():
            img_dir = state.dataset_path

        for lbl_p in sorted(lbl_dir.glob("*.txt")):
            total_files += 1
            # Match image with any image extension
            img_candidates = [img_dir / f"{lbl_p.stem}{ext}" for ext in IMAGE_EXTS if (img_dir / f"{lbl_p.stem}{ext}").exists()]
            im_w, im_h = 640, 640
            if img_candidates:
                try:
                    with Image.open(img_candidates[0]) as im:
                        im_w, im_h = im.size
                except Exception:
                    pass

            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]

            new_lines = []
            has_pruned = False

            for line in lines:
                parts = line.split()
                if len(parts) < 7:
                    has_pruned = True
                    pruned_poly_count += 1
                    continue
                coords = [float(x) for x in parts[1:]]
                pts_px = np.array([[coords[i] * im_w, coords[i+1] * im_h] for i in range(0, len(coords)-1, 2)], dtype=np.float32)
                area = cv2.contourArea(pts_px)
                if area < min_pixels:
                    has_pruned = True
                    pruned_poly_count += 1
                    continue
                new_lines.append(line)

            if has_pruned:
                backup_sub = split if split != "." else "root"
                backup_p = state.backup_dir / backup_sub / f"{lbl_p.stem}_{ts}.txt"
                shutil.copy2(lbl_p, backup_p)

                with open(lbl_p, "w", encoding="utf-8") as f:
                    if new_lines:
                        f.write("\n".join(new_lines) + "\n")
                    else:
                        f.write("")
                modified_files += 1

    return jsonify({
        "success": True,
        "pruned_poly_count": pruned_poly_count,
        "modified_files": modified_files,
        "total_files": total_files,
        "message": f"총 {modified_files}개 파일에서 {pruned_poly_count}개의 20px 극소 노이즈 폴리곤을 안전하게 제거했습니다."
    })


# =========================================================================
# Polygon Editor Core APIs
# =========================================================================
@app.route("/api/classes")
def get_classes():
    return jsonify(state.classes)


@app.route("/api/classes/save", methods=["POST"])
def save_classes():
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    data = request.get_json() or {}
    new_classes = data.get("classes", {})
    if not isinstance(new_classes, dict):
        return jsonify({"error": "유효하지 않은 클래스 데이터입니다."}), 400

    yaml_path = state.dataset_path / "data.yaml"
    names_dict = {}
    classes_meta = {}

    for cid_str, info in sorted(new_classes.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        cname = str(info.get("name", f"class_{cid}")).strip()
        label_ko = str(info.get("label_ko", cname)).strip()
        color = info.get("color", FALLBACK_COLORS[cid % len(FALLBACK_COLORS)])

        names_dict[cid] = cname
        classes_meta[cid] = {
            "name": cname,
            "label_ko": label_ko,
            "color": color
        }

    train_path = "images/train" if (state.dataset_path / "images/train").exists() else ("images" if (state.dataset_path / "images").exists() else ".")
    val_path = "images/val" if (state.dataset_path / "images/val").exists() else ("images" if (state.dataset_path / "images").exists() else ".")

    yaml_data = {
        "path": str(state.dataset_path),
        "train": train_path,
        "val": val_path,
        "nc": len(names_dict),
        "names": names_dict
    }

    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)

        meta_path = state.dataset_path / "classes_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(classes_meta, f, ensure_ascii=False, indent=2)

        # Reload classes in current state
        state._load_classes()
        return jsonify({
            "success": True,
            "classes": state.classes,
            "message": f"총 {len(state.classes)}개 클래스 설정이 data.yaml에 저장되었습니다."
        })
    except Exception as e:
        return jsonify({"error": f"클래스 저장 실패: {str(e)}"}), 500


@app.route("/api/images")
def list_images():
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"total": 0, "items": []})

    filter_cid = request.args.get("class_id", type=int)
    filter_dom = request.args.get("domain", "")
    filter_status = request.args.get("status", "all")
    query = request.args.get("query", "").strip().lower()

    history = state.load_edits_history()
    results = []

    for split in state.split_dirs:
        lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
        img_dir = state.dataset_path / ("images/" + split if split != "." else "images")
        if not lbl_dir.exists():
            lbl_dir = state.dataset_path
        if not img_dir.exists():
            img_dir = state.dataset_path

        for lbl_p in sorted(lbl_dir.glob("*.txt")):
            stem = lbl_p.stem
            dom = stem.split("_")[0]
            if filter_dom and dom != filter_dom:
                continue
            if query and query not in stem.lower():
                continue

            cids = set()
            poly_count = 0
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        cid = int(parts[0])
                        cids.add(cid)
                        poly_count += 1

            if filter_cid is not None and filter_cid not in cids:
                continue

            is_edited = (stem in history) or (lbl_p.stat().st_mtime > 86400 * 365)
            if filter_status == "edited" and not is_edited:
                continue
            if filter_status == "unedited" and is_edited:
                continue

            # Find matching image
            im_match = None
            for ext in IMAGE_EXTS:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    im_match = cand.name
                    break

            if im_match:
                edit_time = history.get(stem, {}).get("timestamp", "")
                if not edit_time and lbl_p.stat().st_mtime > 86400 * 365:
                    edit_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(lbl_p.stat().st_mtime))
                results.append({
                    "stem": stem,
                    "split": split,
                    "filename": im_match,
                    "domain": dom,
                    "classes": sorted(list(cids)),
                    "poly_count": poly_count,
                    "is_edited": is_edited,
                    "edit_time": edit_time,
                })

    return jsonify({
        "total": len(results),
        "items": results
    })


@app.route("/api/image/<path:subpath>")
def get_image(subpath):
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 404
    # Support arbitrary subpath (e.g. train/img.jpg or root/img.png)
    target = state.dataset_path / "images" / subpath
    if not target.exists():
        target = state.dataset_path / subpath
    if not target.exists():
        return jsonify({"error": "Image file not found"}), 404
    return send_file(target)


@app.route("/api/annotation/<path:stem_or_subpath>")
def get_annotation(stem_or_subpath):
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다.", "polygons": []}), 404
    parts = stem_or_subpath.split("/")
    if len(parts) == 2:
        split, stem = parts[0], parts[1]
    else:
        split, stem = ".", parts[0]

    lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
    img_dir = state.dataset_path / ("images/" + split if split != "." else "images")
    if not lbl_dir.exists():
        lbl_dir = state.dataset_path
    if not img_dir.exists():
        img_dir = state.dataset_path

    lbl_p = lbl_dir / f"{stem}.txt"

    # Match image
    img_p = None
    for ext in IMAGE_EXTS:
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            img_p = cand
            break

    if not img_p or not img_p.exists():
        return jsonify({"error": "Image not found"}), 404

    with Image.open(img_p) as im:
        w, h = im.size

    polygons = []
    if lbl_p.exists():
        with open(lbl_p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                cid = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                pts = [[coords[i], coords[i+1]] for i in range(0, len(coords)-1, 2)]
                cinfo = state.classes.get(cid, {"name": str(cid), "label_ko": str(cid), "color": "#FFFFFF"})
                polygons.append({
                    "id": idx,
                    "class_id": cid,
                    "class_name": cinfo["name"],
                    "label_ko": cinfo["label_ko"],
                    "color": cinfo["color"],
                    "points": pts,
                })

    history = state.load_edits_history()
    return jsonify({
        "stem": stem,
        "split": split,
        "filename": img_p.name,
        "width": w,
        "height": h,
        "polygons": polygons,
        "is_edited": stem in history,
        "history": history.get(stem, None)
    })


@app.route("/api/save/<path:stem_or_subpath>", methods=["POST"])
def save_annotation(stem_or_subpath):
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    parts = stem_or_subpath.split("/")
    if len(parts) == 2:
        split, stem = parts[0], parts[1]
    else:
        split, stem = ".", parts[0]

    data = request.get_json()
    if not data or "polygons" not in data:
        return jsonify({"error": "Invalid payload"}), 400

    lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
    if not lbl_dir.exists():
        lbl_dir = state.dataset_path
    lbl_p = lbl_dir / f"{stem}.txt"

    ts = int(time.time())
    if lbl_p.exists():
        backup_sub = split if split != "." else "root"
        backup_p = state.backup_dir / backup_sub / f"{stem}_{ts}.txt"
        shutil.copy2(lbl_p, backup_p)

    new_lines = []
    for poly in data["polygons"]:
        cid = int(poly["class_id"])
        pts = poly["points"]
        if len(pts) < 3:
            continue
        flat = []
        for pt in pts:
            x = max(0.0, min(1.0, float(pt[0])))
            y = max(0.0, min(1.0, float(pt[1])))
            flat.append(f"{x:.6f}")
            flat.append(f"{y:.6f}")
        line_str = f"{cid} " + " ".join(flat)
        new_lines.append(line_str)

    with open(lbl_p, "w", encoding="utf-8") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")
        else:
            f.write("")

    history = state.load_edits_history()
    history[stem] = {
        "split": split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "polygons_count": len(new_lines),
        "classes": sorted(list(set(int(p["class_id"]) for p in data["polygons"] if len(p["points"]) >= 3))),
        "note": data.get("note", "User manual edit")
    }
    state.save_edits_history(history)

    return jsonify({
        "success": True,
        "polygons_count": len(new_lines),
        "timestamp": history[stem]["timestamp"]
    })


@app.route("/api/delete_class/<path:stem_or_subpath>", methods=["POST"])
def delete_class(stem_or_subpath):
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    parts = stem_or_subpath.split("/")
    if len(parts) == 2:
        split, stem = parts[0], parts[1]
    else:
        split, stem = ".", parts[0]

    data = request.get_json()
    if not data or "class_id" not in data:
        return jsonify({"error": "Missing class_id"}), 400

    target_cid = int(data["class_id"])
    lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
    if not lbl_dir.exists():
        lbl_dir = state.dataset_path
    lbl_p = lbl_dir / f"{stem}.txt"

    if not lbl_p.exists():
        return jsonify({"error": "Label file not found"}), 404

    ts = int(time.time())
    backup_sub = split if split != "." else "root"
    backup_p = state.backup_dir / backup_sub / f"{stem}_{ts}.txt"
    shutil.copy2(lbl_p, backup_p)

    with open(lbl_p, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    new_lines = []
    removed_count = 0
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and int(parts[0]) == target_cid:
            removed_count += 1
            continue
        new_lines.append(line)

    with open(lbl_p, "w", encoding="utf-8") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")
        else:
            f.write("")

    history = state.load_edits_history()
    history[stem] = {
        "split": split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "polygons_count": len(new_lines),
        "classes": sorted(list(set(int(l.split()[0]) for l in new_lines if len(l.split()) >= 7))),
        "note": f"Deleted class {state.classes.get(target_cid, {}).get('name', target_cid)} ({removed_count} polys)"
    }
    state.save_edits_history(history)

    return jsonify({
        "success": True,
        "removed_count": removed_count,
        "remaining_count": len(new_lines)
    })


@app.route("/api/mark_reviewed/<path:stem_or_subpath>", methods=["POST"])
def mark_reviewed(stem_or_subpath):
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    parts = stem_or_subpath.split("/")
    if len(parts) == 2:
        split, stem = parts[0], parts[1]
    else:
        split, stem = ".", parts[0]

    lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
    if not lbl_dir.exists():
        lbl_dir = state.dataset_path
    lbl_file = lbl_dir / f"{stem}.txt"

    poly_count = 0
    cids = set()
    if lbl_file.exists():
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    poly_count += 1
                    cids.add(int(parts[0]))

    history = state.load_edits_history()
    history[stem] = {
        "split": split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "polygons_count": poly_count,
        "classes": sorted(list(cids)),
        "note": "Inspected and approved without changes"
    }
    state.save_edits_history(history)
    return jsonify({
        "success": True,
        "stem": stem,
        "polygons_count": poly_count,
        "timestamp": history[stem]["timestamp"]
    })


@app.route("/api/delete_image/<path:stem_or_subpath>", methods=["POST"])
def delete_image(stem_or_subpath):
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    parts = stem_or_subpath.split("/")
    if len(parts) == 2:
        split, stem = parts[0], parts[1]
    else:
        split, stem = ".", parts[0]

    img_dir = state.dataset_path / ("images/" + split if split != "." else "images")
    lbl_dir = state.dataset_path / ("labels/" + split if split != "." else "labels")
    if not img_dir.exists():
        img_dir = state.dataset_path
    if not lbl_dir.exists():
        lbl_dir = state.dataset_path

    img_candidates = list(img_dir.glob(f"{stem}.*"))
    lbl_candidates = list(lbl_dir.glob(f"{stem}.*"))

    if not img_candidates:
        return jsonify({"error": f"Image {stem} not found"}), 404

    img_p = img_candidates[0]
    sub = split if split != "." else "root"
    dest_img = state.excluded_dir / "images" / sub / img_p.name
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(img_p), str(dest_img))

    if lbl_candidates:
        lbl_p = lbl_candidates[0]
        dest_lbl = state.excluded_dir / "labels" / sub / lbl_p.name
        dest_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(lbl_p), str(dest_lbl))

    history = state.load_edits_history()
    history[stem] = {
        "split": split,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "polygons_count": 0,
        "classes": [],
        "note": "User excluded/deleted image from dataset",
        "status": "excluded"
    }
    state.save_edits_history(history)

    return jsonify({
        "success": True,
        "message": f"Image {stem} safely moved to excluded_images",
        "stem": stem
    })


@app.route("/api/stats")
def get_stats():
    history = state.load_edits_history()
    active_history = {k: v for k, v in history.items() if v.get("status") != "excluded"}
    return jsonify({
        "edited_images_count": len(active_history),
        "recent_edits": sorted(list(active_history.items()), key=lambda x: x[1].get("timestamp", ""), reverse=True)[:10]
    })


@app.route("/api/rebuild", methods=["POST"])
def rebuild_dataset():
    if not state.dataset_path or not state.dataset_path.exists():
        return jsonify({"error": "선택된 데이터셋 디렉토리가 없습니다."}), 400
    ds_name = state.dataset_path.name
    archive_out = PACKAGES_DIR / f"{ds_name}.tar.gz"
    parent_dir = state.dataset_path.parent

    cmd = f"tar -czf '{archive_out}' -C '{parent_dir}' '{ds_name}'"
    ret = os.system(cmd)
    if ret == 0:
        return jsonify({
            "success": True,
            "message": f"Dataset successfully repackaged into {archive_out.name}",
            "archive_name": archive_out.name,
            "size_mb": round(archive_out.stat().st_size / (1024 * 1024), 1)
        })
    else:
        return jsonify({"success": False, "error": f"Tar exit code: {ret}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"🚀 [Universal Dataset Studio & Polygon Lab] Running on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
