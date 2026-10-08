"""Convert existing human-video annotations into event truth + training instances.

Edit CONFIG below, then run this file in an IDE or: python convert_datasets.py
No CLI arguments, AI inference, image inspection or manual relabeling.
Python >= 3.10; pandas is optional except for HD-EPIC .pkl.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# ======================= EDIT CONFIG HERE ==========================
DATA_ROOT = Path("/path/to/downloaded_datasets")
OUTPUT_ROOT = Path("./output")
DATASET_DIRS = {
    "meccano": DATA_ROOT / "MECCANO",
    "holoassist": DATA_ROOT / "HoloAssist",
    "hd_epic": DATA_ROOT / "HD-EPIC",
    "feel": DATA_ROOT / "FEEL",
    "epic_kitchens": DATA_ROOT / "EPIC-KITCHENS-100",
    "ego4d": DATA_ROOT / "Ego4D",
    "ego_exo4d": DATA_ROOT / "Ego-Exo4D",
    "assembly101": DATA_ROOT / "Assembly101",
}
ENABLED = list(DATASET_DIRS)  # e.g. ["ego4d", "assembly101"]
OVERWRITE = True
SAVE_POSTHOC = True
SAVE_PROSPECTIVE = True
# MECCANO CSV uses extracted frame names (e.g. 00012.jpg). Set the
# matching extraction FPS explicitly; 12 is commonly used by RULSTM.
MECCANO_ANNOTATION_FPS = 12.0
# HD-EPIC object pickup/putdown annotations provide frame numbers,
# not timestamps. Set FPS only if verified for your aligned MP4.
HD_EPIC_INTERACTION_FPS = None
# FEEL aligned CSVs contain frame_idx and force; choose verified FPS
# and calibrated hysteresis thresholds for each recording before use.
FEEL_FRAME_FPS = None
FEEL_ON_THRESHOLD = None
FEEL_OFF_THRESHOLD = None
FEEL_MIN_STABLE_FRAMES = 2
GENERATE_SENSOR_PROSPECTIVE = False
# ===================================================================

SCHEMA_VERSION = "1.0"
SPLITS = ("train", "validation", "val", "test")


def value(obj: dict, *keys: str, default=None):
    for key in keys:
        x = obj.get(key)
        if x is not None and x != "":
            return x
    return default


def num(x):
    if x is None or x == "":
        return None
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None


def integer(x):
    if x is None or x == "":
        return None
    if isinstance(x, str):
        numeric = num(x)
        if numeric is not None:
            return int(numeric) if numeric.is_integer() else None
        match = re.search(r"(\d+)(?:\.[a-zA-Z]+)?$", x.strip())
        if match:
            return int(match.group(1))
    y = num(x)
    return int(y) if y is not None and y.is_integer() else None


def seconds(x):
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return num(x)
    s = str(x).strip()
    if ":" not in s:
        return num(s)
    try:
        parts = list(map(float, s.split(":")))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except ValueError:
        pass
    return None


def json_safe(x):
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (list, tuple, set)):
        return [json_safe(t) for t in x]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if hasattr(x, "item"):
        return json_safe(x.item())
    if hasattr(x, "tolist"):
        return json_safe(x.tolist())
    return str(x)


def parse_list(x):
    if isinstance(x, (list, tuple)):
        return json_safe(x)
    if isinstance(x, str) and x.strip().startswith(("[", "(")):
        try:
            return json_safe(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            pass
    return [] if x is None or x == "" else [str(x)]


def read_csv(path: Path, *, header=True):
    with path.open(encoding="utf-8-sig", newline="") as f:
        if header:
            yield from csv.DictReader(f)
        else:
            yield from csv.reader(f)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def source_split(path: Path) -> str:
    for token in (path.stem, *reversed(path.parts)):
        lower = token.lower()
        for split in SPLITS:
            if re.search(r"(^|[_\W])" + split + r"($|[_\W])", lower):
                return "val" if split in ("val", "validation") else split
    return "unspecified"


def stable_id(*parts):
    raw = json.dumps(json_safe(parts), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def canonical(
    dataset: str, video_id: str, annotation_id: Any, event_type: str,
    *, start=None, end=None, timestamp=None, frame=None, fps=None,
    split="unspecified", annotation_type="action_segment", action=None,
    verb=None, noun=None, attributes=None, evidence=None, origin="source_interval",
    video_path=None, frame_basis=None, source_path=None, time_note=None,
):
    start, end, timestamp = seconds(start), seconds(end), seconds(timestamp)
    frame = integer(frame)
    fps = num(fps)
    if frame is not None and frame < 0:
        frame = None
    if (start is not None and start < 0) or (end is not None and end < 0):
        raise ValueError("negative interval timestamp")
    if start is not None and end is not None and end < start:
        raise ValueError("interval end before start")
    if timestamp is not None and timestamp < 0:
        raise ValueError("negative event timestamp")
    if timestamp is None and frame is not None and fps and fps > 0:
        timestamp = frame / fps
        time_note = time_note or "estimated_from_annotation_fps; verify video PTS"
    if timestamp is None and frame is None and (start is None or end is None):
        raise ValueError("event lacks a point or a complete interval")
    if (start is None) != (end is None):
        raise ValueError("incomplete interval")
    precision = "frame" if frame is not None else ("timestamp" if timestamp is not None else "interval")
    if event_type == "action_interval" or event_type == "step_interval":
        precision = "interval"
    aid = str(annotation_id)
    eid = dataset + "_" + stable_id(dataset, video_id, aid, event_type, split)
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": eid,
        "source": {
            "dataset": dataset, "split": split,
            "source_video_id": str(video_id), "source_annotation_id": aid,
            "source_annotation_type": annotation_type,
        },
        "media": {
            "video_path": str(video_path) if video_path else None,
            "fps_nominal": fps, "frame_basis": frame_basis,
        },
        "semantics": {
            "coarse_action": action, "verb": verb, "noun": noun,
            "source_attributes": json_safe(attributes or {}),
        },
        "target_event": {
            "event_type": event_type,
            "timestamp_sec": timestamp, "frame_idx": frame,
            "interval_start_sec": start, "interval_end_sec": end,
            "precision": precision, "time_note": time_note,
        },
        "evidence": json_safe(evidence or {}),
        "quality": {
            "label_origin": origin,
            "requires_visual_refinement": precision == "interval",
            "human_verified_for_this_conversion": False,
            "model_annotated_for_this_conversion": False,
        },
        "provenance": {"annotation_path": str(source_path) if source_path else None,
                       "converter_version": SCHEMA_VERSION},
    }


def instances(e: dict):
    t = e["target_event"]
    src = e["source"]
    name = e["semantics"]["coarse_action"] or t["event_type"]
    point = t["timestamp_sec"]
    if SAVE_POSTHOC:
        answer = ({"timestamp_sec": point, "frame_idx": t["frame_idx"]}
                  if t["precision"] != "interval" else
                  {"start_sec": t["interval_start_sec"], "end_sec": t["interval_end_sec"]})
        yield {
            "sample_id": e["event_id"] + ":posthoc", "event_id": e["event_id"],
            "source_dataset": src["dataset"], "split": src["split"], "mode": "post_hoc",
            "input": {"video_path": e["media"]["video_path"],
                      "video_start_sec": t["interval_start_sec"],
                      "video_end_sec": t["interval_end_sec"]},
            "question": ("定位标注事件的发生时刻：" if t["precision"] != "interval"
                         else "定位动作的起止时间：") + str(name),
            "answer": {"event_type": t["event_type"], **answer},
            "label_origin": e["quality"]["label_origin"],
        }
    if (SAVE_PROSPECTIVE and t["precision"] != "interval" and point is not None
            and (e["quality"]["label_origin"] == "source_exact"
                 or (GENERATE_SENSOR_PROSPECTIVE and e["quality"]["label_origin"] == "sensor_derived"))):
        yield {
            "sample_id": e["event_id"] + ":prospective", "event_id": e["event_id"],
            "source_dataset": src["dataset"], "split": src["split"],
            "mode": "prospective",
            "input": {"video_path": e["media"]["video_path"],
                      "query_time_sec": t["interval_start_sec"]},
            "question": "当以下已标注事件发生时立即触发：" + str(name)
                        + " / " + t["event_type"],
            "answer": {"event_type": t["event_type"], "timestamp_sec": point,
                       "frame_idx": t["frame_idx"]},
            "response_policy": {"must_not_respond_before_target": True,
                                "trigger_at_target": True,
                                "target_time_sec": point},
            "label_origin": e["quality"]["label_origin"],
        }


# ----------------------- Dataset adapters --------------------------


def meccano(root: Path) -> Iterator[dict]:
    """Official RULSTM: 5 headerless CSV cols video, action, name_action, start, end."""
    files = [p for p in root.rglob("*.csv")
             if p.stem.lower() in ("train", "validation", "val", "test")]
    for path in sorted(files):
        for index, row in enumerate(read_csv(path, header=False)):
            if len(row) < 5:
                continue
            video, action_id, name_action, start_text, end_text = row[:5]
            a, b = integer(start_text), integer(end_text)
            if a is None or b is None:
                continue
            fps = MECCANO_ANNOTATION_FPS
            if fps is None or fps <= 0:
                raise ValueError("Configure MECCANO_ANNOTATION_FPS for the extracted frames")
            yield canonical("meccano", video, f"{path.name}:{index}", "action_interval",
                            start=a/fps, end=b/fps, fps=fps, split=source_split(path),
                            action=name_action, attributes={"action_class_id": action_id,
                            "start_source_frame": a, "end_source_frame": b},
                            frame_basis="MECCANO extracted frame names",
                            source_path=path, video_path=video)


def holoassist(root: Path) -> Iterator[dict]:
    """Raw HoloAssist time-range annotations, not aligned model predictions."""
    for path in sorted(root.rglob("*.json")):
        data = load_json(path)
        if isinstance(data, list):
            labels, vid = data, path.parent.name
        elif isinstance(data, dict):
            labels = value(data, "events", "annotations")
            vid = str(value(data, "video_id", "video_name", "name", default=path.parent.name))
            if isinstance(labels, dict):
                labels = list(labels.values())
        else:
            continue
        if not isinstance(labels, list):
            continue
        for index, a in enumerate(labels):
            if not isinstance(a, dict) or a.get("label") not in (
                    "Fine grained action", "Coarse grained action"):
                continue
            attrs = a.get("attributes") or {}
            if not isinstance(attrs, dict):
                attrs = {}
            verb, noun = attrs.get("Verb"), attrs.get("Noun")
            description = attrs.get("Action sentence") or " ".join(
                str(x) for x in (verb, noun) if x and str(x).lower() != "none")
            start, end = seconds(a.get("start")), seconds(a.get("end"))
            if start is None or end is None:
                continue
            yield canonical("holoassist", vid, a.get("id", index),
                            "action_interval" if a["label"] == "Fine grained action"
                            else "step_interval", start=start, end=end,
                            annotation_type=a["label"], split=source_split(path),
                            action=description or None, verb=verb, noun=noun,
                            attributes=attrs, source_path=path)


def hd_epic(root: Path) -> Iterator[dict]:
    """Narrations.pkl is a trusted pandas DataFrame; never unpickle unknown files."""
    files = sorted(root.rglob("HD_EPIC_Narrations.pkl"))
    for path in files:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("HD-EPIC .pkl requires pip install pandas") from exc
        df = pd.read_pickle(path)  # Official trusted source only: pickle can execute code.
        required = {"video_id", "start_timestamp", "end_timestamp"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path}: missing {required - set(df.columns)}")
        for idx, row in enumerate(df.to_dict(orient="records")):
            vid = str(row["video_id"])
            verbs = parse_list(row.get("verbs"))
            nouns = parse_list(row.get("nouns"))
            yield canonical("hd_epic", vid,
                            value(row, "unique_narration_id", default=idx),
                            "action_interval",
                            start=row["start_timestamp"], end=row["end_timestamp"],
                            split=source_split(path), action=row.get("narration"),
                            verb=verbs[0] if verbs else None,
                            noun=nouns[0] if nouns else None,
                            attributes={k: json_safe(row.get(k)) for k in (
                                "verbs", "nouns", "pairs", "main_actions",
                                "hands", "narration_timestamp")},
                            source_path=path, video_path=vid)
    # Optional official pickup/putdown object records. Do not conflate pickup
    # with first contact, and do not infer FPS from video filenames.
    for path in sorted(root.rglob("priming_info.json")):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for vid, objects in data.items():
            if not isinstance(objects, dict):
                continue
            for object_id, obj in objects.items():
                if not isinstance(obj, dict):
                    continue
                for key, event_type in (("start", "pickup_frame"),
                                        ("end", "putdown_frame")):
                    record = obj.get(key)
                    if not isinstance(record, dict) or integer(record.get("frame")) is None:
                        continue
                    yield canonical(
                        "hd_epic", vid, f"{object_id}:{key}", event_type,
                        frame=record["frame"], fps=HD_EPIC_INTERACTION_FPS,
                        origin="source_exact", annotation_type="object_interaction",
                        noun=str(object_id), evidence=record,
                        frame_basis="official interaction frames", source_path=path,
                        video_path=vid)


def epic_kitchens(root: Path) -> Iterator[dict]:
    """EPIC-KITCHENS-100 train/validation action CSV, excludes label-free test."""
    for path in sorted(root.rglob("EPIC_100_*.csv")):
        if path.name not in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
            continue
        for index, row in enumerate(read_csv(path)):
            vid = row.get("video_id")
            a, b = seconds(row.get("start_timestamp")), seconds(row.get("stop_timestamp"))
            if not vid or a is None or b is None:
                continue
            yield canonical(
                "epic_kitchens", vid, row.get("narration_id") or index,
                "action_interval", start=a, end=b, split=source_split(path),
                action=row.get("narration"), verb=row.get("verb"), noun=row.get("noun"),
                attributes={k: row.get(k) for k in ("participant_id", "verb_class",
                            "noun_class", "all_nouns", "all_noun_classes",
                            "start_frame", "stop_frame", "narration_timestamp")},
                frame_basis="official extracted action frames (not event frames)",
                source_path=path, video_path=vid)


def _ego_critical(raw, fps):
    if isinstance(raw, dict):
        idx = integer(value(raw, "frame", "frame_number"))
        ts = seconds(value(raw, "sec", "timestamp_sec", "time_sec"))
        # In some annotation exports, critical_frames values are a list/number.
        return idx, ts
    if isinstance(raw, (int, float)):
        return integer(raw), None
    return None, None


def ego4d(root: Path) -> Iterator[dict]:
    """FHO main annotations preferred; fallback FHO hands when main absent."""
    mains = sorted(root.rglob("fho_main.json"))
    if mains:
        for path in mains:
            doc = load_json(path)
            for vid in doc.get("videos", []):
                video_id = vid.get("video_uid")
                if not video_id:
                    continue
                fps = num((vid.get("video_metadata") or {}).get("fps"))
                for clip in vid.get("annotated_intervals", []):
                    for i, action in enumerate(clip.get("narrated_actions", [])):
                        if action.get("is_valid_action") is False or action.get("is_rejected") is True:
                            continue
                        aid = action.get("uid") or f"{clip.get('clip_id')}:{i}"
                        start, end = action.get("start_sec"), action.get("end_sec")
                        details = {"state_transition": action.get("state_transition"),
                                   "clip_uid": clip.get("clip_uid")}
                        yield canonical("ego4d", video_id, aid, "action_interval",
                                        start=start, end=end, fps=fps,
                                        action=action.get("narration_text"),
                                        verb=action.get("structured_verb") or action.get("freeform_verb"),
                                        attributes=details, source_path=path,
                                        video_path=video_id)
                        critical = action.get("critical_frames") or {}
                        if isinstance(critical, dict):
                            for key, kind in (("contact_frame", "first_contact"),
                                              ("pnr_frame", "state_change_pnr")):
                                raw = critical.get(key)
                                frame, time = _ego_critical(raw, fps)
                                if frame is None and time is None:
                                    continue
                                yield canonical(
                                    "ego4d", video_id, aid, kind,
                                    start=start, end=end, timestamp=time,
                                    frame=frame, fps=fps,
                                    annotation_type="fho_critical_frame",
                                    origin="source_exact",
                                    action=action.get("narration_text"),
                                    verb=action.get("structured_verb"),
                                    attributes=details,
                                    evidence={"critical_frame": raw},
                                    frame_basis="Ego4D canonical video frame",
                                    source_path=path, video_path=video_id)
    else:
        for path in sorted(root.rglob("fho_hands_*.json")):
            doc = load_json(path)
            for clip in doc.get("clips", []):
                vid = clip.get("video_uid")
                for i, ann in enumerate(clip.get("frames", [])):
                    for key, kind in (("contact_frame", "first_contact"),
                                      ("pnr_frame", "state_change_pnr")):
                        raw = ann.get(key)
                        if not isinstance(raw, dict) or integer(raw.get("frame")) is None:
                            continue
                        aid = f"{clip.get('clip_uid')}:{i}"
                        yield canonical(
                            "ego4d", vid, aid, kind,
                            start=ann.get("action_start_sec"), end=ann.get("action_end_sec"),
                            frame=raw["frame"], fps=None, split=source_split(path),
                            annotation_type="fho_hands",
                            origin="source_exact", evidence=raw,
                            frame_basis="Ego4D canonical video frame",
                            source_path=path, video_path=vid)
    # Do not process OSCC in parallel with main to avoid redundant PNR labels.


def ego_exo4d(root: Path) -> Iterator[dict]:
    """Official keystep: annotations[take_uid].segments[*]."""
    for path in sorted(root.rglob("*keystep*.json")):
        doc = load_json(path)
        ann = doc.get("annotations") if isinstance(doc, dict) else None
        if isinstance(ann, dict):
            records = ann.values()
        elif isinstance(ann, list):
            records = ann
        else:
            continue
        for take in records:
            if not isinstance(take, dict) or "segments" not in take:
                continue
            vid = take.get("take_uid")
            if not vid:
                continue
            for i, step in enumerate(take.get("segments", [])):
                aid = f"{step.get('step_unique_id', step.get('step_id', 'step'))}:{i}"
                yield canonical(
                    "ego_exo4d", vid, aid, "step_interval",
                    start=step.get("start_time"), end=step.get("end_time"),
                    split=source_split(path), annotation_type="keystep",
                    action=step.get("step_description") or step.get("step_name"),
                    attributes={"scenario": take.get("scenario"),
                                "take_name": take.get("take_name"),
                                "step_name": step.get("step_name"),
                                "step_id": step.get("step_id"),
                                "is_essential": step.get("is_essential")},
                    source_path=path, video_path=vid)


def assembly101(root: Path) -> Iterator[dict]:
    """Fine-grained CSV: frame indices are @30fps, even when raw videos are 60fps."""
    for path in sorted(root.rglob("*.csv")):
        if (path.name not in ("train.csv", "validation.csv", "test.csv")
                or "fine-grained" not in str(path.parent).lower()):
            continue
        for i, row in enumerate(read_csv(path)):
            vid = row.get("video")
            a, b = integer(row.get("start_frame")), integer(row.get("end_frame"))
            if not vid or a is None or b is None:
                continue
            yield canonical(
                "assembly101", vid, f"fine:{row.get('id', i)}", "action_interval",
                start=a/30.0, end=b/30.0, fps=30.0, split=source_split(path),
                annotation_type="fine_grained",
                action=row.get("action_cls"), verb=row.get("verb_cls"),
                noun=row.get("noun_cls"),
                attributes={"start_source_frame": a, "end_source_frame": b,
                            "action_id": row.get("action_id"),
                            "toy_id": row.get("toy_id"), "toy_name": row.get("toy_name"),
                            "is_shared": row.get("is_shared"), "is_RGB": row.get("is_RGB")},
                frame_basis="30fps extracted frames; original MP4 may be 60fps",
                source_path=path, video_path=vid)
    # Sequence-level coarse_labels: 3-column whitespace (start_frame end_frame label).
    # These may represent the same activities as fine-grained labels at a different level.
    coarse_split_by_seq = {}
    for split_file in root.rglob("coarse_splits/*.txt"):
        label = source_split(split_file)
        for line in split_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts and label != "unspecified":
                coarse_split_by_seq[Path(parts[0]).stem] = label
    for path in sorted(root.rglob("coarse_labels/*.txt")):
        split = coarse_split_by_seq.get(path.stem, "unspecified")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            parts = line.strip().split(maxsplit=2)
            if len(parts) != 3:
                continue
            a, b = integer(parts[0]), integer(parts[1])
            if a is None or b is None:
                continue
            yield canonical(
                "assembly101", path.stem, f"coarse:{i}", "step_interval",
                start=a/30.0, end=b/30.0, fps=30.0,
                annotation_type="coarse_grained", action=parts[2], split=split,
                attributes={"start_source_frame": a, "end_source_frame": b},
                frame_basis="30fps extracted frames; sequence-level label",
                source_path=path, video_path=None)


def feel(root: Path) -> Iterator[dict]:
    """FEEL synchronized force CSV: conservative onset detected by hysteresis.

    Requires explicit calibrated thresholds and FPS. No force threshold is
    universally valid; do not turn an uncalibrated number into ground truth.
    Missing (NaN) force and force_row_idx=-1 break a contact run.
    """
    if (FEEL_FRAME_FPS is None or FEEL_FRAME_FPS <= 0
            or FEEL_ON_THRESHOLD is None or FEEL_OFF_THRESHOLD is None):
        print("[feel] Skipped: set FEEL_FRAME_FPS / ON_THRESHOLD / OFF_THRESHOLD "
              "from your calibrated recording to derive sensor events.")
        return
    if not 0 <= FEEL_OFF_THRESHOLD < FEEL_ON_THRESHOLD:
        raise ValueError("FEEL requires 0 <= OFF_THRESHOLD < ON_THRESHOLD")
    for path in sorted(root.rglob("force_*aligned*.csv")):
        hand = "left" if "left" in path.name.lower() else (
            "right" if "right" in path.name.lower() else "unknown")
        video = "/".join(path.relative_to(root).parts[:-1])
        active = False
        armed = False  # an observed non-contact state is required first
        prev_frame = None
        on_run = []
        off_count = 0
        for row in read_csv(path):
            idx = integer(row.get("frame_idx"))
            force = num(row.get("force"))
            valid = (idx is not None and force is not None
                     and integer(row.get("force_row_idx")) != -1)
            if not valid or (prev_frame is not None and idx != prev_frame + 1):
                active, armed, on_run, off_count = False, False, [], 0
                if not valid:
                    prev_frame = idx
                    continue
            prev_frame = idx
            if not active:
                if force <= FEEL_OFF_THRESHOLD:
                    armed = True
                    on_run.clear()
                elif armed and force >= FEEL_ON_THRESHOLD:
                    on_run.append(idx)
                else:
                    on_run.clear()
                if armed and len(on_run) >= FEEL_MIN_STABLE_FRAMES:
                    # The first high-force frame is only a sensor-derived
                    # onset, NOT necessarily first visual contact.
                    f = on_run[0]
                    yield canonical(
                        "feel", video, f"{hand}:{f}", "force_contact_onset",
                        frame=f, fps=FEEL_FRAME_FPS, split="unspecified",
                        annotation_type="aligned_force",
                        origin="sensor_derived",
                        attributes={"hand": hand, "on_threshold": FEEL_ON_THRESHOLD,
                                    "off_threshold": FEEL_OFF_THRESHOLD,
                                    "min_stable_frames": FEEL_MIN_STABLE_FRAMES},
                        evidence={"force_at_confirmation": force},
                        frame_basis="aligned FEEL RGB frame_idx",
                        source_path=path)
                    active, armed, on_run = True, False, []
            elif force <= FEEL_OFF_THRESHOLD:
                off_count += 1
                if off_count >= FEEL_MIN_STABLE_FRAMES:
                    active, armed, off_count = False, True, 0
            else:
                off_count = 0


CONVERTERS = {
    "meccano": meccano,
    "holoassist": holoassist,
    "hd_epic": hd_epic,
    "feel": feel,
    "epic_kitchens": epic_kitchens,
    "ego4d": ego4d,
    "ego_exo4d": ego_exo4d,
    "assembly101": assembly101,
}


def convert_one(name: str, root: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    ev_path, tr_path, bad_path = (
        dest / "events.jsonl", dest / "train_instances.jsonl",
        dest / "rejected_samples.jsonl")
    if not OVERWRITE and (ev_path.exists() or tr_path.exists()):
        raise FileExistsError(f"Output exists: {dest}")
    counts = Counter()
    seen = set()
    with (ev_path.open("w", encoding="utf-8") as ef,
          tr_path.open("w", encoding="utf-8") as tf,
          bad_path.open("w", encoding="utf-8") as bf):
        try:
            iterator = iter(CONVERTERS[name](root))
            while True:
                try:
                    e = next(iterator)
                except StopIteration:
                    break
                except (ValueError, TypeError, KeyError) as err:
                    counts["file_or_row_errors"] += 1
                    bf.write(json.dumps({"error": str(err), "dataset": name},
                                        ensure_ascii=False) + "\n")
                    # A Python generator closes when its body raises; avoid
                    # silently claiming the remaining annotation file passed.
                    print(f"[{name}] CONVERSION STOPPED by malformed row: {err}")
                    break
                if e["event_id"] in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(e["event_id"])
                ef.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n")
                counts["events"] += 1
                counts["event:" + e["target_event"]["event_type"]] += 1
                for s in instances(e):
                    tf.write(json.dumps(s, ensure_ascii=False, allow_nan=False) + "\n")
                    counts["instances"] += 1
                    counts["mode:" + s["mode"]] += 1
        except Exception as err:
            counts["fatal_errors"] += 1
            bf.write(json.dumps({"error": repr(err), "dataset": name},
                                ensure_ascii=False) + "\n")
            raise
    (dest / "stats.json").write_text(
        json.dumps(dict(counts), ensure_ascii=False, indent=2), encoding="utf-8")
    return dict(counts)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ENABLED:
        if name not in CONVERTERS:
            raise ValueError(f"Unknown dataset: {name}")
        root = DATASET_DIRS[name]
        if not root.exists():
            print(f"[{name}] Missing input directory: {root} (skipped)")
            continue
        print(f"[{name}] converting {root}")
        stats = convert_one(name, root, OUTPUT_ROOT / name)
        print(f"[{name}] {stats}")
    print("Finished. Exact-event and interval supervision remain separate.")


if __name__ == "__main__":
    main()
