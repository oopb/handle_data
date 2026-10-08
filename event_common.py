"""Shared serialization only. Dataset parsing and user settings belong to each module."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "2.0"


def number(x):
    if x is None or isinstance(x, bool) or x == "":
        return None
    try:
        n = float(x)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def integer(x):
    n = number(x)
    return int(n) if n is not None and n.is_integer() else None


def clean(value):
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return clean(value.item())
    return str(value)


def event(
    *, dataset, video_id, annotation_id, event_type,
    frame=None, timestamp=None, fps=None,
    context_start=None, context_end=None, action=None, verb=None, noun=None,
    split="unspecified", label_origin="source_exact",
    annotation_path=None, video_path=None, evidence=None, attributes=None,
    time_basis=None, clip_uid=None,
):
    """Frame is a canonical-video frame, never a clip-local frame.

    No invented timestamps: frame/fps is only emitted if an explicit FPS is
    supplied and is marked as nominal. Source-provided seconds take priority.
    """
    frame = integer(frame)
    timestamp, fps = number(timestamp), number(fps)
    context_start, context_end = number(context_start), number(context_end)
    if (frame is None and timestamp is None) or (frame is not None and frame < 0):
        raise ValueError("source event needs a nonnegative frame or timestamp")
    if timestamp is not None and timestamp < 0:
        raise ValueError("negative event timestamp")
    if (context_start is None) != (context_end is None):
        raise ValueError("context interval must have two endpoints")
    if (context_start is not None and
            (context_start < 0 or context_end < context_start)):
        raise ValueError("invalid context interval")
    if fps is not None and fps <= 0:
        raise ValueError("FPS must be positive")
    note = "source_timestamp" if timestamp is not None else None
    if timestamp is None and frame is not None and fps is not None:
        timestamp = frame / fps
        note = "nominal_fps_approximation_verify_pts"
    ident = hashlib.sha256(
        json.dumps([dataset, video_id, str(annotation_id), event_type],
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": f"{dataset}_{ident}",
        "source": {
            "dataset": dataset, "split": split,
            "source_video_id": str(video_id),
            "source_clip_uid": clip_uid,
            "source_annotation_id": str(annotation_id),
        },
        "media": {
            "video_path": str(video_path) if video_path else None,
            "video_ref": str(video_id),
            "fps_nominal": fps, "time_basis": time_basis,
        },
        "semantics": {
            "coarse_action": action, "verb": verb, "noun": noun,
            "source_attributes": clean(attributes or {}),
        },
        "target_event": {
            "event_type": event_type, "frame_idx": frame,
            "timestamp_sec": timestamp, "time_note": note,
            "precision": "source_frame" if frame is not None else "source_timestamp",
            "context_start_sec": context_start,
            "context_end_sec": context_end,
        },
        "evidence": clean(evidence or {}),
        "quality": {
            "label_origin": label_origin,
            "human_verified_for_this_conversion": False,
            "model_annotated_for_this_conversion": False,
        },
        "provenance": {
            "annotation_path": str(annotation_path) if annotation_path else None,
            "converter_version": SCHEMA_VERSION,
        },
    }


def training_instances(item, *, include_prospective, history_seconds):
    t, s = item["target_event"], item["source"]
    ident = item["event_id"]
    timestamp = t["timestamp_sec"]
    question = f"定位 {t['event_type']} 事件"
    yield {
        "sample_id": ident + ":posthoc", "event_id": ident,
        "mode": "post_hoc", "split": s["split"],
        "input": {
            "video_path": item["media"]["video_path"],
            "video_id": s["source_video_id"],
            "context_start_sec": t["context_start_sec"],
            "context_end_sec": t["context_end_sec"],
        },
        "question": question,
        "answer": {
            "event_type": t["event_type"], "frame_idx": t["frame_idx"],
            "timestamp_sec": timestamp,
        },
        "label_origin": item["quality"]["label_origin"],
    }
    if not include_prospective or timestamp is None:
        return
    # Prospective contains the event time and an explicit silence-before
    # contract; it is NOT claiming pre-event negative frames were inspected.
    query_at = max(0.0, timestamp - history_seconds)
    yield {
        "sample_id": ident + ":prospective", "event_id": ident,
        "mode": "prospective", "split": s["split"],
        "input": {
            "video_path": item["media"]["video_path"],
            "video_id": s["source_video_id"],
            "query_time_sec": query_at,
        },
        "question": f"当 {t['event_type']} 事件发生时触发。",
        "answer": {
            "event_type": t["event_type"],
            "frame_idx": t["frame_idx"], "timestamp_sec": timestamp,
        },
        "response_policy": {
            "must_not_respond_before_target": True,
            "trigger_at_target": True,
            "target_time_sec": timestamp,
        },
        "label_origin": item["quality"]["label_origin"],
    }


def write_records(rows, output_dir, *, include_prospective=True, history_seconds=3.0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, seen = Counter(), set()
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as dst, \
         (output_dir / "train_instances.jsonl").open("w", encoding="utf-8") as samples, \
         (output_dir / "rejected_samples.jsonl").open("w", encoding="utf-8") as rejects:
        for record in rows:
            if "_reject" in record:
                counts["rejected"] += 1
                rejects.write(json.dumps(clean(record["_reject"]), ensure_ascii=False) + "\n")
                continue
            eid = record["event_id"]
            if eid in seen:
                counts["duplicate"] += 1
                continue
            seen.add(eid)
            dst.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            counts["events"] += 1
            counts["type:" + record["target_event"]["event_type"]] += 1
            for inst in training_instances(
                record, include_prospective=include_prospective,
                history_seconds=history_seconds,
            ):
                samples.write(json.dumps(inst, ensure_ascii=False, allow_nan=False) + "\n")
                counts["instances"] += 1
                counts["mode:" + inst["mode"]] += 1
    (output_dir / "stats.json").write_text(
        json.dumps(dict(counts), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dict(counts)
