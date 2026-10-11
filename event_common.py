"""Shared serialization only. Dataset parsing and user settings belong to each module."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = "2.2"


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
    time_basis=None, clip_uid=None, fps_source=None,
    media_source_start_frame=0, media_source_start_sec=0.0, media_duration_sec=None,
):
    """Frame is a canonical-video frame, never a clip-local frame.

    No invented timestamps: frame/fps is only emitted if an explicit FPS is
    supplied and is marked as nominal. Source-provided seconds take priority.
    """
    if frame is not None and integer(frame) is None:
        raise ValueError("event frame must be an integer")
    for name, value in (("timestamp", timestamp), ("FPS", fps),
                        ("context_start", context_start), ("context_end", context_end)):
        if value is not None and number(value) is None:
            raise ValueError(f"{name} must be finite numeric data")
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
    media_start_frame = integer(media_source_start_frame)
    media_start_sec, media_duration_sec = number(media_source_start_sec), number(media_duration_sec)
    if media_start_frame is None or media_start_frame < 0 or media_start_sec is None or media_start_sec < 0:
        raise ValueError("media source offsets must be nonnegative")
    if media_duration_sec is not None and media_duration_sec <= 0:
        raise ValueError("media duration must be positive")
    note = "source_timestamp" if timestamp is not None else None
    if timestamp is None and frame is not None and fps is not None:
        timestamp = frame / fps
        note = "nominal_fps_approximation_verify_pts"
    if timestamp is not None and context_start is not None:
        tolerance = 1 / fps if note == "nominal_fps_approximation_verify_pts" else 0
        if not context_start - tolerance <= timestamp <= context_end + tolerance:
            raise ValueError("event timestamp lies outside its source context")
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
            "fps_nominal": fps, "fps_source": str(fps_source) if fps_source else None,
            "time_basis": time_basis,
            "source_start_frame": media_start_frame, "source_start_sec": media_start_sec,
            "duration_sec": media_duration_sec,
            "available": Path(video_path).is_file() if video_path else None,
        },
        "semantics": {
            "coarse_action": action, "verb": verb, "noun": noun,
            "source_attributes": clean(attributes or {}),
        },
        "target_event": {
            "event_type": event_type, "frame_idx": frame,
            "timestamp_sec": timestamp, "time_note": note,
            "precision": ("sensor_frame" if label_origin == "sensor_derived" else
                          "source_frame") if frame is not None else "source_timestamp",
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
    history_seconds = number(history_seconds)
    if history_seconds is None or history_seconds <= 0:
        raise ValueError("history_seconds must be finite and positive")
    t, s = item["target_event"], item["source"]
    media = item["media"]
    if media.get("available") is False:
        return
    # Canonical labels remain in source coordinates; training consumes the
    # available MP4 and must use that file's frame and second coordinates.
    frame = t["frame_idx"]
    if frame is not None:
        frame -= media.get("source_start_frame", 0)
    ident = item["event_id"]
    timestamp = t["timestamp_sec"]
    if timestamp is not None:
        if t["time_note"] == "nominal_fps_approximation_verify_pts" and frame is not None:
            timestamp = frame / media["fps_nominal"]
        else:
            timestamp -= media.get("source_start_sec", 0.0)
    if (frame is not None and frame < 0) or (timestamp is not None and timestamp < 0):
        return
    duration = media.get("duration_sec")
    if duration is not None and timestamp is not None and timestamp >= duration:
        return
    start = t["context_start_sec"]
    end = t["context_end_sec"]
    if start is not None:
        start = max(0.0, start - media.get("source_start_sec", 0.0))
        end -= media.get("source_start_sec", 0.0)
        if duration is not None:
            end = min(end, duration)
    semantics = item["semantics"]
    attributes = semantics["source_attributes"]
    subject = semantics["coarse_action"] or semantics["noun"]
    if not subject and attributes.get("hand") in {"left", "right"}:
        subject = "左手" if attributes["hand"] == "left" else "右手"
    # An opaque object ID cannot identify an object in RGB video. Keep such
    # events in the canonical layer until a grounded query is available.
    if not subject:
        return
    description = {
        "first_contact": "首次接触",
        "state_change_pnr": "物体状态改变的 PNR",
        "object_pickup": "物体拾取",
        "object_putdown": "物体放下",
        "gaze_priming_before_pickup": "拾取前的注视预期",
        "gaze_priming_before_putdown": "放下前的注视预期",
        "force_contact_onset": "传感器接触上升沿",
    }.get(t["event_type"], t["event_type"])
    question = f"在给定视频范围内，定位「{subject}」的{description}事件。"
    yield {
        "sample_id": ident + ":posthoc", "event_id": ident,
        "mode": "post_hoc", "split": s["split"],
        "input": {
            "video_path": item["media"]["video_path"],
            "video_id": Path(media["video_path"]).stem if media["video_path"] else s["source_video_id"],
            "source_video_id": s["source_video_id"],
            "context_start_sec": start, "context_end_sec": end,
            "time_basis": "video_file",
        },
        "question": question,
        "answer": {
            "event_type": t["event_type"], "frame_idx": frame,
            "timestamp_sec": timestamp,
            "time_note": t["time_note"], "precision": t["precision"],
        },
        "label_origin": item["quality"]["label_origin"],
    }
    if not include_prospective or timestamp is None or timestamp <= 0:
        return
    # Prospective contains the event time and an explicit silence-before
    # contract; it is NOT claiming pre-event negative frames were inspected.
    query_at = max(0.0, start or 0.0, timestamp - history_seconds)
    if query_at >= timestamp:
        return
    yield {
        "sample_id": ident + ":prospective", "event_id": ident,
        "mode": "prospective", "split": s["split"],
        "input": {
            "video_path": item["media"]["video_path"],
            "video_id": Path(media["video_path"]).stem if media["video_path"] else s["source_video_id"],
            "source_video_id": s["source_video_id"],
            "query_time_sec": query_at,
            "playback_start_sec": query_at,
            "time_basis": "video_file",
        },
        "question": f"从查询时刻开始观察，当「{subject}」的{description}事件发生时触发。",
        "answer": {
            "event_type": t["event_type"],
            "frame_idx": frame, "timestamp_sec": timestamp,
            "time_note": t["time_note"], "precision": t["precision"],
        },
        "response_policy": {
            "must_not_respond_before_target": True,
            "trigger_at_target": True,
            "target_time_sec": timestamp,
            "only_observe_frames_at_or_before_current_time": True,
        },
        "label_origin": item["quality"]["label_origin"],
    }


def _query_key(instance):
    payload = [instance["mode"], instance["input"], instance["question"]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def write_records(rows, output_dir, *, include_prospective=True, history_seconds=3.0):
    history_seconds = number(history_seconds)
    if history_seconds is None or history_seconds <= 0:
        raise ValueError("history_seconds must be finite and positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, seen, video_splits = Counter(events=0, instances=0), {}, {}
    query_answers, ambiguous_queries = {}, set()
    # Finish conversion in temporary files before replacing any previous output.
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staging = Path(staging)
        with (staging / "events.jsonl").open("w", encoding="utf-8") as dst, \
             (staging / "train_instances.jsonl").open("w", encoding="utf-8") as samples, \
             (staging / "rejected_samples.jsonl").open("w", encoding="utf-8") as rejects:
            for record in rows:
                if "_reject" in record:
                    counts["rejected"] += 1
                    rejects.write(json.dumps(clean(record["_reject"]), ensure_ascii=False) + "\n")
                    continue
                eid = record["event_id"]
                signature = json.dumps([record["source"]["split"], record["target_event"]],
                                       sort_keys=True, allow_nan=False)
                if eid in seen:
                    if seen[eid] != signature:
                        raise ValueError(f"Conflicting label or split for event_id {eid}")
                    counts["duplicate"] += 1
                    continue
                seen[eid] = signature
                source = record["source"]
                video_key = f"{source['dataset']}:{source['source_video_id']}"
                if source["split"] != "unspecified":
                    video_splits.setdefault(video_key, set()).add(source["split"])
                dst.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                counts["events"] += 1
                counts["type:" + record["target_event"]["event_type"]] += 1
                counts["origin:" + record["quality"]["label_origin"]] += 1
                if record["media"].get("available") is False:
                    counts["events_without_available_media"] += 1
                generated = 0
                for inst in training_instances(
                    record, include_prospective=include_prospective,
                    history_seconds=history_seconds,
                ):
                    samples.write(json.dumps(inst, ensure_ascii=False, allow_nan=False) + "\n")
                    generated += 1
                    key = _query_key(inst)
                    answer = json.dumps([inst["split"], inst["answer"]],
                                        sort_keys=True, allow_nan=False)
                    if key in query_answers and query_answers[key] != answer:
                        ambiguous_queries.add(key)
                    query_answers[key] = answer
                if not generated and record["media"].get("available") is not False:
                    counts["events_without_grounded_query"] += 1
        # Filter on disk so repeated events cannot create identical visual
        # queries with contradictory answers. Canonical events stay intact.
        emitted_queries = set()
        with (staging / "train_instances.jsonl").open(encoding="utf-8") as raw, \
             (staging / "filtered_instances.jsonl").open("w", encoding="utf-8") as filtered:
            for line in raw:
                inst = json.loads(line)
                key = _query_key(inst)
                if key in ambiguous_queries:
                    counts["instances_with_ambiguous_query"] += 1
                    continue
                if key in emitted_queries:
                    counts["duplicate_training_instances"] += 1
                    continue
                emitted_queries.add(key)
                filtered.write(line)
                counts["instances"] += 1
                counts["mode:" + inst["mode"]] += 1
        os.replace(staging / "filtered_instances.jsonl", staging / "train_instances.jsonl")
        stats = dict(counts)
        stats["cross_split_videos"] = {
            key: sorted(splits) for key, splits in sorted(video_splits.items())
            if len(splits) > 1
        }
        (staging / "stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        for name in ("events.jsonl", "train_instances.jsonl", "rejected_samples.jsonl", "stats.json"):
            os.replace(staging / name, output_dir / name)
    return stats
