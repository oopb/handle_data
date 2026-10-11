"""Ego4D FHO: official CONTACT / PNR frames, no video-based relabeling.

Edit SETTINGS directly and run: python ego4d.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from event_common import event, integer, number, write_records

# ======================== SETTINGS =========================
DATASET_ROOT = Path(__file__).resolve().parent.parent / "datadisk/datasets/ego4d_fho"
ANNOTATION_ROOT = DATASET_ROOT
OUTPUT_DIR = Path(__file__).resolve().parent / "output/ego4d"
VIDEO_ROOT = DATASET_ROOT / "v2/full_scale"
CLIP_VIDEO_ROOT = DATASET_ROOT / "v2/clips"
DEFAULT_FPS = None  # Keep None unless you have verified the video frame rate.
INCLUDE_CONTACT = True
INCLUDE_PNR = True
INCLUDE_PROSPECTIVE = True
HISTORY_SECONDS = 3.0
# ==========================================================


def split_from_file(path):
    match = re.search(r"fho_hands_(train|val|test)\.json$", path.name, flags=re.I)
    return match.group(1).lower() if match else "unspecified"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _items(path, key):
    """Stream large official JSON arrays without loading the 2 GB master file."""
    try:
        import ijson
    except ImportError:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ImportError("Large Ego4D JSON requires ijson: python -m pip install ijson")
        yield from _read(path).get(key, [])
        return
    with path.open("rb") as file:
        yield from ijson.items(file, key + ".item", use_float=True)


def _clip_splits(root):
    """Retain splits only for an exact clip UID match to official LTA files."""
    index = {}
    for split in ("train", "val", "test"):
        name = f"fho_lta_{split}{'_unannotated' if split == 'test' else ''}.json"
        path = _unique_file(root, name)
        if path is None:
            continue
        for clip in _items(path, "clips"):
            uid = clip.get("clip_uid")
            if uid:
                previous = index.get(uid)
                index[uid] = (split, str(path)) if previous is None or previous[0] == split \
                    else ("unspecified", None)
    return index


def _clip_media_index():
    path = CLIP_VIDEO_ROOT / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {row["exported_clip_uid"]: row for row in csv.DictReader(file)}


def _media(vid, clip, clip_media):
    full = VIDEO_ROOT / f"{vid}.mp4"
    uid = clip.get("clip_uid")
    path = CLIP_VIDEO_ROOT / f"{uid}.mp4"
    if full.is_file() or not path.is_file():
        return {"video_path": full}
    mapping = clip_media.get(uid, {})
    if mapping and mapping["parent_video_uid"] != vid:
        raise ValueError("clip manifest parent video does not match the annotation")
    start_frame = integer(clip.get("clip_parent_start_frame", mapping.get("parent_frame_start")))
    start_sec = number(clip.get("clip_parent_start_sec", mapping.get("parent_start_sec")))
    end_sec = number(clip.get("clip_parent_end_sec", mapping.get("parent_end_sec")))
    if start_frame is None or start_sec is None:
        raise ValueError("available clip needs source frame and second offsets")
    return {"video_path": path, "media_source_start_frame": start_frame,
            "media_source_start_sec": start_sec,
            "media_duration_sec": end_sec - start_sec if end_sec is not None else None}


def _unique_file(root, name):
    paths = sorted(root.rglob(name))
    if len(paths) > 1:
        raise ValueError(f"Multiple {name} files; select one annotation release: {paths}")
    return paths[0] if paths else None


def _video_fps(root):
    """Metadata is used only for NOMINAL frame-to-seconds conversion."""
    result = {}
    for name in ("fho_main.json", "ego4d.json"):
        path = _unique_file(root, name)
        if path is None:
            continue
        for video in _items(path, "videos"):
            uid = video.get("video_uid")
            fps = number((video.get("video_metadata") or {}).get("fps"))
            if uid and fps and fps > 0:
                result.setdefault(uid, (fps, f"{path}:video_metadata.fps"))
    return result


def _semantic_index(root):
    """Join hands action frame intervals to FHO narrations without a model."""
    path = _unique_file(root, "fho_main.json")
    if path is None:
        return {}
    index = {}
    for video in _items(path, "videos"):
        vid = video.get("video_uid")
        for clip in video.get("annotated_intervals", []):
            if clip.get("redacted"):
                continue
            for act in clip.get("narrated_actions", []):
                if act.get("is_rejected") is True or act.get("is_invalid_annotation") is True \
                        or act.get("is_valid_action") is False:
                    continue
                start, end = integer(act.get("start_frame")), integer(act.get("end_frame"))
                if vid and start is not None and end is not None:
                    key = (vid, start, end)
                    info = {
                        "annotation_id": act.get("uid"),
                        "action": act.get("narration_text"),
                        "verb": act.get("structured_verb") or act.get("freeform_verb"),
                        "state_transition": act.get("state_transition"),
                        "annotation_path": str(path),
                    }
                    # Multiple narrations can share a frame interval. Do not
                    # silently assign the last narration to all hands events.
                    index[key] = info if key not in index or index[key] == info else None
    return index


def _point(frame_record):
    """Return canonical video frame / optional source timestamp."""
    if not isinstance(frame_record, dict):
        # fho_main critical_frames stores canonical integer frame numbers;
        # hands splits store dictionaries containing frame / clip_frame.
        return frame_record, None
    return (
        frame_record.get("frame"),
        frame_record.get("sec") if frame_record.get("sec") is not None
        else frame_record.get("timestamp_sec"),
    )


def convert(root=ANNOTATION_ROOT):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Download Ego4D FHO annotations first: {root}")
    hands_files = sorted(p for p in root.rglob("fho_hands_*.json")
                         if not p.name.endswith("_unannotated.json"))
    if len({p.name for p in hands_files}) != len(hands_files):
        raise ValueError("Multiple hands files for the same split; select one annotation release")
    clip_media = _clip_media_index()

    if hands_files:
        fps_by_video = _video_fps(root)
        semantic_index = _semantic_index(root)
        for path in hands_files:
            data = _read(path)
            split = data.get("split") or split_from_file(path)
            for clip in data.get("clips", []):
                vid = clip.get("video_uid")
                clip_uid = clip.get("clip_uid")
                if not vid:
                    continue
                fps, fps_source = fps_by_video.get(vid, (DEFAULT_FPS, "ego4d.DEFAULT_FPS"))
                for action in clip.get("frames", []):
                    start_frame = integer(action.get("action_start_frame"))
                    end_frame = integer(action.get("action_end_frame"))
                    info = semantic_index.get((vid, start_frame, end_frame)) or {}
                    action_id = info.get("annotation_id") or (
                        f"{clip_uid or clip.get('clip_id')}:{start_frame}:{end_frame}"
                    )
                    start_sec = action.get("action_start_sec")
                    end_sec = action.get("action_end_sec")
                    for key, kind, enabled in (
                        ("contact_frame", "first_contact", INCLUDE_CONTACT),
                        ("pnr_frame", "state_change_pnr", INCLUDE_PNR),
                    ):
                        if not enabled:
                            continue
                        raw = action.get(key)
                        frame, timestamp = _point(raw)
                        if frame is None and timestamp is None:
                            continue
                        try:
                            yield event(
                                dataset="ego4d", video_id=vid,
                                annotation_id=(f"{action_id}:{key}" if start_frame is not None
                                               and end_frame is not None else
                                               f"{action_id}:{key}:{frame}:{timestamp}"),
                                event_type=kind, frame=frame,
                                timestamp=timestamp, fps=fps, fps_source=fps_source,
                                context_start=start_sec, context_end=end_sec,
                                action=info.get("action"), verb=info.get("verb"),
                                split=split, label_origin="source_exact",
                                clip_uid=clip_uid, time_basis="canonical_video_frame",
                                **_media(vid, clip, clip_media),
                                annotation_path=path,
                                evidence={
                                    "source_frame_record": raw,
                                    "action_start_frame": start_frame,
                                    "action_end_frame": end_frame,
                                    "source_action_uid": info.get("annotation_id"),
                                    "semantic_annotation_path": info.get("annotation_path"),
                                },
                                attributes={"state_transition": info.get("state_transition")},
                            )
                        except ValueError as exc:
                            yield {"_reject": {"annotation": action_id, "error": str(exc)}}
    else:
        # Some releases contain fho_main.json but not derived hands splits.
        # Exact official LTA clip matches supply a traceable split; unmatched
        # clips remain unspecified. This is not the hands benchmark split.
        path = _unique_file(root, "fho_main.json")
        if path is None:
            raise FileNotFoundError("Expected fho_hands_{train,val}.json or fho_main.json")
        clip_splits = _clip_splits(root)
        for video in _items(path, "videos"):
            vid = video.get("video_uid")
            if not vid:
                continue
            fps = number((video.get("video_metadata") or {}).get("fps")) or DEFAULT_FPS
            fps_source = f"{path}:video_metadata.fps" if number(
                (video.get("video_metadata") or {}).get("fps")) else "ego4d.DEFAULT_FPS"
            for clip in video.get("annotated_intervals", []):
                if clip.get("redacted"):
                    continue
                split, split_path = clip_splits.get(clip.get("clip_uid"), ("unspecified", None))
                for action in clip.get("narrated_actions", []):
                    if action.get("is_rejected") or action.get("is_invalid_annotation") \
                            or action.get("is_valid_action") is False:
                        continue
                    critical = action.get("critical_frames") or {}
                    if not isinstance(critical, dict):
                        continue
                    action_id = action.get("uid") or (
                        f"{clip.get('clip_uid') or clip.get('clip_id')}:"
                        f"{action.get('start_frame')}:{action.get('end_frame')}:"
                        f"{action.get('start_sec')}:{action.get('end_sec')}"
                    )
                    for key, kind, enabled in (
                        ("contact_frame", "first_contact", INCLUDE_CONTACT),
                        ("pnr_frame", "state_change_pnr", INCLUDE_PNR),
                    ):
                        if not enabled:
                            continue
                        raw = critical.get(key)
                        frame, timestamp = _point(raw)
                        if frame is None and timestamp is None:
                            continue
                        try:
                            yield event(
                                dataset="ego4d", video_id=vid,
                                annotation_id=(f"{action_id}:{key}" if action.get("uid") else
                                               f"{action_id}:{key}:{frame}:{timestamp}"),
                                event_type=kind, frame=frame,
                                timestamp=timestamp, fps=fps, fps_source=fps_source,
                                context_start=action.get("start_sec"),
                                context_end=action.get("end_sec"),
                                action=action.get("narration_text"),
                                verb=action.get("structured_verb") or action.get("freeform_verb"),
                                split=split, label_origin="source_exact",
                                clip_uid=clip.get("clip_uid"),
                                **_media(vid, clip, clip_media),
                                time_basis="canonical_video_frame",
                                annotation_path=path, evidence={
                                    "source_frame_record": raw,
                                    "source_clip_frame": (action.get("clip_critical_frames") or {}).get(key),
                                    "split_annotation_path": split_path,
                                },
                                attributes={"state_transition": action.get("state_transition")},
                            )
                        except ValueError as exc:
                            yield {"_reject": {"annotation": str(action_id), "error": str(exc)}}


def main():
    print(write_records(
        convert(), OUTPUT_DIR, include_prospective=INCLUDE_PROSPECTIVE,
        history_seconds=HISTORY_SECONDS,
    ))


if __name__ == "__main__":
    main()
