"""Ego4D FHO: official CONTACT / PNR frames, no video-based relabeling.

Edit SETTINGS directly and run: python ego4d.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from event_common import event, integer, number, write_records

# ======================== SETTINGS =========================
ANNOTATION_ROOT = Path("/data/Ego4D/annotations")
OUTPUT_DIR = Path("./output/ego4d")
VIDEO_ROOT = Path("/data/Ego4D/full_scale")
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


def _video_fps(root):
    """Metadata is used only for NOMINAL frame-to-seconds conversion."""
    result = {}
    for name in ("fho_main.json", "ego4d.json"):
        paths = list(root.rglob(name))
        if not paths:
            continue
        for video in _read(paths[0]).get("videos", []):
            uid = video.get("video_uid")
            fps = number((video.get("video_metadata") or {}).get("fps"))
            if uid and fps and fps > 0:
                result[uid] = fps
    return result


def _semantic_index(root):
    """Join hands action frame intervals to FHO narrations without a model."""
    paths = list(root.rglob("fho_main.json"))
    if not paths:
        return {}
    index = {}
    for video in _read(paths[0]).get("videos", []):
        vid = video.get("video_uid")
        for clip in video.get("annotated_intervals", []):
            for act in clip.get("narrated_actions", []):
                if act.get("is_rejected") is True or act.get("is_invalid_annotation") is True \
                        or act.get("is_valid_action") is False:
                    continue
                start, end = integer(act.get("start_frame")), integer(act.get("end_frame"))
                if vid and start is not None and end is not None:
                    index[(vid, start, end)] = {
                        "action": act.get("narration_text"),
                        "verb": act.get("structured_verb") or act.get("freeform_verb"),
                        "state_transition": act.get("state_transition"),
                    }
    return index


def _point(frame_record, fps):
    """Return canonical video frame / optional source timestamp."""
    if not isinstance(frame_record, dict):
        return None, None
    return (
        integer(frame_record.get("frame")),
        number(frame_record.get("sec", frame_record.get("timestamp_sec"))),
    )


def convert(root=ANNOTATION_ROOT):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Download Ego4D FHO annotations first: {root}")
    hands_files = sorted(root.rglob("fho_hands_*.json"))
    fps_by_video = _video_fps(root)
    semantic_index = _semantic_index(root)

    if hands_files:
        for path in hands_files:
            data = _read(path)
            split = data.get("split") or split_from_file(path)
            for clip in data.get("clips", []):
                vid = clip.get("video_uid")
                clip_uid = clip.get("clip_uid")
                if not vid:
                    continue
                fps = fps_by_video.get(vid, DEFAULT_FPS)
                for i, action in enumerate(clip.get("frames", [])):
                    start_frame = integer(action.get("action_start_frame"))
                    end_frame = integer(action.get("action_end_frame"))
                    info = semantic_index.get((vid, start_frame, end_frame), {})
                    action_id = f"{clip_uid}:{start_frame}:{end_frame}:{i}"
                    start_sec = action.get("action_start_sec")
                    end_sec = action.get("action_end_sec")
                    for key, kind, enabled in (
                        ("contact_frame", "first_contact", INCLUDE_CONTACT),
                        ("pnr_frame", "state_change_pnr", INCLUDE_PNR),
                    ):
                        if not enabled:
                            continue
                        raw = action.get(key)
                        frame, timestamp = _point(raw, fps)
                        if frame is None and timestamp is None:
                            continue
                        try:
                            yield event(
                                dataset="ego4d", video_id=vid,
                                annotation_id=f"{action_id}:{key}",
                                event_type=kind, frame=frame,
                                timestamp=timestamp, fps=fps,
                                context_start=start_sec, context_end=end_sec,
                                action=info.get("action"), verb=info.get("verb"),
                                split=split, label_origin="source_exact",
                                clip_uid=clip_uid, time_basis="canonical_video_frame",
                                video_path=VIDEO_ROOT / f"{vid}.mp4",
                                annotation_path=path,
                                evidence={
                                    "source_frame_record": raw,
                                    "action_start_frame": start_frame,
                                    "action_end_frame": end_frame,
                                },
                                attributes={"state_transition": info.get("state_transition")},
                            )
                        except ValueError as exc:
                            yield {"_reject": {"annotation": action_id, "error": str(exc)}}
    else:
        # Some releases contain fho_main.json but not derived hands splits.
        # Never pretend to know the train/val split when it is absent.
        mains = list(root.rglob("fho_main.json"))
        if not mains:
            raise FileNotFoundError("Expected fho_hands_{train,val}.json or fho_main.json")
        path = mains[0]
        for video in _read(path).get("videos", []):
            vid = video.get("video_uid")
            if not vid:
                continue
            fps = fps_by_video.get(vid, DEFAULT_FPS)
            for clip in video.get("annotated_intervals", []):
                for i, action in enumerate(clip.get("narrated_actions", [])):
                    if action.get("is_rejected") or action.get("is_invalid_annotation") \
                            or action.get("is_valid_action") is False:
                        continue
                    critical = action.get("critical_frames") or {}
                    if not isinstance(critical, dict):
                        continue
                    action_id = action.get("uid") or f"{clip.get('clip_uid')}:{i}"
                    for key, kind, enabled in (
                        ("contact_frame", "first_contact", INCLUDE_CONTACT),
                        ("pnr_frame", "state_change_pnr", INCLUDE_PNR),
                    ):
                        if not enabled:
                            continue
                        raw = critical.get(key)
                        frame, timestamp = _point(raw, fps)
                        if frame is None and timestamp is None:
                            continue
                        try:
                            yield event(
                                dataset="ego4d", video_id=vid,
                                annotation_id=f"{action_id}:{key}",
                                event_type=kind, frame=frame,
                                timestamp=timestamp, fps=fps,
                                context_start=action.get("start_sec"),
                                context_end=action.get("end_sec"),
                                action=action.get("narration_text"),
                                verb=action.get("structured_verb") or action.get("freeform_verb"),
                                split="unspecified", label_origin="source_exact",
                                clip_uid=clip.get("clip_uid"),
                                video_path=VIDEO_ROOT / f"{vid}.mp4",
                                time_basis="canonical_video_frame",
                                annotation_path=path, evidence={"source_frame_record": raw},
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
