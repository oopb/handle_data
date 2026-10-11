"""HD-EPIC: object pickup / putdown and optionally gaze-priming frames.

Input: eye_gaze_priming/priming_info.json (official HD-EPIC annotations).
Edit SETTINGS and run: python hd_epic.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from event_common import event, integer, number, write_records

# ======================== SETTINGS =========================
DATASET_ROOT = Path(__file__).resolve().parent.parent / "datadisk/datasets/HD-EPIC"
ANNOTATION_ROOT = DATASET_ROOT
OUTPUT_DIR = Path(__file__).resolve().parent / "output/hd_epic"
VIDEO_ROOT = DATASET_ROOT / "Videos"
VIDEO_EXT = ".mp4"
# Never assume the dataset's video FPS; use actual aligned MP4 information.
DEFAULT_FPS = None
FPS_BY_VIDEO = {
    # "P01-20240202-110250": 30.0,
}
INCLUDE_PICKUP = True
INCLUDE_PUTDOWN = True
INCLUDE_GAZE_PRIMING = True
INCLUDE_PROSPECTIVE = True
HISTORY_SECONDS = 3.0
# ==========================================================


def _video_path(video_id):
    participant_path = VIDEO_ROOT / video_id.split("-")[0] / (video_id + VIDEO_EXT)
    flat_path = VIDEO_ROOT / (video_id + VIDEO_EXT)
    return flat_path if flat_path.is_file() else participant_path


def _video_times(video_path):
    path = video_path.with_name(video_path.stem + "_mp4_to_vrs_time_ns.csv")
    if not path.is_file():
        return None, None
    times = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "mp4_time_ns" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing mp4_time_ns column")
        for row in reader:
            ns = number(row.get("mp4_time_ns"))
            if ns is None or ns < 0 or (times and ns / 1e9 <= times[-1]):
                raise ValueError(f"{path}: invalid MP4 timestamp mapping")
            times.append(ns / 1e9)
    return times, str(path)


def convert(root=ANNOTATION_ROOT):
    root = Path(root)
    paths = sorted(root.rglob("priming_info.json"))
    if not paths:
        raise FileNotFoundError(
            f"Cannot find eye_gaze_priming/priming_info.json under {root}. "
            "Download official HD-EPIC annotations."
        )
    if len(paths) > 1:
        raise ValueError(f"Multiple priming_info.json files; keep only one: {paths}")
    path = paths[0]
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("priming_info.json should be video_id -> object_id -> data")
    for video_id, objects in doc.items():
        if not isinstance(objects, dict):
            continue
        fps = FPS_BY_VIDEO.get(video_id, DEFAULT_FPS)
        fps_source = (f"hd_epic.FPS_BY_VIDEO[{video_id!r}]" if video_id in FPS_BY_VIDEO
                      else "hd_epic.DEFAULT_FPS")
        video_path = _video_path(video_id)
        times, time_path = _video_times(video_path)
        for object_id, details in objects.items():
            if not isinstance(details, dict):
                continue
            for direction, kind, enabled in (
                ("start", "object_pickup", INCLUDE_PICKUP),
                ("end", "object_putdown", INCLUDE_PUTDOWN),
            ):
                if not enabled and not INCLUDE_GAZE_PRIMING:
                    continue
                annotation = details.get(direction)
                if not isinstance(annotation, dict):
                    continue
                frame = integer(annotation.get("frame"))
                if frame is None or frame < 0:
                    continue
                source_id = f"{object_id}:{direction}"
                if times is not None and frame >= len(times):
                    yield {"_reject": {"annotation": source_id, "video_id": video_id,
                                       "error": "interaction frame is outside MP4 time mapping"}}
                    continue
                if enabled:
                    try:
                        yield event(
                            dataset="hd_epic", video_id=video_id,
                            annotation_id=source_id,
                            event_type=kind, frame=frame, fps=fps, fps_source=fps_source,
                            timestamp=times[frame] if times is not None else None,
                            split="unspecified", label_origin="source_exact",
                            noun=None, attributes={
                                "object_id": str(object_id), "movement_role": direction,
                            },
                            evidence={"original_interaction": annotation, "timestamp_mapping_path": time_path},
                            video_path=video_path,
                            time_basis="hd_epic_mp4_frame_zero_based",
                            annotation_path=path,
                        )
                    except ValueError as exc:
                        yield {"_reject": {"annotation": source_id, "error": str(exc)}}
                        continue
                # Frame of gaze anticipation, already provided in the official
                # annotations. It is NOT the physical contact / pickup frame.
                if not INCLUDE_GAZE_PRIMING:
                    continue
                prime = annotation.get("prime_stats")
                if not isinstance(prime, dict):
                    continue
                primed_frame = integer(prime.get("frame_primed"))
                if primed_frame is None or primed_frame < 0:
                    # -1 means no priming, -2 means excluded; never train
                    # them as positive gaze events.
                    continue
                window_start = integer(prime.get("prime_window_start"))
                if primed_frame > frame or (window_start is not None and
                                             primed_frame < window_start):
                    yield {"_reject": {
                        "annotation": source_id + ":gaze_priming", "video_id": video_id,
                        "error": "gaze priming frame lies outside its source window",
                        "official_prime_stats": prime,
                    }}
                    continue
                try:
                    yield event(
                        dataset="hd_epic", video_id=video_id,
                        annotation_id=source_id + ":gaze_priming",
                        event_type=(
                            "gaze_priming_before_pickup" if direction == "start"
                            else "gaze_priming_before_putdown"
                        ),
                        frame=primed_frame, fps=fps, fps_source=fps_source,
                        timestamp=times[primed_frame] if times is not None else None,
                        split="unspecified", label_origin="source_derived_gaze",
                        attributes={
                            "object_id": str(object_id),
                            "relative_to_interaction": direction,
                            "interaction_frame": frame,
                            "prime_gap_sec": number(prime.get("prime_gap")),
                        },
                        evidence={"official_prime_stats": prime, "timestamp_mapping_path": time_path},
                        video_path=video_path,
                        time_basis="hd_epic_mp4_frame_zero_based",
                        annotation_path=path,
                    )
                except ValueError as exc:
                    yield {"_reject": {"annotation": source_id, "error": str(exc)}}


def main():
    print(write_records(
        convert(), OUTPUT_DIR, include_prospective=INCLUDE_PROSPECTIVE,
        history_seconds=HISTORY_SECONDS,
    ))


if __name__ == "__main__":
    main()
