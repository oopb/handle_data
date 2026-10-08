"""HD-EPIC: object pickup / putdown and optionally gaze-priming frames.

Input: eye_gaze_priming/priming_info.json (official HD-EPIC annotations).
Edit SETTINGS and run: python hd_epic.py
"""
from __future__ import annotations

import json
from pathlib import Path

from event_common import event, integer, number, write_records

# ======================== SETTINGS =========================
ANNOTATION_ROOT = Path("/data/HD-EPIC/annotations")
OUTPUT_DIR = Path("./output/hd_epic")
VIDEO_ROOT = Path("/data/HD-EPIC/videos")
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
        video_path = VIDEO_ROOT / (video_id + VIDEO_EXT)
        for object_id, details in objects.items():
            if not isinstance(details, dict):
                continue
            for direction, kind, enabled in (
                ("start", "object_pickup", INCLUDE_PICKUP),
                ("end", "object_putdown", INCLUDE_PUTDOWN),
            ):
                if not enabled:
                    continue
                annotation = details.get(direction)
                if not isinstance(annotation, dict):
                    continue
                frame = integer(annotation.get("frame"))
                if frame is None or frame < 0:
                    continue
                source_id = f"{object_id}:{direction}"
                try:
                    yield event(
                        dataset="hd_epic", video_id=video_id,
                        annotation_id=source_id,
                        event_type=kind, frame=frame, fps=fps,
                        split="unspecified", label_origin="source_exact",
                        noun=None, attributes={
                            "object_id": str(object_id), "movement_role": direction,
                        },
                        evidence={"original_interaction": annotation},
                        video_path=video_path,
                        time_basis="hd_epic_mp4_frame_zero_based",
                        annotation_path=path,
                    )
                except ValueError as exc:
                    yield {"_reject": {"annotation": source_id, "error": str(exc)}}
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
                try:
                    yield event(
                        dataset="hd_epic", video_id=video_id,
                        annotation_id=source_id + ":gaze_priming",
                        event_type=(
                            "gaze_priming_before_pickup" if direction == "start"
                            else "gaze_priming_before_putdown"
                        ),
                        frame=primed_frame, fps=fps,
                        split="unspecified", label_origin="source_derived_gaze",
                        attributes={
                            "object_id": str(object_id),
                            "relative_to_interaction": direction,
                            "interaction_frame": frame,
                            "prime_gap_sec": number(prime.get("prime_gap")),
                        },
                        evidence={"official_prime_stats": prime},
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
