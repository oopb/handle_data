"""FEEL: deterministic force/contact-state rising edges, with no human/VLM labels.

This is SENSOR-DERIVED supervision, NOT visually verified first contact.
Edit SETTINGS and run: python feel.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from event_common import event, integer, number, write_records

# ======================== SETTINGS =========================
DATA_ROOT = Path("/data/FEEL")
OUTPUT_DIR = Path("./output/feel")
FORCE_GLOB = "force_*aligned*.csv"

# This CSV identifies frames on the synchronized FEEL recording timeline.
FRAME_COLUMN = "frame_idx"
VALID_ROW_COLUMN = "force_row_idx"  # -1 -> no aligned sample
FORCE_COLUMN = "force"               # change if your CSV uses another name
# Use a precomputed official contact/no-contact column if actually present;
# set to None to derive contact states from force readings with thresholds.
LABEL_COLUMN = None
CONTACT_VALUES = {"1", "true", "contact", "on"}
NONCONTACT_VALUES = {"0", "false", "no_contact", "non_contact", "off"}
# Calibrate these for the actual force units; the dataset has no universal
# threshold valid for every glove/sensor/session.
CONTACT_ON_THRESHOLD = None
CONTACT_OFF_THRESHOLD = None
MIN_STABLE_FRAMES = 2

# Match each aligned session to the actual available VIDEO and FPS; don't
# invent a canonical video timebase. Example key "Kitchen/P01".
FPS_BY_SESSION = {
    # "Kitchen/P01": 30.0,
}
DEFAULT_FPS = None
VIDEO_PATH_BY_SESSION = {
    # "Kitchen/P01": "/data/FEEL/Kitchen/P01_full.mp4",
}
# Prospective generation is disabled until synchronisation + video path is
# verified for each recording. No real video is downloaded by this script.
INCLUDE_PROSPECTIVE = False
HISTORY_SECONDS = 3.0
# ==========================================================


def _state(row):
    """True=contact, False=no-contact, None=ambiguous or missing."""
    if LABEL_COLUMN:
        value = str(row.get(LABEL_COLUMN, "")).strip().lower()
        if value in CONTACT_VALUES:
            return True
        if value in NONCONTACT_VALUES:
            return False
        return None
    reading = number(row.get(FORCE_COLUMN))
    if reading is None:
        return None
    if reading >= CONTACT_ON_THRESHOLD:
        return True
    if reading <= CONTACT_OFF_THRESHOLD:
        return False
    return None  # explicit grey zone, never guessed contact


def _aligned_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {FRAME_COLUMN}
        if LABEL_COLUMN:
            required.add(LABEL_COLUMN)
        else:
            required.add(FORCE_COLUMN)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        yield from reader


def convert(root=DATA_ROOT):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"FEEL data directory not found: {root}")
    if LABEL_COLUMN is None:
        if CONTACT_ON_THRESHOLD is None or CONTACT_OFF_THRESHOLD is None:
            raise ValueError("Set CONTACT_ON_THRESHOLD and CONTACT_OFF_THRESHOLD "
                             "to calibrated sensor values, or LABEL_COLUMN to "
                             "an existing official contact label column.")
        if not (0 <= CONTACT_OFF_THRESHOLD < CONTACT_ON_THRESHOLD):
            raise ValueError("Expected 0 <= CONTACT_OFF_THRESHOLD < CONTACT_ON_THRESHOLD")
    if MIN_STABLE_FRAMES < 1:
        raise ValueError("MIN_STABLE_FRAMES must be positive")
    paths = sorted(root.rglob(FORCE_GLOB))
    if not paths:
        raise FileNotFoundError(f"No FEEL aligned force CSV matching {FORCE_GLOB}")
    for path in paths:
        session = path.parent.relative_to(root).as_posix()
        hand = "left" if "left" in path.stem.lower() else (
            "right" if "right" in path.stem.lower() else "unknown")
        fps = FPS_BY_SESSION.get(session, DEFAULT_FPS)
        video_path = VIDEO_PATH_BY_SESSION.get(session)
        if INCLUDE_PROSPECTIVE and (fps is None or not video_path):
            raise ValueError(
                f"Prospective FEEL requires FPS and video path mapping for {session}"
            )
        previous_idx = None
        active = False
        armed = False
        noncontact_run = 0
        candidate_frames = []
        off_run = 0
        for row in _aligned_rows(path):
            frame = integer(row.get(FRAME_COLUMN))
            is_aligned = (not VALID_ROW_COLUMN or
                          integer(row.get(VALID_ROW_COLUMN)) != -1)
            state = _state(row) if is_aligned else None
            contiguous = (frame is not None and previous_idx is not None
                          and frame == previous_idx + 1)
            if not contiguous:
                # Cannot infer first contact from the first frame of a clip.
                active, armed = False, False
                candidate_frames, noncontact_run, off_run = [], 0, 0
            previous_idx = frame
            if frame is None or state is None:
                # Unknown labels/invalid sensor values break runs.
                active, armed = False, False
                candidate_frames, noncontact_run, off_run = [], 0, 0
                continue
            if not active:
                if state is False:
                    noncontact_run += 1
                    candidate_frames = []
                    if noncontact_run >= MIN_STABLE_FRAMES:
                        armed = True
                elif state is True and armed:
                    candidate_frames.append(frame)
                    if len(candidate_frames) >= MIN_STABLE_FRAMES:
                        onset = candidate_frames[0]
                        source_id = f"{hand}:{onset}"
                        yield event(
                            dataset="feel", video_id=session,
                            annotation_id=source_id,
                            event_type="force_contact_onset",
                            frame=onset, fps=fps, time_basis="aligned_sensor_frame",
                            split="unspecified", label_origin="sensor_derived",
                            video_path=video_path,
                            attributes={
                                "hand": hand, "label_column": LABEL_COLUMN,
                                "force_column": None if LABEL_COLUMN else FORCE_COLUMN,
                                "on_threshold": CONTACT_ON_THRESHOLD if not LABEL_COLUMN else None,
                                "off_threshold": CONTACT_OFF_THRESHOLD if not LABEL_COLUMN else None,
                                "min_stable_frames": MIN_STABLE_FRAMES,
                            },
                            evidence={"confirmed_at_frame": frame,
                                      "source_csv": path.name},
                            annotation_path=path,
                        )
                        active, armed = True, False
                        candidate_frames, noncontact_run = [], 0
                else:
                    candidate_frames = []
                    noncontact_run = 0
            elif state is False:
                off_run += 1
                if off_run >= MIN_STABLE_FRAMES:
                    active, armed = False, True
                    noncontact_run, off_run = MIN_STABLE_FRAMES, 0
            else:
                off_run = 0


def main():
    print(write_records(
        convert(), OUTPUT_DIR, include_prospective=INCLUDE_PROSPECTIVE,
        history_seconds=HISTORY_SECONDS,
    ))


if __name__ == "__main__":
    main()
