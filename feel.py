"""FEEL-Benchmark: export synchronized force as auxiliary signals only.

Legacy aligned CSV inputs require explicit contact labels or calibration.
Edit SETTINGS and run: python feel.py
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from event_common import event, integer, number, write_records

# ======================== SETTINGS =========================
DATA_ROOT = Path(__file__).resolve().parent.parent / "datadisk/datasets/FEEL"
OUTPUT_DIR = Path(__file__).resolve().parent / "output/feel"
FORCE_GLOB = "force_*aligned*.csv"

# This CSV identifies frames on the synchronized FEEL recording timeline.
FRAME_COLUMN = "frame_idx"
VALID_ROW_COLUMN = "force_row_idx"  # -1 -> no aligned sample; None only if alignment is verified
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


def auxiliary_force_observations(root=DATA_ROOT):
    """Read FEEL-Benchmark as signals only, never as contact event truth."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ImportError("FEEL Parquet signals require pyarrow: python -m pip install pyarrow") from exc
    root = Path(root)
    for info_path in sorted(root.rglob("meta/info.json")):
        trial = info_path.parent.parent
        info = json.loads(info_path.read_text(encoding="utf-8"))
        trial_id = trial.relative_to(root).as_posix()
        seen = set()
        # The episode manifest is authoritative. Extra stale MP4 files must
        # not create additional episodes or inflate the dataset size.
        with (trial / "meta/episodes.jsonl").open(encoding="utf-8") as episodes:
            for line in episodes:
                episode = json.loads(line)
                index = integer(episode["episode_index"])
                if index is None or index < 0 or index in seen:
                    raise ValueError(f"{trial}: invalid or duplicate episode index")
                seen.add(index)
                fields = {"episode_index": index,
                          "episode_chunk": index // info["chunks_size"],
                          "video_key": "observation.images.main"}
                data_path = trial / info["data_path"].format(**fields)
                video_path = trial / info["video_path"].format(**fields)
                columns = ["frame_index", "timestamp", "timestamp_ns",
                           "action.tactile.left.force_total", "action.tactile.right.force_total"]
                rows = parquet.read_table(data_path, columns=columns).to_pylist()
                if len(rows) != episode["length"]:
                    raise ValueError(f"{data_path}: episode length disagrees with its manifest")
                for frame, row in enumerate(rows):
                    if integer(row["frame_index"]) != frame:
                        raise ValueError(f"{data_path}: non-contiguous episode frame indices")
                    yield {
                        "record_type": "auxiliary_force_observation",
                        "dataset": "feel_benchmark", "trial_id": trial_id,
                        "split_group_id": trial_id, "episode_index": index, "frame_idx": frame,
                        "video_path": str(video_path), "video_available": video_path.is_file(),
                        "aligned_timestamp_sec": number(row["timestamp"]),
                        "device_timestamp_ns": integer(row["timestamp_ns"]),
                        "fps_nominal": number(info["fps"]),
                        "instrumented_hand": "right",
                        "module1_force_raw": number(row["action.tactile.left.force_total"]),
                        "module2_force_raw": number(row["action.tactile.right.force_total"]),
                        "force_units": "raw_counts", "usable_as_contact_truth": False,
                        "time_note": "dataset_aligned_relative_timestamp_verify_video_pts",
                        "provenance": {"data_path": str(data_path), "info_path": str(info_path)},
                    }
        if len(seen) != info["total_episodes"]:
            raise ValueError(f"{trial}: episode manifest disagrees with total_episodes")


def write_auxiliary_force(root=DATA_ROOT, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir) / "auxiliary"
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, episodes, trials = Counter(), set(), set()
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staging = Path(staging)
        with (staging / "force_observations.jsonl").open("w", encoding="utf-8") as dst:
            for row in auxiliary_force_observations(root):
                dst.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                counts["force_observations"] += 1
                episodes.add((row["trial_id"], row["episode_index"]))
                trials.add(row["trial_id"])
                if not row["video_available"]:
                    counts["observations_without_video"] += 1
                if row["module1_force_raw"] is None or row["module2_force_raw"] is None:
                    counts["observations_with_missing_force"] += 1
        stats = dict(counts, trials=len(trials), episodes=len(episodes), events=0, instances=0,
                     usable_as_contact_truth=False)
        (staging / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        for name in ("force_observations.jsonl", "stats.json"):
            os.replace(staging / name, output_dir / name)
    return stats


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
    if reading >= number(CONTACT_ON_THRESHOLD):
        return True
    if reading <= number(CONTACT_OFF_THRESHOLD):
        return False
    return None  # explicit grey zone, never guessed contact


def _aligned_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {FRAME_COLUMN}
        if VALID_ROW_COLUMN:
            required.add(VALID_ROW_COLUMN)
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
    if next(root.rglob("meta/info.json"), None) is not None:
        raise ValueError(
            "Found FEEL-Benchmark LeRobot data, not the aligned force CSV dataset. "
            "Its force values are auxiliary signals, not validated contact-onset/end truth; "
            "left/right columns are two sensor modules on the right hand. "
            "No contact event labels will be generated from these Parquet files."
        )
    if not LABEL_COLUMN:
        on, off = number(CONTACT_ON_THRESHOLD), number(CONTACT_OFF_THRESHOLD)
        if on is None or off is None:
            raise ValueError("Set CONTACT_ON_THRESHOLD and CONTACT_OFF_THRESHOLD "
                             "to calibrated sensor values, or LABEL_COLUMN to "
                             "an existing official contact label column.")
        if not (0 <= off < on):
            raise ValueError("Expected 0 <= CONTACT_OFF_THRESHOLD < CONTACT_ON_THRESHOLD")
    stable_frames = integer(MIN_STABLE_FRAMES)
    if stable_frames is None or stable_frames < 1:
        raise ValueError("MIN_STABLE_FRAMES must be a positive integer")
    paths = sorted(root.rglob(FORCE_GLOB))
    if not paths:
        raise FileNotFoundError(f"No FEEL aligned force CSV matching {FORCE_GLOB}")
    for path in paths:
        session = path.parent.relative_to(root).as_posix()
        hand = "left" if "left" in path.stem.lower() else (
            "right" if "right" in path.stem.lower() else "unknown")
        fps = FPS_BY_SESSION.get(session, DEFAULT_FPS)
        video_path = VIDEO_PATH_BY_SESSION.get(session)
        if INCLUDE_PROSPECTIVE and (number(fps) is None or number(fps) <= 0 or not video_path):
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
            if frame is not None and frame < 0:
                frame = None
            aligned_idx = integer(row.get(VALID_ROW_COLUMN)) if VALID_ROW_COLUMN else None
            is_aligned = (not VALID_ROW_COLUMN or
                          (aligned_idx is not None and aligned_idx >= 0))
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
                    if noncontact_run >= stable_frames:
                        armed = True
                elif state is True and armed:
                    candidate_frames.append(frame)
                    if len(candidate_frames) >= stable_frames:
                        onset = candidate_frames[0]
                        source_id = f"{path.relative_to(root).as_posix()}:{hand}:{onset}"
                        yield event(
                            dataset="feel", video_id=session,
                            annotation_id=source_id,
                            event_type="force_contact_onset",
                            frame=onset, fps=fps, time_basis="aligned_sensor_frame",
                            fps_source=(f"feel.FPS_BY_SESSION[{session!r}]" if session in
                                        FPS_BY_SESSION else "feel.DEFAULT_FPS"),
                            split="unspecified", label_origin="sensor_derived",
                            video_path=video_path,
                            attributes={
                                "hand": hand, "label_column": LABEL_COLUMN,
                                "force_column": None if LABEL_COLUMN else FORCE_COLUMN,
                                "on_threshold": CONTACT_ON_THRESHOLD if not LABEL_COLUMN else None,
                                "off_threshold": CONTACT_OFF_THRESHOLD if not LABEL_COLUMN else None,
                                "min_stable_frames": stable_frames,
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
                if off_run >= stable_frames:
                    active, armed = False, True
                    noncontact_run, off_run = stable_frames, 0
            else:
                off_run = 0


def main():
    if next(DATA_ROOT.rglob("meta/info.json"), None) is not None:
        print(write_auxiliary_force())
        return
    print(write_records(
        convert(), OUTPUT_DIR, include_prospective=INCLUDE_PROSPECTIVE,
        history_seconds=HISTORY_SECONDS,
    ))


if __name__ == "__main__":
    main()
