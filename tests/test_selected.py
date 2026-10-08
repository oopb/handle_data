"""Synthetic validation for the selected three model-free datasets."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ego4d
import feel
import hd_epic
from event_common import event, write_records


def dump_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_direct_event_requires_existing_timestamp_or_frame(self):
        with self.assertRaises(ValueError):
            event(dataset="x", video_id="a", annotation_id="a",
                  event_type="first_contact")
        record = event(dataset="x", video_id="a", annotation_id="a",
                       event_type="first_contact", frame=30, fps=30)
        self.assertEqual(record["target_event"]["timestamp_sec"], 1)
        self.assertEqual(record["target_event"]["time_note"],
                         "nominal_fps_approximation_verify_pts")

    def test_ego4d_hands_and_narration_join(self):
        dump_json(self.root / "fho_main.json", {"videos": [{
            "video_uid": "v1", "video_metadata": {"fps": 30},
            "annotated_intervals": [{"narrated_actions": [{
                "uid": "narr1", "start_frame": 10, "end_frame": 90,
                "start_sec": 10/30, "end_sec": 3,
                "structured_verb": "take", "narration_text": "take the cup"
            }]}]
        }]})
        dump_json(self.root / "fho_hands_train.json", {
            "split": "train", "clips": [{
                "clip_uid": "clip1", "video_uid": "v1", "frames": [{
                    "action_start_sec": 10/30, "action_end_sec": 3,
                    "action_start_frame": 10, "action_end_frame": 90,
                    "contact_frame": {"frame": 40, "clip_frame": 30},
                    "pnr_frame": {"frame": 50, "clip_frame": 40}
                }]
            }]
        })
        events = list(ego4d.convert(self.root))
        self.assertEqual(len(events), 2)
        self.assertEqual([e["target_event"]["event_type"] for e in events],
                         ["first_contact", "state_change_pnr"])
        self.assertEqual(events[0]["source"]["split"], "train")
        self.assertEqual(events[0]["semantics"]["coarse_action"], "take the cup")
        self.assertEqual(events[0]["target_event"]["frame_idx"], 40)
        self.assertEqual(events[0]["target_event"]["timestamp_sec"], 40/30)
        stats = write_records(events, self.root / "out")
        self.assertEqual(stats["events"], 2)
        self.assertEqual(stats["mode:prospective"], 2)

    def test_ego4d_fallback_main(self):
        dump_json(self.root / "fho_main.json", {"videos": [{
            "video_uid": "v1", "video_metadata": {"fps": 30},
            "annotated_intervals": [{"clip_uid": "c", "narrated_actions": [{
                "uid": "a", "start_sec": 0, "end_sec": 1,
                "critical_frames": {"contact_frame": {"sec": .5}}
            }]}]
        }]})
        events = list(ego4d.convert(self.root))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["target_event"]["timestamp_sec"], .5)
        self.assertEqual(events[0]["source"]["split"], "unspecified")

    def test_hd_epic_picks_puts_gaze_and_exclusion(self):
        dump_json(self.root / "eye_gaze_priming" / "priming_info.json", {
            "P01-01": {
                "0": {
                    "start": {"frame": 180, "prime_stats": {
                        "frame_primed": 170, "prime_gap": .333
                    }},
                    "end": {"frame": 250, "prime_stats": {
                        "frame_primed": -2
                    }}
                },
            }
        })
        with patch.dict(hd_epic.FPS_BY_VIDEO, {"P01-01": 30}):
            events = list(hd_epic.convert(self.root))
        self.assertEqual(len(events), 3)
        self.assertEqual(
            [e["target_event"]["event_type"] for e in events],
            ["object_pickup", "gaze_priming_before_pickup", "object_putdown"],
        )
        self.assertEqual(events[0]["target_event"]["timestamp_sec"], 6)
        self.assertEqual(events[1]["quality"]["label_origin"], "source_derived_gaze")

    def test_hd_epic_unknown_fps_does_not_invent_time(self):
        dump_json(self.root / "eye_gaze_priming" / "priming_info.json",
                  {"vid": {"0": {"start": {"frame": 20}}}})
        events = list(hd_epic.convert(self.root))
        self.assertEqual(events[0]["target_event"]["frame_idx"], 20)
        self.assertIsNone(events[0]["target_event"]["timestamp_sec"])
        stats = write_records(events, self.root / "out")
        self.assertNotIn("mode:prospective", stats)

    def test_feel_hysteresis_and_start_without_noncontact(self):
        path = self.root / "Kitchen" / "P01" / "force_left_aligned.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        forces = [1.6, 1.6, 0.0, 0.0, 1.2, 1.3, 1.4, 0.0, 0.0, 1.2, 1.5]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "force_row_idx", "force"])
            w.writerows((i, i, force) for i, force in enumerate(forces))
        with patch.object(feel, "CONTACT_ON_THRESHOLD", 1.0), \
             patch.object(feel, "CONTACT_OFF_THRESHOLD", 0.2), \
             patch.dict(feel.FPS_BY_SESSION, {"Kitchen/P01": 30}):
            events = list(feel.convert(self.root))
        self.assertEqual([e["target_event"]["frame_idx"] for e in events], [4, 9])
        self.assertEqual(events[0]["quality"]["label_origin"], "sensor_derived")

    def test_feel_existing_contact_column(self):
        path = self.root / "Lab" / "P02" / "force_right_aligned.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "contact_label"])
            for i, label in enumerate(["0", "0", "1", "1", "1", "0", "0"]):
                w.writerow([i, label])
        with patch.object(feel, "LABEL_COLUMN", "contact_label"):
            events = list(feel.convert(self.root))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["target_event"]["frame_idx"], 2)

    def test_feel_never_infers_uncalibrated_contact(self):
        with patch.object(feel, "LABEL_COLUMN", None), \
             patch.object(feel, "CONTACT_ON_THRESHOLD", None), \
             patch.object(feel, "CONTACT_OFF_THRESHOLD", None):
            with self.assertRaisesRegex(ValueError, "calibrated"):
                list(feel.convert(self.root))


if __name__ == "__main__":
    unittest.main()
