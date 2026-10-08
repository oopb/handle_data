"""Small synthetic fixtures for all eight source adapters; no downloaded videos needed."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import convert_datasets as c


def jwrite(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def cwrite(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_time_utilities(self):
        self.assertEqual(c.seconds("01:02:03.5"), 3723.5)
        self.assertEqual(c.integer("000120.jpg"), 120)
        self.assertEqual(c.integer("10.0"), 10)
        self.assertIsNone(c.integer("10.5"))

    def test_no_fake_contact_from_interval(self):
        e = c.canonical("demo", "v", 1, "action_interval",
                        start=1, end=2, action="pick up")
        self.assertEqual(e["target_event"]["precision"], "interval")
        self.assertEqual(e["quality"]["label_origin"], "source_interval")
        self.assertEqual([x["mode"] for x in c.instances(e)], ["post_hoc"])
        with self.assertRaises(ValueError):
            c.canonical("demo", "v", 1, "action_interval", start=3, end=2)

    def test_meccano(self):
        cwrite(self.root / "train.csv", None,
               [["01", "3", "assemble gear", "000024.jpg", "000048.jpg"]])
        events = list(c.meccano(self.root))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["target_event"]["interval_start_sec"], 2)
        self.assertEqual(events[0]["source"]["split"], "train")

    def test_holoassist(self):
        jwrite(self.root / "Session01" / "annotations.json", {
            "annotations": [
                {"id": 1, "label": "Fine grained action", "start": 1.2,
                 "end": 2.8, "attributes": {"Verb": "approach", "Noun": "camera"}},
                {"id": 2, "label": "Conversation", "start": 1, "end": 2},
                {"id": 3, "label": "Coarse grained action", "start": 3,
                 "end": 4, "attributes": {"Action sentence": "set up camera"}}
            ]})
        events = list(c.holoassist(self.root))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["semantics"]["verb"], "approach")
        self.assertEqual(events[1]["target_event"]["event_type"], "step_interval")

    def test_hd_epic_priming(self):
        jwrite(self.root / "eye_gaze_priming" / "priming_info.json", {
            "P01-video": {"0": {
                "start": {"frame": 30, "prime_stats": {"frame_primed": 20}},
                "end": {"frame": 60, "prime_stats": {"frame_primed": -1}}
            }}})
        with patch.object(c, "HD_EPIC_INTERACTION_FPS", None):
            events = list(c.hd_epic(self.root))
        self.assertEqual([x["target_event"]["event_type"] for x in events],
                         ["pickup_frame", "putdown_frame"])
        self.assertIsNone(events[0]["target_event"]["timestamp_sec"])
        self.assertEqual([x["mode"] for x in c.instances(events[0])], ["post_hoc"])

    def test_feel_sensor_transitions(self):
        p = self.root / "Kitchen" / "P01" / "force_right_aligned.csv"
        forces = [0, 0, 1.2, 1.3, 1.5, 0.1, 0.1, 1.5, 1.6]
        cwrite(p, ["frame_idx", "force_row_idx", "force"],
               [[i, i, force] for i, force in enumerate(forces)])
        with patch.object(c, "FEEL_FRAME_FPS", 30), \
             patch.object(c, "FEEL_ON_THRESHOLD", 1.0), \
             patch.object(c, "FEEL_OFF_THRESHOLD", 0.3):
            events = list(c.feel(self.root))
        self.assertEqual([x["target_event"]["frame_idx"] for x in events], [2, 7])
        self.assertTrue(all(x["quality"]["label_origin"] == "sensor_derived"
                            for x in events))

    def test_epic_kitchens(self):
        cwrite(self.root / "EPIC_100_train.csv",
               ["narration_id", "video_id", "start_timestamp", "stop_timestamp",
                "narration", "verb", "noun", "start_frame", "stop_frame"],
               [["P01_01_0", "P01_01", "00:00:01.10", "00:00:03.25",
                 "open door", "open", "door", "66", "195"]])
        events = list(c.epic_kitchens(self.root))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["target_event"]["interval_end_sec"], 3.25)
        self.assertEqual(events[0]["target_event"]["precision"], "interval")

    def test_ego4d_source_contact(self):
        jwrite(self.root / "fho_main.json", {"videos": [{
            "video_uid": "v1", "video_metadata": {"fps": 30},
            "annotated_intervals": [{"clip_id": 1, "narrated_actions": [{
                "uid": "a1", "start_sec": 0.5, "end_sec": 2.5,
                "narration_text": "take cup", "structured_verb": "take",
                "critical_frames": {
                    "contact_frame": {"frame": 30},
                    "pnr_frame": {"frame": 40}}}]}]}]})
        events = list(c.ego4d(self.root))
        self.assertEqual(len(events), 3)
        self.assertEqual(events[1]["target_event"]["event_type"], "first_contact")
        self.assertEqual(events[1]["target_event"]["timestamp_sec"], 1.0)
        self.assertIn("prospective", [x["mode"] for x in c.instances(events[1])])

    def test_ego_exo4d(self):
        jwrite(self.root / "keystep_train.json", {"annotations": {
            "abc": {"take_uid": "abc", "scenario": "Assembly",
                    "segments": [{"start_time": 2, "end_time": 4,
                                  "step_id": 1, "step_unique_id": "S001",
                                  "step_name": "attach wheel"}]}}})
        events = list(c.ego_exo4d(self.root))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["semantics"]["coarse_action"], "attach wheel")
        self.assertEqual(events[0]["source"]["split"], "train")

    def test_assembly101(self):
        cwrite(self.root / "fine-grained-annotations" / "train.csv",
               ["id", "video", "start_frame", "end_frame", "action_cls",
                "verb_cls", "noun_cls"],
               [[7, "sequence/C10404_rgb.mp4", 60, 90, "pick up wheel",
                 "pick up", "wheel"]])
        coarse = self.root / "coarse-annotations" / "coarse_labels" / "assembly_seq.txt"
        coarse.parent.mkdir(parents=True, exist_ok=True)
        coarse.write_text("000000030 000000060 attach wheel\n", encoding="utf-8")
        splits = self.root / "coarse-annotations" / "coarse_splits" / "train_coarse_assembly.txt"
        splits.parent.mkdir(parents=True, exist_ok=True)
        splits.write_text("assembly_seq.txt notshared a30 suv\n", encoding="utf-8")
        events = list(c.assembly101(self.root))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["target_event"]["interval_start_sec"], 2)
        self.assertEqual(events[1]["source"]["split"], "train")

    def test_writer_outputs(self):
        cwrite(self.root / "EPIC_100_validation.csv",
               ["narration_id", "video_id", "start_timestamp",
                "stop_timestamp", "narration"],
               [["a", "P01_01", "1", "2", "wash cup"]])
        stats = c.convert_one("epic_kitchens", self.root, self.root / "out")
        self.assertEqual(stats["events"], 1)
        record = json.loads((self.root / "out" / "events.jsonl").read_text())
        self.assertEqual(record["semantics"]["coarse_action"], "wash cup")
        sample = json.loads((self.root / "out" / "train_instances.jsonl").read_text())
        self.assertEqual(sample["mode"], "post_hoc")


if __name__ == "__main__":
    unittest.main()
