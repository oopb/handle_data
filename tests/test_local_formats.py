"""Small fixtures following the structures found in datadisk/datasets."""
import csv
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ego4d
import feel
import hd_epic
from event_common import event, training_instances, write_records


class LocalFormatsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, directory in (("CLIP_VIDEO_ROOT", "clips"), ("VIDEO_ROOT", "full_scale")):
            setting = patch.object(ego4d, name, self.root / directory)
            setting.start()
            self.addCleanup(setting.stop)

    def dump(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def main_annotation(self):
        self.dump("fho_main.json", {"videos": [{
            "video_uid": "v", "video_metadata": {"fps": 30},
            "annotated_intervals": [{
                "clip_uid": "c", "clip_parent_start_sec": 592,
                "clip_parent_start_frame": 17759, "clip_parent_end_sec": 908,
                "narrated_actions": [{"uid": "a", "start_sec": 601.9543619333334,
                    "end_sec": 609.9543619333334, "narration_text": "grab a container",
                    "critical_frames": {"contact_frame": 18169},
                    "clip_critical_frames": {"contact_frame": 410}}]
            }]
        }]})

    def test_unannotated_hands_does_not_hide_main_labels(self):
        self.main_annotation()
        self.dump("fho_hands_test_unannotated.json", {"split": "test", "clips": [
            {"video_uid": "v", "clip_uid": "c", "frames": [{"pre_frame": {"frame": 1}}]}]})
        records = list(ego4d.convert(self.root))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_event"]["frame_idx"], 18169)

    def test_available_clip_uses_local_training_coordinates_and_source_labels(self):
        self.main_annotation()
        (self.root / "clips").mkdir()
        (self.root / "clips/c.mp4").touch()
        self.dump("fho_lta_train.json", {"clips": [{"video_uid": "v", "clip_uid": "c"}]})
        record = list(ego4d.convert(self.root))[0]
        self.assertEqual(record["source"]["split"], "train")
        self.assertEqual(record["target_event"]["frame_idx"], 18169)
        instances = list(training_instances(record, include_prospective=True, history_seconds=3))
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]["input"]["video_id"], "c")
        self.assertEqual(instances[0]["answer"]["frame_idx"], 410)
        # The source's frame and second offsets differ by one frame. Subtract
        # frame offsets before converting nominal time, not source seconds.
        self.assertAlmostEqual(instances[0]["answer"]["timestamp_sec"], 410 / 30)
        self.assertAlmostEqual(instances[0]["input"]["context_start_sec"], 9.9543619333334)
        self.assertEqual(instances[0]["input"]["source_video_id"], "v")

    def test_missing_video_retains_event_without_emitting_unusable_training(self):
        self.main_annotation()
        stats = write_records(ego4d.convert(self.root), self.root / "out")
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["instances"], 0)
        self.assertEqual(stats["events_without_available_media"], 1)

    def test_clip_split_conflict_remains_unspecified(self):
        self.main_annotation()
        for split in ("train", "val"):
            self.dump(f"fho_lta_{split}.json", {"clips": [{"clip_uid": "c"}]})
        record = list(ego4d.convert(self.root))[0]
        self.assertEqual(record["source"]["split"], "unspecified")

    def test_full_scale_media_has_zero_offsets(self):
        self.main_annotation()
        (self.root / "full_scale").mkdir()
        (self.root / "full_scale/v.mp4").touch()
        record = list(ego4d.convert(self.root))[0]
        self.assertEqual(record["media"]["source_start_frame"], 0)
        self.assertEqual(record["media"]["source_start_sec"], 0)

    def test_hd_epic_nested_video_and_mp4_sidecar_time(self):
        self.dump("priming_info.json", {"P01-vid": {"0": {"start": {"frame": 2}}}})
        directory = self.root / "Videos/P01"
        directory.mkdir(parents=True)
        (directory / "P01-vid.mp4").touch()
        with (directory / "P01-vid_mp4_to_vrs_time_ns.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["mp4_time_ns", "vrs_device_time_ns"])
            writer.writerows([(0, 1000000000000), (33000000, 1000033000000),
                             (67000000, 1000067000000)])
        with patch.object(hd_epic, "VIDEO_ROOT", self.root / "Videos"):
            record = list(hd_epic.convert(self.root))[0]
        self.assertEqual(record["target_event"]["timestamp_sec"], .067)
        self.assertEqual(record["target_event"]["time_note"], "source_timestamp")
        self.assertTrue(record["media"]["available"])

    def test_lerobot_force_is_not_misidentified_or_promoted_to_contact_truth(self):
        self.dump("20260110/trial/meta/info.json", {"fps": 10, "features": {
            "action.tactile.left.force_total": {"dtype": "float32"}}})
        with self.assertRaisesRegex(ValueError, "auxiliary signals"):
            list(feel.convert(self.root))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "requires optional Parquet reader")
    def test_lerobot_export_preserves_force_and_clock_without_contact_labels(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        info = {"fps": 10, "total_episodes": 1, "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
        self.dump("20260110/trial/meta/info.json", info)
        trial = self.root / "20260110/trial"
        (trial / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n')
        data = trial / "data/chunk-000/episode_000000.parquet"
        data.parent.mkdir(parents=True)
        parquet.write_table(arrow.table({"frame_index": [0, 1], "timestamp": [0., .1],
            "timestamp_ns": [1000000000000, 1000100000000],
            "action.tactile.left.force_total": [0., 5.],
            "action.tactile.right.force_total": [0., 7.]}), data)
        # An extra MP4 has no corresponding episode in the manifest.
        (trial / "extra_episode.mp4").touch()
        rows = list(feel.auxiliary_force_observations(self.root))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["module1_force_raw"], 5.)
        self.assertEqual(rows[1]["instrumented_hand"], "right")
        self.assertEqual(rows[1]["aligned_timestamp_sec"], .1)
        self.assertEqual(rows[1]["device_timestamp_ns"], 1000100000000)
        self.assertFalse(rows[1]["usable_as_contact_truth"])
        self.assertNotIn("event_type", rows[1])
        stats = feel.write_auxiliary_force(self.root, self.root / "out")
        self.assertEqual((stats["episodes"], stats["events"], stats["instances"]), (1, 0, 0))
        self.assertFalse((self.root / "out/events.jsonl").exists())

    def test_media_outside_event_has_no_training_sample(self):
        record = event(dataset="x", video_id="v", annotation_id="a", event_type="first_contact",
                       frame=30, fps=30, action="take cup", media_source_start_frame=40)
        self.assertEqual(list(training_instances(record, include_prospective=True, history_seconds=3)), [])


if __name__ == "__main__":
    unittest.main()
