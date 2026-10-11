"""Regression checks for source fidelity, causal templates and failed exports."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ego4d
import feel
import hd_epic
from event_common import event, training_instances, write_records


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def dump(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def record(self, **kwargs):
        args = dict(dataset="x", video_id="v", annotation_id="a",
                    event_type="first_contact", frame=30, fps=30,
                    action="take the cup", time_basis="canonical_video_frame")
        args.update(kwargs)
        return event(**args)

    def force_csv(self, rows, fields=None, name="force_left_aligned.csv"):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(fields or ["frame_idx", "force_row_idx", "force"])
            writer.writerows(rows)
        return path

    def test_main_accepts_official_integer_critical_frames(self):
        self.dump("fho_main.json", {"videos": [{
            "video_uid": "v", "video_metadata": {"fps": 30},
            "annotated_intervals": [{"clip_uid": "c", "narrated_actions": [{
                "uid": "a", "start_sec": 0, "end_sec": 2,
                "critical_frames": {"contact_frame": 30, "pnr_frame": 45},
                "clip_critical_frames": {"contact_frame": 5, "pnr_frame": 20}
            }]}]
        }]})
        records = list(ego4d.convert(self.root))
        self.assertEqual([r["target_event"]["frame_idx"] for r in records], [30, 45])
        self.assertIn("fho_main.json", records[0]["media"]["fps_source"])

    def test_ego4d_source_timestamp_takes_priority(self):
        self.dump("fho_hands_train.json", {"clips": [{
            "clip_uid": "c", "video_uid": "v", "frames": [{
                "contact_frame": {"frame": 30, "sec": None, "timestamp_sec": 1.1}
            }]
        }]})
        with patch.object(ego4d, "DEFAULT_FPS", 30):
            record = list(ego4d.convert(self.root))[0]
        self.assertEqual(record["target_event"]["timestamp_sec"], 1.1)
        self.assertEqual(record["target_event"]["time_note"], "source_timestamp")

    def test_ego4d_ids_survive_reordering(self):
        actions = [dict(action_start_frame=start, action_end_frame=start + 20,
                        contact_frame={"frame": start + 5}) for start in (10, 50)]
        def convert_order(order):
            self.dump("fho_hands_train.json", {"clips": [{
                "clip_uid": "c", "video_uid": "v", "frames": order
            }]})
            return {r["event_id"] for r in ego4d.convert(self.root)}
        self.assertEqual(convert_order(actions), convert_order(actions[::-1]))

    def test_ego4d_ambiguous_narration_is_not_arbitrarily_joined(self):
        self.dump("fho_main.json", {"videos": [{"video_uid": "v",
            "annotated_intervals": [{"narrated_actions": [
                dict(uid=uid, start_frame=0, end_frame=90, narration_text=text)
                for uid, text in (("a", "take cup"), ("b", "take spoon"))
            ]}]
        }]})
        self.dump("fho_hands_train.json", {"clips": [{
            "clip_uid": "c", "video_uid": "v", "frames": [{
                "action_start_frame": 0, "action_end_frame": 90,
                "contact_frame": {"frame": 30}
            }]
        }]})
        record = list(ego4d.convert(self.root))[0]
        self.assertIsNone(record["semantics"]["coarse_action"])

    def test_ego4d_multiple_main_releases_fail_explicitly(self):
        self.dump("v1/fho_main.json", {"videos": []})
        self.dump("v2/fho_main.json", {"videos": []})
        with self.assertRaisesRegex(ValueError, "Multiple"):
            list(ego4d.convert(self.root))

    def test_event_rejects_invalid_coordinates_and_context(self):
        for kwargs in ({"frame": 1.5, "timestamp": 1}, {"timestamp": float("nan")},
                       {"fps": float("inf")}, {"context_start": "bad", "context_end": 2},
                       {"timestamp": 5, "context_start": 0, "context_end": 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.record(**kwargs)

    def test_prospective_is_strictly_before_target_and_within_context(self):
        records = list(training_instances(
            self.record(frame=180, context_start=5, context_end=7),
            include_prospective=True, history_seconds=3))
        stream = records[1]
        self.assertEqual(stream["input"]["query_time_sec"], 5)
        self.assertLess(stream["input"]["query_time_sec"], stream["answer"]["timestamp_sec"])
        self.assertIn("take the cup", stream["question"])
        self.assertTrue(stream["response_policy"]["only_observe_frames_at_or_before_current_time"])
        for kwargs in ({"frame": 0}, {"context_start": 1, "context_end": 2}):
            with self.subTest(kwargs=kwargs):
                samples = list(training_instances(self.record(**kwargs),
                               include_prospective=True, history_seconds=3))
                self.assertEqual([s["mode"] for s in samples], ["post_hoc"])

    def test_opaque_object_reference_keeps_event_but_skips_training(self):
        stats = write_records([self.record(action=None, attributes={"object_id": "0"})],
                              self.root / "out")
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["events_without_grounded_query"], 1)
        self.assertEqual((self.root / "out/train_instances.jsonl").read_text(), "")

    def test_bad_history_is_rejected_before_output_changes(self):
        for value in (0, -1, float("nan"), float("inf"), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                write_records([self.record()], self.root / "out", history_seconds=value)
        self.assertFalse((self.root / "out").exists())

    def test_failed_conversion_preserves_all_previous_outputs(self):
        output = self.root / "out"
        write_records([self.record()], output)
        previous = {p.name: p.read_bytes() for p in output.iterdir()}
        def failing():
            yield self.record(annotation_id="b")
            raise ValueError("bad source")
        with self.assertRaisesRegex(ValueError, "bad source"):
            write_records(failing(), output)
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, previous)

    def test_conflicting_duplicate_labels_or_splits_fail(self):
        for kwargs in ({"frame": 31}, {"split": "val"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "Conflicting"):
                write_records([self.record(split="train"), self.record(**kwargs)],
                              self.root / "out")
        stats = write_records([self.record(), self.record()], self.root / "out")
        self.assertEqual(stats["events"], 1)
        self.assertEqual(stats["duplicate"], 1)

    def test_stats_report_origins_and_source_video_split_overlap(self):
        stats = write_records([self.record(split="train"),
                              self.record(annotation_id="b", split="val")], self.root / "out")
        self.assertEqual(stats["origin:source_exact"], 2)
        self.assertEqual(stats["cross_split_videos"], {"x:v": ["train", "val"]})

    def test_repeated_events_do_not_export_identical_queries_with_different_answers(self):
        stats = write_records([self.record(), self.record(annotation_id="b", frame=60)],
                              self.root / "out", include_prospective=False)
        self.assertEqual(stats["events"], 2)
        self.assertEqual(stats["instances"], 0)
        self.assertEqual(stats["instances_with_ambiguous_query"], 2)

    def test_identical_training_queries_are_not_counted_as_new_instances(self):
        stats = write_records([self.record(), self.record(annotation_id="b")],
                              self.root / "out", include_prospective=False)
        self.assertEqual(stats["events"], 2)
        self.assertEqual(stats["instances"], 1)
        self.assertEqual(stats["duplicate_training_instances"], 1)

    def test_same_training_query_across_splits_is_excluded_from_both(self):
        stats = write_records([self.record(split="train"),
                              self.record(annotation_id="b", split="val")],
                              self.root / "out", include_prospective=False)
        self.assertEqual(stats["instances"], 0)
        self.assertEqual(stats["instances_with_ambiguous_query"], 2)
        self.assertEqual(stats["cross_split_videos"], {"x:v": ["train", "val"]})

    def test_feel_missing_sync_column_is_not_assumed_aligned(self):
        self.force_csv([(0, 0), (1, 0), (2, 2), (3, 2)], fields=["frame_idx", "force"])
        with patch.object(feel, "CONTACT_ON_THRESHOLD", 1), \
             patch.object(feel, "CONTACT_OFF_THRESHOLD", .2):
            with self.assertRaisesRegex(ValueError, "force_row_idx"):
                list(feel.convert(self.root))

    def test_feel_bad_sync_indices_break_contact_runs(self):
        for bad in ("", "oops", -1, -2, 1.5):
            self.force_csv([(0, 0, 0), (1, 1, 0), (2, bad, 2), (3, 3, 2)])
            with self.subTest(bad=bad), patch.object(feel, "CONTACT_ON_THRESHOLD", 1), \
                 patch.object(feel, "CONTACT_OFF_THRESHOLD", .2):
                records = list(feel.convert(self.root))
                self.assertFalse(any("target_event" in r for r in records))

    def test_feel_frame_gaps_and_unknowns_do_not_bridge_edges(self):
        for frames in ((0, 1, 3, 4), (0, 1, -1, 2), (0, 1, "", 2)):
            self.force_csv([(f, i, force) for i, (f, force) in
                            enumerate(zip(frames, (0, 0, 2, 2)))])
            with self.subTest(frames=frames), patch.object(feel, "CONTACT_ON_THRESHOLD", 1), \
                 patch.object(feel, "CONTACT_OFF_THRESHOLD", .2):
                self.assertFalse(any("target_event" in r for r in feel.convert(self.root)))

    def test_feel_thresholds_and_stability_are_validated(self):
        self.force_csv([])
        for on, off, stable in ((float("inf"), .2, 2), (1, .2, 1.5), (1, .2, True)):
            with self.subTest(on=on, stable=stable), patch.object(feel, "CONTACT_ON_THRESHOLD", on), \
                 patch.object(feel, "CONTACT_OFF_THRESHOLD", off), \
                 patch.object(feel, "MIN_STABLE_FRAMES", stable):
                with self.assertRaises(ValueError):
                    list(feel.convert(self.root))

    def test_feel_source_streams_have_distinct_ids_and_sensor_precision(self):
        rows = [(0, 0, 0), (1, 1, 0), (2, 2, 2), (3, 3, 2)]
        self.force_csv(rows, name="force_a_aligned.csv")
        self.force_csv(rows, name="force_b_aligned.csv")
        with patch.object(feel, "CONTACT_ON_THRESHOLD", 1), \
             patch.object(feel, "CONTACT_OFF_THRESHOLD", .2):
            records = list(feel.convert(self.root))
        self.assertEqual(len({r["event_id"] for r in records}), 2)
        self.assertEqual(records[0]["target_event"]["precision"], "sensor_frame")
        self.assertEqual(records[0]["evidence"]["confirmed_at_frame"], 3)

    def test_hd_epic_rejects_priming_outside_official_window(self):
        for primed in (90, 181):
            self.dump("priming_info.json", {"v": {"0": {"start": {
                "frame": 180, "prime_stats": {"frame_primed": primed, "prime_window_start": 100}
            }}}})
            with self.subTest(primed=primed):
                records = list(hd_epic.convert(self.root))
                positives = [r for r in records if "target_event" in r]
                self.assertEqual([r["target_event"]["event_type"] for r in positives], ["object_pickup"])
                self.assertTrue(any("_reject" in r for r in records))

    def test_hd_epic_gaze_selection_is_independent_of_pickup_selection(self):
        self.dump("priming_info.json", {"v": {"0": {"start": {
            "frame": 180, "prime_stats": {"frame_primed": 170}
        }}}})
        with patch.object(hd_epic, "INCLUDE_PICKUP", False):
            records = list(hd_epic.convert(self.root))
        self.assertEqual([r["target_event"]["event_type"] for r in records],
                         ["gaze_priming_before_pickup"])

    def test_ego4d_invalid_source_time_is_not_replaced_by_fps(self):
        self.dump("fho_hands_train.json", {"clips": [{
            "clip_uid": "c", "video_uid": "v", "frames": [{
                "contact_frame": {"frame": 30, "sec": "bad"}
            }]
        }]})
        with patch.object(ego4d, "DEFAULT_FPS", 30):
            records = list(ego4d.convert(self.root))
        self.assertEqual(len(records), 1)
        self.assertIn("_reject", records[0])

    def test_official_integer_fixture_exports_both_tasks_without_future_inputs(self):
        self.dump("fho_main.json", {"videos": [{
            "video_uid": "v", "video_metadata": {"fps": 30},
            "annotated_intervals": [{"clip_uid": "c", "narrated_actions": [{
                "uid": "a", "start_sec": 0, "end_sec": 2,
                "narration_text": "take the cup",
                "critical_frames": {"contact_frame": 30, "pnr_frame": 45}
            }]}]
        }]})
        output = self.root / "out"
        (self.root / "v.mp4").touch()
        with patch.object(ego4d, "VIDEO_ROOT", self.root):
            stats = write_records(ego4d.convert(self.root), output)
        self.assertEqual((stats["events"], stats["instances"]), (2, 4))
        instances = [json.loads(line) for line in
                     (output / "train_instances.jsonl").read_text().splitlines()]
        streams = [s for s in instances if s["mode"] == "prospective"]
        for sample in streams:
            self.assertNotIn("target_time_sec", sample["input"])
            self.assertNotIn("context_end_sec", sample["input"])
            self.assertEqual(sample["answer"]["time_note"],
                             "nominal_fps_approximation_verify_pts")
            self.assertIn("take the cup", sample["question"])


if __name__ == "__main__":
    unittest.main()
