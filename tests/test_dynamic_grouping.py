"""Golden-label, automatic preparation, and history-boundary regressions."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import multiprocessing
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from multics.data.dataset import PrecomputedLatentDataset
from multics.grouping.budget import resolve
from multics.grouping.labels import generate_labels, track_labels
from multics.grouping.prepare import ensure_group_path, validate_generated_labels
from multics.grouping.relevance import history_components, relevance_periods
from multics.grouping.state import GroupingState, eligible_players

EXAMPLES = Path(__file__).resolve().parents[1] / "asset/examples/4players"


def _prepare_process(args):
    row, base, cache = args
    torch.set_num_threads(1)
    return ensure_group_path(row, base, num_players=4, cache_dir=cache)


class DynamicGroupingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.metadata = pd.read_csv(EXAMPLES / "metadata.csv")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = self.metadata.iloc[0].to_dict()
        self.state = GroupingState.load(EXAMPLES / self.row["grouping_state_path"])

    def missing_row(self):
        return dict(self.row, group_path="")

    def dataset(self, metadata, **kwargs):
        path = self.root / "metadata.csv"
        metadata.to_csv(path, index=False)
        return PrecomputedLatentDataset(
            str(EXAMPLES),
            str(path),
            steps_per_epoch=5,
            num_players=4,
            use_group_labels=True,
            num_groups=2,
            hier_group_format="membership",
            use_camera_pose=True,
            debug_no_retry=kwargs.pop("debug_no_retry", True),
            grouping_cache_dir=self.root / "cache",
            **kwargs,
        )

    def test_all_five_examples_all_24_fields_exact(self):
        self.assertEqual(len(self.metadata), 5)
        for _, row in self.metadata.iterrows():
            with self.subTest(case=int(row["case_id"])):
                state = GroupingState.load(EXAMPLES / row["grouping_state_path"])
                actual = generate_labels(state)
                expected = torch.load(
                    EXAMPLES / row["group_path"], weights_only=True
                )
                self.assertEqual(set(actual), set(expected))
                self.assertEqual(len(actual), 24)
                for key in actual:
                    tensor = torch.from_numpy(actual[key])
                    self.assertEqual(tensor.dtype, expected[key].dtype, key)
                    self.assertTrue(torch.equal(tensor, expected[key]), key)
                validate_generated_labels(
                    {k: torch.from_numpy(v) for k, v in actual.items()}, 4
                )

    def test_dynamic_and_round_start_original_fixtures(self):
        folder = Path(__file__).parent / "fixtures/grouping"
        for name in ("p4_round_start", "p6_moving_pairs", "p10_moving_pairs"):
            with self.subTest(name=name):
                state = GroupingState.load(folder / f"{name}.state.npz")
                actual = generate_labels(state)
                with np.load(
                    folder / f"{name}.labels.npz", allow_pickle=False
                ) as expected:
                    self.assertEqual(set(actual), set(expected.files))
                    for key in actual:
                        self.assertEqual(actual[key].dtype, expected[key].dtype, key)
                        np.testing.assert_array_equal(
                            actual[key], expected[key], err_msg=key
                        )
                if "moving" in name:
                    self.assertTrue(
                        any(
                            not np.array_equal(actual["period_membership"][0], frame)
                            for frame in actual["period_membership"][1:]
                        )
                    )

    def test_existing_labels_never_call_generator_or_require_state(self):
        row = dict(self.row, grouping_state_path="does-not-exist.npz")
        with patch(
            "multics.grouping.prepare.generate_labels", side_effect=AssertionError
        ):
            path = ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)
        self.assertEqual(Path(path), EXAMPLES / self.row["group_path"])
        self.assertFalse(list(self.root.iterdir()))

    def test_missing_path_generates_then_reuses_cache(self):
        row = dict(self.row, group_path="missing-labels.pt")
        path = ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with patch(
            "multics.grouping.prepare.generate_labels", side_effect=AssertionError
        ):
            self.assertEqual(
                path,
                ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root),
            )
        self.assertEqual(digest, hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def test_missing_column_and_empty_cells_work(self):
        for metadata in (
            self.metadata.drop(columns="group_path"),
            self.metadata.assign(group_path=""),
        ):
            with self.subTest(columns=list(metadata.columns)):
                dataset = self.dataset(metadata)
                self.assertTrue(all(Path(p).is_file() for p in dataset.group_paths))

    def test_dataset_inputs_and_rng_are_identical(self):
        random.seed(51)
        np.random.seed(51)
        torch.manual_seed(51)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        old = self.dataset(self.metadata)
        new = self.dataset(self.metadata.drop(columns="group_path"))
        for index in range(5):
            left, right = old[index], new[index]
            self.assertEqual(set(left), set(right))
            for key in left:
                if torch.is_tensor(left[key]):
                    self.assertTrue(torch.equal(left[key], right[key]), key)
                elif isinstance(left[key], dict):
                    for name in left[key]:
                        self.assertTrue(
                            torch.equal(left[key][name], right[key][name]), name
                        )
                else:
                    self.assertEqual(left[key], right[key], key)
        self.assertEqual(python_before, random.getstate())
        now = np.random.get_state()
        self.assertEqual(numpy_before[0], now[0])
        np.testing.assert_array_equal(numpy_before[1], now[1])
        self.assertEqual(numpy_before[2:], now[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_selected_samples_do_not_require_unselected_inputs(self):
        metadata = self.metadata.copy()
        metadata.loc[1:, "group_path"] = "unselected-missing.pt"
        metadata.loc[1:, "grouping_state_path"] = "unselected-missing.npz"
        dataset = self.dataset(metadata, grouping_sample_indices=[0])
        self.assertEqual(dataset.prepared_group_rows, frozenset({0}))
        with self.assertRaisesRegex(ValueError, "not prepared"):
            dataset[1]

    def test_corrupt_existing_file_is_not_replaced(self):
        path = self.root / "bad.pt"
        path.write_bytes(b"broken")
        with self.assertRaises(Exception):
            ensure_group_path(
                dict(self.row, group_path=str(path)), EXAMPLES, num_players=4
            )
        self.assertEqual(path.read_bytes(), b"broken")

    def test_corrupt_generated_cache_is_rejected(self):
        row = self.missing_row()
        path = Path(
            ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)
        )
        payload = torch.load(path, weights_only=True)
        payload["period_pair_used"][0] += 2
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "pair accounting"):
            ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)

    def test_consecutive_sample_prepares_every_source_row(self):
        metadata = self.metadata.iloc[:2].copy()
        metadata.loc[1, "start_frame"] = int(metadata.loc[0, "start_frame"]) + 165
        for player in range(1, 5):
            metadata.loc[1, f"video_path{player}"] = metadata.loc[
                0, f"video_path{player}"
            ]
        dataset = self.dataset(
            metadata, consecutive_rows=2, grouping_sample_indices=[0]
        )
        self.assertEqual(dataset.sample_start_rows, [0])
        self.assertEqual(dataset.prepared_group_rows, frozenset({0, 1}))
        joined = dataset._load_group_rows([0, 1], [42, 42], 83)
        for key, value in joined.items():
            expected = torch.cat(
                [dataset._load_group_labels(i, 42)[key] for i in (0, 1)]
            )[:83]
            self.assertTrue(torch.equal(value, expected), key)

    def test_incompatible_model_profile_only_blocks_generation(self):
        profile = (0, 6, 1.0, 3)
        self.dataset(self.metadata, grouping_model_profile=profile)
        with self.assertRaisesRegex(ValueError, "requires local/global hubs"):
            self.dataset(
                self.metadata.drop(columns="group_path"), grouping_model_profile=profile
            )

    def test_precomputed_other_player_count_remains_supported(self):
        payload = {
            "membership": torch.ones(42, 3, 5, dtype=torch.bool),
            "global_only": torch.zeros(42, 5, dtype=torch.bool),
            "eligible": torch.ones(5, dtype=torch.bool),
        }
        payload["membership"][:, 1:] = False
        path = self.root / "five_players.pt"
        torch.save(payload, path)
        actual = ensure_group_path({"group_path": str(path)}, EXAMPLES, num_players=5)
        self.assertEqual(actual, str(path))

    def test_existing_inconsistent_labels_rejected(self):
        payload = torch.load(EXAMPLES / self.row["group_path"], weights_only=True)
        payload["labels"][0, 0] = 1
        path = self.root / "inconsistent.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "disagree"):
            ensure_group_path(
                dict(self.row, group_path=str(path)), EXAMPLES, num_players=4
            )

    def test_partial_payload_cannot_pass_as_complete(self):
        payload = torch.load(EXAMPLES / self.row["group_path"], weights_only=True)
        del payload["period_membership"], payload["clip_total_cost"]
        payload["period_scores"][0, 0] = float("nan")
        path = self.root / "partial.pt"
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "all 24 fields"):
            ensure_group_path(
                dict(self.row, group_path=str(path)), EXAMPLES, num_players=4
            )

    def test_retry_stays_within_prepared_samples(self):
        dataset = self.dataset(
            self.metadata, grouping_sample_indices=[0], debug_no_retry=False
        )
        load = torch.load
        failed = False

        def fail_once(path, *args, **kwargs):
            nonlocal failed
            if str(path) == dataset.latent_paths[0][0] and not failed:
                failed = True
                raise OSError("transient input error")
            return load(path, *args, **kwargs)

        with patch("multics.data.dataset.torch.load", side_effect=fail_once):
            item = dataset[0]
        self.assertTrue(failed)
        self.assertEqual(item["source_rows"].tolist(), [0])

    def test_missing_state_and_reordered_players_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "48 frames"):
            ensure_group_path(
                dict(self.missing_row(), grouping_state_path="missing.npz"),
                EXAMPLES,
                num_players=4,
                cache_dir=self.root,
            )
        row = self.missing_row()
        row["video_path1"], row["video_path2"] = row["video_path2"], row["video_path1"]
        with self.assertRaisesRegex(ValueError, "player order"):
            ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)

    def test_missing_history_and_wrong_dtype_rejected(self):
        shorter = replace(
            self.state,
            **{
                key: getattr(self.state, key)[2:]
                for key in ("ticks", "pos", "yaw", "pitch", "alive", "present")
            },
        )
        with self.assertRaisesRegex(ValueError, "history"):
            shorter.validate()
        with self.assertRaisesRegex(ValueError, "yaw"):
            replace(self.state, yaw=self.state.yaw.astype(np.float64)).validate()

    def test_future_pose_does_not_change_current_period(self):
        state = self.state
        other = replace(state, pos=state.pos.copy())
        t = state.clip_start_frame
        cutoff = np.searchsorted(state.ticks, state.frame_to_tick(t))
        other.pos[cutoff + 1 :, 0, 0] += 10000
        span = state.frame_window(t, 165)
        left = relevance_periods(state, span, t, np.arange(4))[0]
        right = relevance_periods(other, span, t, np.arange(4))[0]
        np.testing.assert_array_equal(left.smooth, right.smooth)
        other.pos[cutoff, 0, 0] += 10000
        changed = relevance_periods(other, span, t, np.arange(4))[0]
        self.assertFalse(np.array_equal(left.smooth, changed.smooth))

    def test_wider_interaction_validity_survives_pose_only(self):
        state = replace(
            self.state,
            pos=np.zeros_like(self.state.pos),
            yaw=np.zeros_like(self.state.yaw),
            pitch=np.zeros_like(self.state.pitch),
            alive=np.ones_like(self.state.alive),
            present=np.ones_like(self.state.present),
        )
        state.pos[:, 1, 0] = 1300
        state.yaw[:, 1] = 180
        parts = history_components(state, state.clip_start_frame - 12, 13)
        self.assertEqual(parts["mutual"][0, 1], 0)
        self.assertEqual(parts["interaction_valid"][0, 1], 1)

    def test_eligibility_ignores_all_empty_ticks_but_not_missing_player(self):
        state = replace(self.state, present=np.ones_like(self.state.present))
        span = state.frame_window(state.clip_start_frame, 165)
        state.present[span.start] = False
        self.assertTrue(eligible_players(state, span).all())
        state.present[span.start + 1, 0] = False
        self.assertFalse(eligible_players(state, span)[0])

    def test_group_ids_follow_members_not_list_positions(self):
        self.assertEqual(
            track_labels([[[0, 1], [2, 3]], [[2, 3], [0, 1]]], 2), [[0, 1], [1, 0]]
        )

    def test_budget_profiles_and_unknown_player_count(self):
        for n, groups, pairs, size in (
            (4, 2, 10, 3), (6, 3, 15, 4), (8, 3, 19, 4), (10, 4, 45, 7)
        ):
            budget = resolve(n)
            self.assertEqual(
                (budget.groups, budget.pair_budget, budget.max_size),
                (groups, pairs, size),
            )
            self.assertEqual(
                (budget.local_attn_size, budget.a_w, budget.b_w), (7, 315, 126)
            )
        with self.assertRaisesRegex(ValueError, "4/6/8/10"):
            resolve(5)

    def test_cache_invalidates_when_state_or_algorithm_changes(self):
        row = self.missing_row()
        first = ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)
        state = replace(self.state, yaw=self.state.yaw.copy())
        state.yaw[0, 0] += 0.01
        state.save(self.root / "changed.npz")
        second = ensure_group_path(
            dict(row, grouping_state_path=str(self.root / "changed.npz")),
            EXAMPLES,
            num_players=4,
            cache_dir=self.root,
        )
        self.assertNotEqual(first, second)
        with patch("multics.grouping.prepare.ALGORITHM_VERSION", "test-next-version"):
            third = ensure_group_path(row, EXAMPLES, num_players=4, cache_dir=self.root)
        self.assertNotEqual(first, third)

    def test_atomic_failure_leaves_no_valid_cache(self):
        with patch(
            "multics.grouping.prepare.torch.save", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                ensure_group_path(
                    self.missing_row(), EXAMPLES, num_players=4, cache_dir=self.root
                )
        self.assertFalse(list(self.root.glob("*.pt")))
        self.assertFalse(list(self.root.glob("*.tmp")))
        self.assertTrue(
            Path(
                ensure_group_path(
                    self.missing_row(), EXAMPLES, num_players=4, cache_dir=self.root
                )
            ).is_file()
        )

    def test_two_processes_share_atomic_cache(self):
        args = (self.missing_row(), str(EXAMPLES), str(self.root))
        with ProcessPoolExecutor(
            2, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            paths = list(pool.map(_prepare_process, [args, args]))
        self.assertEqual(paths[0], paths[1])
        self.assertEqual(len(list(self.root.glob("*.pt"))), 1)
        self.assertFalse(list(self.root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
