"""Raw example inputs reproduce the bundled precomputed inputs exactly."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from multics.data.dataset import PrecomputedLatentDataset
from multics.data.raw_inputs import load_first_frame, prepare_raw_inputs

EXAMPLES = Path(__file__).resolve().parents[1] / "asset/examples/4players"


def _dataset(base, metadata, cache, **kwargs):
    options = dict(
        steps_per_epoch=5,
        num_players=4,
        use_camera_pose=True,
        use_group_labels=True,
        num_groups=2,
        hier_group_format="membership",
        debug_no_retry=True,
        grouping_cache_dir=cache,
    )
    return PrecomputedLatentDataset(str(base), str(metadata), **{**options, **kwargs})


class RawInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.raw = pd.read_csv(EXAMPLES / "inputs.csv")
        cls.prepared_path = prepare_raw_inputs(
            cls.raw, EXAMPLES, range(len(cls.raw)), 4, 165, cls.root / "prepared"
        )
        cls.prepared = pd.read_csv(cls.prepared_path)
        cls.reference = pd.read_csv(EXAMPLES / "metadata.csv")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_first_frames_are_model_resolution(self):
        for _, row in self.prepared.iterrows():
            for i in range(1, 5):
                frame = load_first_frame(row[f"image_path{i}"])
                self.assertEqual(frame.shape, (352, 640, 3))
                self.assertEqual(frame.dtype, np.uint8)

    def test_batches_match_precomputed_inputs(self):
        for index, row in self.prepared.iterrows():
            ref = self.reference.iloc[index]
            self.assertEqual(int(row["case_id"]), int(ref["case_id"]))
            for i in range(1, 5):
                latent = torch.load(EXAMPLES / ref[f"vae_path{i}"], weights_only=True)
                torch.save(latent[:, :1].clone(), row[f"vae_path{i}"])
        raw = _dataset(
            EXAMPLES,
            self.prepared_path,
            self.root / "cache-raw",
            allow_first_frame_latents=True,
        )
        precomputed = _dataset(
            EXAMPLES, EXAMPLES / "metadata.csv", self.root / "cache-precomputed"
        )
        for index in range(len(self.prepared)):
            with self.subTest(example=index + 1):
                a, b = raw[index], precomputed[index]
                self.assertEqual(tuple(a["latents"].shape), (4, 48, 1, 22, 40))
                self.assertTrue(torch.equal(a["latents"], b["latents"][:, :, :1]))
                for key in (
                    "keyboard",
                    "mouse",
                    "camera_pose",
                    "group_labels",
                    "active_group",
                    "membership",
                    "global_only",
                ):
                    self.assertEqual(a[key].dtype, b[key].dtype, key)
                    self.assertTrue(torch.equal(a[key], b[key]), key)
                self.assertEqual(a["prompt"], b["prompt"])

    def test_full_latents_still_required_without_opt_in(self):
        with self.assertRaises(RuntimeError):
            for index, row in self.prepared.iterrows():
                for i in range(1, 5):
                    torch.save(torch.zeros(48, 1, 22, 40), row[f"vae_path{i}"])
            _dataset(EXAMPLES, self.prepared_path, self.root / "cache-strict")[0]

    def _check_player_examples(self, players):
        base = EXAMPLES.parent / f"{players}players"
        raw = pd.read_csv(base / "inputs.csv")
        prepared = prepare_raw_inputs(
            raw, base, range(len(raw)), players, 165,
            self.root / f"prepared-{players}"
        )
        for _, row in pd.read_csv(prepared).iterrows():
            for i in range(1, players + 1):
                torch.save(
                    torch.zeros(48, 1, 22, 40, dtype=torch.bfloat16),
                    row[f"vae_path{i}"],
                )
        dataset = _dataset(
            base,
            prepared,
            self.root / f"cache-{players}",
            num_players=players,
            num_groups=(players - 1).bit_length(),
            allow_first_frame_latents=True,
        )
        for index in range(len(raw)):
            with self.subTest(example=index + 1):
                sample = dataset[index]
                self.assertEqual(tuple(sample["latents"].shape), (players, 48, 1, 22, 40))
                self.assertEqual(tuple(sample["keyboard"].shape), (players, 165, 23))
                self.assertEqual(tuple(sample["camera_pose"].shape), (players, 42, 12))
                self.assertEqual(tuple(sample["membership"].shape),
                                 (42, (players - 1).bit_length(), players))
                self.assertEqual(len(sample["prompt"]), players)

    def test_cross_player_examples_load(self):
        for players in (6, 8, 10):
            with self.subTest(players=players):
                self._check_player_examples(players)

    def test_runs_do_not_share_prepared_files(self):
        first = prepare_raw_inputs(self.raw, EXAMPLES, [0], 4, 165,
                                   self.root / "shared-output")
        before = Path(first).read_bytes()
        second = prepare_raw_inputs(self.raw, EXAMPLES, [1], 4, 165,
                                    self.root / "shared-output")
        self.assertNotEqual(Path(first).parent, Path(second).parent)
        self.assertEqual(Path(first).read_bytes(), before)

    def test_selected_case_does_not_open_other_tick_logs(self):
        raw = self.raw.copy()
        raw.loc[1, "ticks_path1"] = "missing-unselected.parquet"
        metadata = prepare_raw_inputs(raw, EXAMPLES, [0], 4, 165,
                                      self.root / "selected-only")
        table = pd.read_csv(metadata)
        self.assertEqual(len(table), len(raw))
        for i in range(1, 5):
            latent = torch.load(EXAMPLES / self.reference.iloc[0][f"vae_path{i}"],
                                weights_only=True)
            torch.save(latent[:, :1].clone(), table.iloc[0][f"vae_path{i}"])
        dataset = _dataset(EXAMPLES, metadata, self.root / "selected-cache",
                           allow_first_frame_latents=True,
                           grouping_sample_indices=[0])
        self.assertEqual(tuple(dataset[0]["latents"].shape), (4, 48, 1, 22, 40))


if __name__ == "__main__":
    unittest.main()
