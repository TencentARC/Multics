"""CPU-only checks for inferred topology and checkpoint compatibility."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from multics.inference_args import parse_args as infer_args
from multics.training_args import parse_args as train_args


ROOT = Path(__file__).resolve().parents[1]


class HierConfigTests(unittest.TestCase):
    def resolve(self, kind, *options):
        parser = train_args if kind == "train" else infer_args
        output = io.StringIO()
        with patch.object(sys, "argv", [f"{kind}.py", *options, "--print-config"]):
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as result:
                    parser()
        self.assertEqual(result.exception.code, 0)
        return json.loads(output.getvalue())

    def test_player_boundaries_and_output_paths(self):
        # Check exact powers of two and their immediate neighbors.
        for kind in ("train", "inference"):
            for players, groups in (
                (2, 1),
                (3, 2),
                (4, 2),
                (5, 3),
                (6, 3),
                (8, 3),
                (9, 4),
                (10, 4),
            ):
                with self.subTest(kind=kind, players=players):
                    args = self.resolve(kind, "--num_players", str(players))
                    self.assertEqual(args["num_groups"], groups)
                    self.assertEqual(args["num_hub_tokens"], groups * 5 + 6)
                    output = args["output_path" if kind == "train" else "output_dir"]
                    expected = (
                        f"outputs/train_{players}players"
                        if kind == "train"
                        else "outputs/inference/" + (
                            "4players" if players == 4 else f"4infer{players}"
                        )
                    )
                    self.assertEqual(output, expected)

    def test_explicit_output_path_is_preserved(self):
        args = self.resolve(
            "inference", "--num_players", "6", "--output_dir", "my-results"
        )
        self.assertEqual(args["output_dir"], "my-results")

    def test_demo_settings_are_preserved(self):
        for kind, filename, output in (
            ("train", "train_demo.yaml", "outputs/train_demo"),
            ("inference", "inference_demo.yaml", "outputs/inference/4players"),
        ):
            args = self.resolve(kind, "--config", str(ROOT / "configs" / filename))
            self.assertEqual(
                (args["num_players"], args["num_groups"], args["num_hub_tokens"]),
                (4, 2, 16),
            )
            self.assertEqual(
                args["output_path" if kind == "train" else "output_dir"], output
            )
            self.assertEqual(args["dataset_path"], "asset/examples/4players")
        args = self.resolve(
            "inference",
            "--config",
            str(ROOT / "configs" / "inference_demo.yaml"),
            "--num_players",
            "6",
        )
        self.assertEqual(
            (args["num_groups"], args["num_hub_tokens"], args["dataset_path"]),
            (3, 21, "asset/examples/6players"),
        )
        self.assertEqual(args["output_dir"], "outputs/inference/4infer6")

    def test_prepared_inputs_are_separated_by_inference_setting(self):
        for players, setting in ((4, "4players"), (6, "4infer6"),
                                 (8, "4infer8"), (10, "4infer10")):
            with self.subTest(players=players):
                args = self.resolve("inference", "--num_players", str(players))
                self.assertEqual(args["inference_setting"], setting)
                self.assertEqual(args["prepared_inputs_dir"],
                                 f"outputs/prepared_inputs/{setting}")
        args = self.resolve("inference", "--prepared_inputs_dir", "custom-inputs")
        self.assertEqual(args["prepared_inputs_dir"], "custom-inputs")

    def test_group_and_hub_inputs_are_rejected(self):
        for kind in ("train", "inference"):
            for key in ("num_groups", "num_hub_tokens"):
                parser = train_args if kind == "train" else infer_args
                with self.subTest(kind=kind, key=key):
                    with patch.object(
                        sys, "argv", [f"{kind}.py", f"--{key}", "9", "--print-config"]
                    ):
                        with (
                            contextlib.redirect_stderr(io.StringIO()),
                            self.assertRaises(SystemExit) as error,
                        ):
                            parser()
                    self.assertEqual(error.exception.code, 2)
                    with tempfile.TemporaryDirectory() as tmp:
                        config = Path(tmp) / "config.yaml"
                        config.write_text(yaml.safe_dump({key: 9}))
                        with patch.object(
                            sys,
                            "argv",
                            [f"{kind}.py", "--config", str(config), "--print-config"],
                        ):
                            stderr = io.StringIO()
                            with (
                                contextlib.redirect_stderr(stderr),
                                self.assertRaises(SystemExit) as error,
                            ):
                                parser()
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn("computed", stderr.getvalue())

    def test_invalid_player_counts_are_rejected(self):
        for value in (0, 1, -4, 4.5, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / "config.yaml"
                config.write_text(yaml.safe_dump({"num_players": value}))
                with self.assertRaisesRegex(ValueError, "integer >= 2"):
                    self.resolve("train", "--config", str(config))

    def test_checkpoint_budget_and_resume_window(self):
        for players, groups, total in ((4, 2, 16), (6, 3, 21), (10, 4, 26)):
            with self.subTest(players=players), tempfile.TemporaryDirectory() as tmp:
                version = Path(tmp)
                (version / "checkpoints").mkdir()
                hp = version / "hparams.yaml"
                saved = dict(
                    num_players=str(players),
                    num_groups=str(groups),
                    num_hub_tokens=str(total),
                    local_attn_size=7,
                    local_attn_size_after=21,
                    local_attn_switch_step=10000,
                    train_step_offset=10000,
                )
                hp.write_text(yaml.safe_dump(saved))
                options = (
                    "--num_players",
                    str(players),
                    "--checkpoint",
                    str(version / "checkpoints/step15000.ckpt"),
                )
                args = self.resolve("inference", *options)
                self.assertEqual(args["local_attn_size"], 21)
                for key in ("num_groups", "num_hub_tokens"):
                    hp.write_text(yaml.safe_dump({**saved, key: 99}))
                    with self.assertRaisesRegex(ValueError, f"derived {key}"):
                        self.resolve("inference", *options)

    def test_budget_uses_restored_checkpoint_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            version = Path(tmp)
            config = version / "config.yaml"
            config.write_text("{}\n")
            (version / "checkpoints").mkdir()
            (version / "hparams.yaml").write_text(
                yaml.safe_dump(
                    dict(
                        num_players=6,
                        num_local_hub_tokens=7,
                        num_global_hub_tokens=5,
                        num_groups=3,
                        num_hub_tokens=26,
                    )
                )
            )
            args = self.resolve(
                "inference",
                "--config",
                str(config),
                "--checkpoint",
                str(version / "checkpoints/step15000.ckpt"),
            )
            self.assertEqual(
                (args["num_players"], args["num_groups"], args["num_hub_tokens"]),
                (6, 3, 26),
            )


if __name__ == "__main__":
    unittest.main()
