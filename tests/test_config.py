"""Public example identity and safe checkpoint configuration regressions."""

from types import SimpleNamespace
import pandas as pd
import pytest
from multics.config import select_case_rows, case_output_id, _scalar
from multics.inference import _existing_batch_indices


def test_demo_identity_and_skip_are_independent_of_csv_order(tmp_path):
    metadata = pd.DataFrame({"case_id": [74, 103, 128, 84, 116]})
    args = SimpleNamespace(
        cases=[84, 74],
        no_save_separate_players=False,
        no_save_grid=False,
        num_players=4,
    )
    assert select_case_rows(args, metadata, list(range(5))) == [0, 3]
    output_ids = [case_output_id(metadata, i) for i in range(5)]
    files = {"video74_grid.mp4"} | {f"video74_player{i:02}.mp4" for i in range(1, 5)}
    assert _existing_batch_indices(5, args, files, output_ids) == {0}
    assert _existing_batch_indices(5, args, {"video0_grid.mp4"}, output_ids) == set()


def test_requested_cases_cannot_silently_disappear():
    metadata = pd.DataFrame({"case_id": [74, 103]})
    with pytest.raises(ValueError, match="excluded"):
        select_case_rows(SimpleNamespace(cases=[103]), metadata, [0])
    with pytest.raises(ValueError, match="Unknown"):
        select_case_rows(SimpleNamespace(cases=[999]), metadata, [0, 1])


def test_consecutive_windows_use_source_row_identity():
    metadata = pd.DataFrame({"case_id": [74, 103, 128, 84]})
    start_rows = [0, 2]
    samples = metadata.iloc[start_rows].reset_index(drop=True)
    assert select_case_rows(SimpleNamespace(cases=[128]), samples, [0, 1]) == [1]
    assert case_output_id(metadata, start_rows[1]) == 128


def test_stringified_lightning_hparams():
    assert _scalar("False") is False
    assert _scalar("None") is None
    assert _scalar("10000") == 10000
    assert _scalar("1.0") == 1.0
    assert _scalar("shared") == "shared"


@pytest.mark.parametrize(
    "key,value",
    [
        ("attention_mode", "full_attn"),
        ("loss_type", "bidirectional"),
        ("use_simplex_rope", False),
    ],
)
def test_yaml_cannot_silently_enable_retired_modes(tmp_path, monkeypatch, key, value):
    import sys
    import yaml
    from multics.training_args import parse_args

    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({key: value}))
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--config", str(config), "--print-config"]
    )
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_checkpoint_window_includes_resume_offset(tmp_path, monkeypatch):
    import sys
    import yaml
    from multics.inference_args import parse_args

    version = tmp_path / "version_47"
    checkpoints = version / "checkpoints"
    checkpoints.mkdir(parents=True)
    (version / "hparams.yaml").write_text(
        yaml.safe_dump(
            {
                "attention_mode": "hier_hub",
                "num_players": "4",
                "local_attn_size": "7",
                "local_attn_size_after": "21",
                "local_attn_switch_step": "10000",
                "train_step_offset": "10000",
            }
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset_path": "data",
                "metadata_file_name": "metadata.csv",
                "wan_model_root": "models",
                "num_players": 4,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoints / "step15000.ckpt"),
        ],
    )
    assert parse_args().local_attn_size == 21
    config.write_text(config.read_text().replace("num_players: 4", "num_players: 6"))
    args = parse_args()
    assert (args.num_players, args.num_groups, args.num_hub_tokens) == (6, 3, 21)
    hparams = yaml.safe_load((version / "hparams.yaml").read_text())
    hparams["hier_local_hub_mode"] = "per_group"
    (version / "hparams.yaml").write_text(yaml.safe_dump(hparams))
    with pytest.raises(ValueError, match="per-group local hubs"):
        parse_args()
