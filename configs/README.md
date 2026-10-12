# Configurations

Use [train.yaml](train.yaml) for training and [inference.yaml](inference.yaml) for inference. Each task has one shared configuration. Set `num_players` in YAML or override it on the CLI.

## Automatic communication budget

The number of group slots is `num_groups = ceil(log2(num_players))`. The total hub count is `num_groups * num_local_hub_tokens + num_global_hub_tokens`.

`num_groups` and `num_hub_tokens` are computed internally and included in resolved settings and saved training hyperparameters. Do not supply them in YAML or on the CLI. Checkpoint values are checked against the computed result; mismatched architecture is rejected.

The defaults retain a Simplex pool of 10, 5 local hubs per group, 6 global hubs, shared local hub parameters, membership labels, camera conditioning, 165 RGB frames, and 3 latent frames per block. Input player streams and membership labels must agree with the selected player count and computed group slots. The selected player count must fit the checkpoint's Simplex pool.

Training uses learning rate `3e-5`, text/camera dropout `0.1`, random vertex sampling, and an attention window of 7 that switches to 21 at training step 10000. Inference defaults to 50 sampling steps, guidance scale 5, seed 0, and window 21; a checkpoint named `stepN.ckpt` restores its scheduled attention window from the hyperparameters. A released name such as `model.ckpt` keeps the inference configuration’s window.

## Paths

- Set `dataset_path` and `metadata_file_name` in YAML or on the CLI. Inference accepts either a raw-input CSV, with `image_path`/`ticks_path` columns, or a precomputed one; see the [example inputs](../asset/examples/README.md). Inputs prepared from a raw-input CSV are written to `prepared_inputs_dir` (default `outputs/prepared_inputs/{inference_setting}`; each run gets its own `run-*` subdirectory).
- Inference requires `ckpt_path` in YAML or `--checkpoint` on the CLI. `hparams.yaml` is discovered next to the checkpoint's `checkpoints/` directory; use `--hparams` for another location.
- Base model paths default to `models/Wan-AI/` and can be edited in YAML.
- The `{num_players}` placeholder selects the dataset directory. `{inference_setting}` selects `4players`, `4infer6`, `4infer8`, or `4infer10` for the released 4-player model and separates output and prepared-input directories. A supplied checkpoint's saved training player count is used for other models. A literal path supplied on the CLI overrides the template.
- Player identities are derived from the input metadata. For a custom mapping, set `group_label_player_ids` to match the membership file's player columns.
- `grouping_cache_dir` defaults to `outputs/grouping_cache`. Missing group labels are generated automatically from the CSV's `grouping_state_path`. Automatic generation requires compact CSV player order and one of the supported 4/6/8/10-player profiles; the label-generation budget window stays 7 even when the model window is 21.

## Bundled examples

[inference_demo.yaml](inference_demo.yaml) reads the raw inputs in `asset/examples/{num_players}players/inputs.csv` and writes to `outputs/inference/{inference_setting}/`. It selects the four-player examples by default. Use `--num_players 6 --cases 1 2 3`, `--num_players 8 --cases 1 2`, or `--num_players 10 --cases 1 2` for the other sets. All bundled examples use the 4-player checkpoint and its `hparams.yaml`.

[train_demo.yaml](train_demo.yaml) uses the precomputed four-player inputs in `asset/examples/4players/metadata.csv` for five training steps on one GPU, with no data-loader workers. Both example configurations derive their group and hub counts automatically.

For other player counts, supply the corresponding data. Training reads precomputed inputs, which the 6-, 8-, and 10-player examples do not include.

CLI values override YAML defaults. Checkpoint architecture conflicts are rejected; use `--help` on either entry point to list all options.

## Inspect settings

```bash
python train.py --config configs/train.yaml --num_players 6 --print-config
python inference.py --config configs/inference.yaml --num_players 10 --print-config
```

Both training and inference default to four players. Other player counts use the same formula, with a minimum of two players.
