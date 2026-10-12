# Dynamic grouping

Training and inference reuse an existing `group_path`. When the column is absent, a cell is empty, or the file is missing, they compute groups from the player poses in `grouping_state_path`. A broken or incompatible existing label file raises an error rather than being replaced.

## Included examples

The five 4-player examples contain both the reference labels and their lossless trajectory inputs. To recompute and compare all 24 fields:

```bash
python scripts/prepare_groups.py --dataset_path asset/examples/4players --verify_existing
```

This leaves the reference labels unchanged and writes the recomputed labels to `outputs/grouping_cache/`. Removing the `group_path` column from a **copy** of the example CSV exercises automatic generation during normal training or inference; retain `grouping_state_path` and use the original example directory as `dataset_path`.

## Prepare your own inputs

Use one CSV row per 165-frame clip. `video_path1..N` must identify the same round and preserve the player order used by the latents/actions/cameras. Supply `start_frame`, `video1_frame..videoN_frame` and either existing group labels or a grouping state file.

For tick parquet inputs, run:

```bash
python scripts/prepare_groups.py \
  --dataset_path /path/to/preprocessed_data \
  --metadata_file_name train.csv \
  --rounds_root /path/to/raw/rounds \
  --num_players 6 \
  --output_csv /path/to/prepared/train.grouped.csv
```

The script resolves raw files as `match_id=.../map_name=.../round=XX/player=XX/ticks.parquet`. It reads `tick, x, y, z, yaw, pitch, is_alive`, preserves missing tick records, aligns the selected players, and saves compact states under the grouping cache. The new CSV retains the other paths relative to the original `dataset_path`; use that same dataset root when training. Existing label files are reused. Add `--verify_existing` to independently rebuild and compare them.

Inference from a raw-input CSV such as `asset/examples/4players/inputs.csv` builds the same state from each player's `ticks_path` column, so no separate preparation step is needed; see the [example inputs](../asset/examples/README.md).

## State format

`grouping_state.npz` contains non-object NumPy arrays and loads with `allow_pickle=False`:

| Field | Type / meaning |
| --- | --- |
| `ticks` | int64 `[T]`, continuous 64Hz tick grid, including missing records |
| `pos` | float32 `[T,N,3]`, original world-space foot positions |
| `yaw`, `pitch` | float32 `[T,N]`, degrees; positive pitch points downward |
| `alive`, `present` | bool `[T,N]`, distinct life/record-presence flags |
| `player_slots` | int64 `[N]`, physical slots in CSV column order |
| `round_tick_start`, `round_tick_end` | original inclusive round bounds before cropping |
| `match_id`, `map_name`, `round_no` | source identity |
| `clip_start_frame`, `clip_num_frames` | clip range on the original 32fps video timeline |
| `format_version` | integer scalar, currently 1 |

Keep ticks from `max(0, start_frame-48)` through the entire target clip. Align and fill original round states before cropping; retain the original round tick origin. Do not interpolate the 42 latent camera matrices to approximate this input.

## Fixed generation profile

Automatic generation currently supports 4/6/8/10 players, 165 RGB frames, 5 local hub tokens per group, 6 global hub tokens, direct fraction1, and 3 latent frames per block. Group slots are derived as `ceil(log2(N))`. Other player counts can still use compatible precomputed membership labels.

At every 48-frame boundary, relevance uses the observed current state plus fast12/slow48-frame histories, half-lives4/12, scene weights0.75/0.25, and separate scene/interaction EMA with alpha0.60. Pose-only disables firing scores while retaining the wider interaction-validity distance gate. Selection uses threshold0.30, coverage-first bounded clique search, anchor constraints, and idle-hub recovery without an extra score floor. Group slots are matched across periods before mapping to latent frames.

The budget is computed with 42 latent frames, block3, **budget window7** and 220 tokens per player. This window belongs to the label-generation protocol; the model's training/inference attention window may differ. Keep these settings separate to reproduce existing labels.

Relevance does not read poses after its current update boundary. Eligibility still checks presence over the entire clip, as in the original dataset protocol. This preparation step assumes the clip's trajectory input is available; it is not an online estimator of unknown future cameras.

## Cache and failures

`--grouping_cache_dir` changes the cache location for training or inference; the preparation script calls this option `--cache_dir`. Cache identity includes the fixed algorithm version, state content and ordered clip identity. Writers use a per-key lock and atomic replacement. Only selected inference samples and their constituent source rows are prepared; the source CSV and reference label files are not changed.

Missing history, reordered player identities, invalid tensors, ineligible selected players and incompatible generation profiles fail before data workers start. Generation makes no draws from the training random-number streams.
