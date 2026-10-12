"""Configuration and stable demo identities; no model code is imported here."""

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DERIVED_FIELDS = {"num_groups", "num_hub_tokens"}
ARCHITECTURE = {
    "simplex_pool_size",
    "simplex_gamma",
    "num_hub_tokens",
    "num_groups",
    "num_local_hub_tokens",
    "num_global_hub_tokens",
    "hier_local_hub_mode",
    "hier_no_local_hub",
    "hier_direct_schedule",
    "hier_direct_frac",
    "hier_direct_layers",
    "hier_group_format",
    "hier_group_mode",
    "hier_group_seed",
    "hier_random_preserve_sizes",
    "group_label_player_ids",
    "use_camera_pose",
    "camera_dim",
    "camera_dropout_p",
    "num_frame_per_block",
}


def read_config(path):
    import yaml

    with open(path, encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return result


def _scalar(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            if value in ("True", "False"):
                return value == "True"
            if value in ("None", ""):
                return None
    return value


def derive_hier_budget(args, hparams=None):
    """Resolve M = ceil(log2(N)) and the resulting hub budget."""
    n = args.num_players
    if type(n) is not int or n < 2:
        raise ValueError("num_players must be an integer >= 2")
    for key in ("num_local_hub_tokens", "num_global_hub_tokens"):
        value = getattr(args, key)
        if type(value) is not int or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    # Exact integer form of ceil(log2(N)), including powers of two.
    args.num_groups = (n - 1).bit_length()
    args.num_hub_tokens = (
        args.num_groups * args.num_local_hub_tokens + args.num_global_hub_tokens
    )
    for key in DERIVED_FIELDS:
        if hparams is not None and key in hparams:
            if hparams[key] != getattr(args, key):
                raise ValueError(
                    f"Checkpoint/config mismatch: derived {key}={getattr(args, key)}, "
                    f"checkpoint={hparams[key]!r}"
                )


def resolve_args(parser, kind):
    default_config = ROOT / "configs" / f"{kind}.yaml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--grouping_cache_dir", default="outputs/grouping_cache")
    if kind == "train":
        parser.add_argument("--seed", type=int, default=0)
    else:
        parser.add_argument("--checkpoint", dest="ckpt_path", default=argparse.SUPPRESS)
        parser.add_argument("--hparams", default=None)
        parser.add_argument("--case", "--cases", dest="cases", type=int, nargs="+")
        parser.add_argument(
            "--prepared_inputs_dir",
            default="outputs/prepared_inputs/{inference_setting}",
            help="Where inputs prepared from a raw-input CSV are written",
        )
    # Read config independently so required CLI paths may come from YAML.
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=str(default_config))
    config_path = probe.parse_known_args()[0].config
    config = read_config(config_path)
    environment = config.pop("environment", {})
    for key, value in environment.items():
        if not key.startswith(("WAN_", "TORCHINDUCTOR_", "PYTORCH_")):
            raise ValueError(f"Unsupported runtime environment setting: {key}")
        os.environ.setdefault(key, str(value))
    actions = {a.dest: a for a in parser._actions}
    supplied_derived = DERIVED_FIELDS & config.keys()
    if supplied_derived:
        parser.error(
            f"{sorted(supplied_derived)} are computed from num_players and the "
            "local/global hub sizes; remove them from the config"
        )
    unknown = set(config) - actions.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    parser.set_defaults(**config)
    # Required model/data paths are checked below, after --print-config.
    for action in parser._actions:
        action.required = False
    args = parser.parse_args()
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if (
            action.choices is not None
            and value is not None
            and value not in action.choices
        ):
            parser.error(
                f"{action.dest} must be one of {list(action.choices)}, got {value!r}"
            )
    for name, expected in {
        "training_stage": "stage2",
        "loss_type": "diffusion_forcing",
        "attention_mode": "hier_hub",
        "use_simplex_rope": True,
    }.items():
        if hasattr(args, name) and getattr(args, name) != expected:
            parser.error(f"{name} is fixed to {expected!r} in Multics")
    hparams = None
    if kind == "inference" and args.ckpt_path:
        hp = (
            Path(args.hparams)
            if args.hparams
            else Path(args.ckpt_path).parent.parent / "hparams.yaml"
        )
        if hp.exists():
            hparams = {k: _scalar(v) for k, v in read_config(hp).items()}
            if hparams.get("attention_mode", "hier_hub") != "hier_hub":
                raise ValueError("Only hier checkpoints are supported")
            trained_players = hparams.get("num_players")
            if trained_players is not None:
                explicit = "num_players" in config or any(
                    x.split("=")[0] in actions["num_players"].option_strings
                    for x in sys.argv[1:]
                )
                if not explicit:
                    args.num_players = trained_players
                elif args.num_players != trained_players:
                    # Shared local hubs have no per-group weights, so the
                    # group and hub counts can follow the new player count.
                    if hparams.get("hier_local_hub_mode", "shared") != "shared":
                        raise ValueError(
                            f"Checkpoint trained with {trained_players} players uses "
                            "per-group local hubs and cannot run "
                            f"{args.num_players} players"
                        )
                    hparams = {
                        k: v for k, v in hparams.items() if k not in DERIVED_FIELDS
                    }
            for key in ARCHITECTURE:
                if key in hparams and key in actions:
                    value = hparams[key]
                    # Reject explicit conflicts, including explicit YAML architecture.
                    explicit = key in config or any(
                        x.split("=")[0] in actions[key].option_strings
                        for x in sys.argv[1:]
                    )
                    if explicit and getattr(args, key) != value:
                        raise ValueError(
                            f"Checkpoint/config mismatch: {key}={getattr(args, key)!r}, checkpoint={value!r}"
                        )
                    setattr(args, key, value)
            # stepN.ckpt names select the scheduled window. A released name such
            # as model.ckpt keeps the window from the inference config.
            step = re.fullmatch(r"step(\d+)\.ckpt", Path(args.ckpt_path).name)
            if step:
                window = int(hparams.get("local_attn_size", args.local_attn_size))
                switch = hparams.get("local_attn_switch_step")
                after = hparams.get("local_attn_size_after")
                if switch is not None and after is not None:
                    effective = int(step[1]) + int(
                        hparams.get("train_step_offset", 0)
                    )
                    if effective >= int(switch):
                        window = int(after)
                args.local_attn_size = window
        elif args.hparams:
            raise FileNotFoundError(hp)
    derive_hier_budget(args, hparams)
    replacements = {"num_players": str(args.num_players)}
    path_keys = ["output_path" if kind == "train" else "output_dir", "dataset_path"]
    if kind == "inference":
        # Public examples use the four-player checkpoint. A checkpoint's saved
        # training player count takes precedence when another model is supplied.
        trained_players = int((hparams or {}).get("num_players", 4))
        args.inference_setting = (
            f"{trained_players}players"
            if trained_players == args.num_players
            else f"{trained_players}infer{args.num_players}"
        )
        replacements["inference_setting"] = args.inference_setting
        path_keys.append("prepared_inputs_dir")
    for key in path_keys:
        value = getattr(args, key)
        if isinstance(value, str):
            for name, replacement in replacements.items():
                value = value.replace("{" + name + "}", replacement)
            setattr(args, key, value)
    if args.print_config:
        print(json.dumps(vars(args), indent=2))
        raise SystemExit(0)
    for key in ("dataset_path", "metadata_file_name") + (
        ("ckpt_path", "wan_model_root")
        if kind == "inference"
        else ("dit_path", "vae_path", "text_encoder_path")
    ):
        if not getattr(args, key, None):
            parser.error(f"{key} must be supplied in config or CLI")
    return args


def select_case_rows(args, metadata, planned_rows):
    if not args.cases:
        return planned_rows
    if "case_id" not in metadata:
        ids = list(range(len(metadata)))
    else:
        ids = [int(x) for x in metadata.case_id]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case_id in metadata")
    missing = set(args.cases) - set(ids)
    if missing:
        raise ValueError(f"Unknown cases: {sorted(missing)}")
    selected = set(args.cases)
    excluded = selected - {ids[row] for row in planned_rows}
    if excluded:
        raise ValueError(
            f"Requested cases excluded by the row range or max_batches: {sorted(excluded)}"
        )
    return [row for row in planned_rows if ids[row] in selected]


def case_output_id(metadata, row):
    return int(metadata.iloc[row].case_id) if "case_id" in metadata else row
