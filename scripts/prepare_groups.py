"""Prepare missing group labels, or recompute and compare with existing labels."""

import argparse
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from multics.grouping.prepare import DEFAULT_CACHE, ensure_group_path
from multics.grouping.state import state_from_parquet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--metadata_file_name", default="metadata.csv")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--cache_dir", type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument(
        "--rounds_root",
        type=Path,
        help="Optional local raw rounds root for preparing tick states",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        help="Write a new CSV with resolved group/state paths",
    )
    parser.add_argument(
        "--verify_existing",
        action="store_true",
        help="Recompute independently and compare all 24 fields; never overwrite originals",
    )
    args = parser.parse_args()
    base = args.dataset_path.resolve()
    source = base / args.metadata_file_name
    if args.output_csv and args.output_csv.resolve() == source.resolve():
        parser.error("--output_csv must differ from the source CSV")
    table = pd.read_csv(source)
    for index, series in table.iterrows():
        row = series.to_dict()
        original = row.get("group_path")
        original_path = base / str(original) if pd.notna(original) else None
        if args.verify_existing:
            if original_path is None or not original_path.is_file():
                raise FileNotFoundError(
                    f"Row {index} has no existing labels to compare"
                )
            row["group_path"] = None
        missing = original_path is None or not original_path.is_file()
        if (missing or args.verify_existing) and args.rounds_root:
            # Serialize the exact aligned/cropped state; no RGB or camera interpolation.
            state = state_from_parquet(args.rounds_root, row, args.num_players)
            buffer = io.BytesIO()
            state.save(buffer)
            raw = buffer.getvalue()
            folder = args.cache_dir.resolve() / "states"
            folder.mkdir(parents=True, exist_ok=True)
            state_path = folder / (hashlib.sha256(raw).hexdigest() + ".npz")
            fd, temporary = tempfile.mkstemp(dir=folder, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(raw)
                os.replace(temporary, state_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            row["grouping_state_path"] = str(state_path)
            table.loc[index, "grouping_state_path"] = str(state_path)
        output = ensure_group_path(
            row, base, num_players=args.num_players, cache_dir=args.cache_dir
        )
        if args.verify_existing:
            expected = torch.load(original_path, map_location="cpu", weights_only=True)
            actual = torch.load(output, map_location="cpu", weights_only=True)
            if set(expected) != set(actual):
                raise ValueError(f"Row {index}: label keys differ")
            for name in expected:
                a, b = expected[name], actual[name]
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                    raise ValueError(f"Row {index}: label field {name} differs")
            print(f"Case {row.get('case_id', index)}: all 24 fields exactly equal")
        else:
            print(f"Case {row.get('case_id', index)}: {output}")
        table.loc[index, "group_path"] = str(Path(output).resolve())
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
        print(f"Prepared CSV: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
