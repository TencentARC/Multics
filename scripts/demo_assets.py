"""Verify bundled example files against their SHA-256 manifests."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "asset" / "examples"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Verify all bundled inputs (default)"
    )
    parser.parse_args()
    folders = sorted(path.parent for path in ROOT.glob("*/manifest.json"))
    if not folders:
        parser.error("No bundled example manifests found")
    manifests = {
        folder: json.loads((folder / "manifest.json").read_text())
        for folder in folders
    }
    failures = []
    for folder, manifest in manifests.items():
        for name, key in (
            ("metadata.csv", "metadata_sha256"),
            ("inputs.csv", "inputs_sha256"),
        ):
            path = folder / name
            if manifest.get(key) and (
                not path.is_file() or sha256(path) != manifest[key]
            ):
                failures.append(f"{folder.name}/{name}")
        for case in manifest["cases"]:
            for record in case["files"]:
                path = folder / record["path"]
                if not path.is_file() or sha256(path) != record["sha256"]:
                    failures.append(f"{folder.name}/{record['path']}")
    if failures:
        parser.error("Missing or changed example inputs:\n" + "\n".join(failures))
    for folder, manifest in manifests.items():
        cases = ", ".join(case["example_id"] for case in manifest["cases"])
        print(f"Verified {folder.name}: {cases}")


if __name__ == "__main__":
    main()
