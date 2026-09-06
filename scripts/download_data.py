#!/usr/bin/env python3
"""Download the two public benchmark datasets used by HAF-Net."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TAIWAN_URL = "https://archive.ics.uci.edu/static/public/572/taiwanese%2Bbankruptcy%2Bprediction.zip"
AMERICAN_URL = "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/american_bankruptcy_dataset.csv"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "HAF-Net/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def download_taiwan(root: Path) -> Path:
    """Download the UCI archive and extract its data.csv table."""
    destination = root / "data" / "raw" / "taiwan" / "data.csv"
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "taiwan.zip"
        _download(TAIWAN_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            candidates = [name for name in archive.namelist() if Path(name).name.lower() == "data.csv"]
            if not candidates:
                raise RuntimeError("The Taiwan archive does not contain data.csv.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(candidates[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    return destination


def download_american(root: Path) -> Path:
    """Download the public American firm-year CSV table."""
    destination = root / "data" / "raw" / "american" / "american_bankruptcy_dataset.csv"
    _download(AMERICAN_URL, destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HAF-Net benchmark datasets from their public sources.")
    parser.add_argument("--dataset", choices=["taiwan", "american", "all"], default="all")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing the data directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.root.resolve()
    paths: list[Path] = []
    if args.dataset in {"taiwan", "all"}:
        paths.append(download_taiwan(root))
    if args.dataset in {"american", "all"}:
        paths.append(download_american(root))
    for path in paths:
        print(path)
