#!/usr/bin/env python3
"""Download the two public benchmark datasets used by HAF-Net."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

TAIWAN_URL = "https://archive.ics.uci.edu/static/public/572/taiwanese%2Bbankruptcy%2Bprediction.zip"
TAIWAN_SHA256 = "67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8"
AMERICAN_URL = "https://raw.githubusercontent.com/sowide/bankruptcy_dataset/main/american_bankruptcy_dataset.csv"
AMERICAN_SHA256 = "d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "HAF-Net/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            extracted = Path(temporary_directory) / "data.csv"
            with archive.open(candidates[0]) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        if _sha256(extracted) != TAIWAN_SHA256:
            raise ValueError("The Taiwan dataset checksum does not match the pinned release.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(extracted, destination)
    return destination


def download_american(root: Path) -> Path:
    """Download the pinned American firm-year CSV table."""
    destination = root / "data" / "raw" / "american" / "american_bankruptcy_dataset.csv"
    with tempfile.TemporaryDirectory() as temporary_directory:
        downloaded = Path(temporary_directory) / destination.name
        _download(AMERICAN_URL, downloaded)
        if _sha256(downloaded) != AMERICAN_SHA256:
            raise ValueError("The American dataset checksum does not match the pinned release.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(downloaded, destination)
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
