"""
Download and optionally extract batik dataset into data/raw.

Example:
    python scripts/download_dataset.py --url "https://example.com/batik.zip" --out_dir data/raw --extract
"""

import argparse
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request


def filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name or "dataset_download"


def download(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename_from_url(url)

    req = Request(url, headers={"User-Agent": "batik-cnn-classifier/1.0"})
    with urlopen(req) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)

    return dest


def extract_archive(path: Path, extract_dir: Path) -> Path:
    target = extract_dir / path.stem
    target.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
        return target

    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(target)
        return target

    raise ValueError(f"Unsupported archive format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Download batik dataset")
    parser.add_argument("--url", required=True, help="Dataset download URL")
    parser.add_argument("--out_dir", default="data/raw", help="Output raw data directory")
    parser.add_argument("--extract", action="store_true", help="Extract .zip/.tar archive after download")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    downloaded = download(args.url, out_dir)
    print(f"Downloaded: {downloaded}")

    if args.extract:
        extracted = extract_archive(downloaded, out_dir)
        print(f"Extracted to: {extracted}")
        print("Next: inspect folder names, then run scripts/prepare_dataset.py --raw_dir <extracted_folder>")


if __name__ == "__main__":
    main()
