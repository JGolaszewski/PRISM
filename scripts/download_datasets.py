"""Download the complete MERL and RGL BRDF datasets into dataset/raw/."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "dataset" / "raw"
MERL_INDEX = "https://people.csail.mit.edu/wojciech/BRDFDatabase/brdfs/"
MERL_ROOT = "https://people.csail.mit.edu/wojciech/BRDFDatabase/"
RGL_CATALOG = "https://rgl.epfl.ch/materials"
USER_AGENT = "PRISM-workspace-dataset-downloader/1.0"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Already present: {destination.relative_to(REPO_ROOT)}")
        return

    partial = destination.with_name(destination.name + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if partial.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty file from {url}")
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Downloaded: {destination.relative_to(REPO_ROOT)}")


def merl_files() -> list[tuple[str, str]]:
    parser = LinkParser()
    parser.feed(fetch_text(MERL_INDEX))
    files = sorted(
        (urljoin(MERL_INDEX, href), Path(urlparse(href).path).name)
        for href in parser.hrefs
        if urlparse(href).path.lower().endswith(".binary")
    )
    if len(files) != 100:
        raise RuntimeError(f"Expected 100 MERL materials, found {len(files)} at {MERL_INDEX}")
    return files


def rgl_files() -> list[tuple[str, str]]:
    page = fetch_text(RGL_CATALOG)
    # The catalog embeds direct CloudFront URLs for both spectral and RGB files.
    urls = sorted(set(re.findall(r"https://[^\s\"'<>]+\.bsdf", page)))
    files = []
    for url in urls:
        name = Path(urlparse(url).path).name
        if name.endswith(("_spec.bsdf", "_rgb.bsdf")):
            files.append((url, name))
    if not files:
        raise RuntimeError(f"No RGL .bsdf files found in catalog at {RGL_CATALOG}")
    if not any(name.endswith("_spec.bsdf") for _, name in files):
        raise RuntimeError("RGL catalog did not contain spectral (_spec.bsdf) files")
    if not any(name.endswith("_rgb.bsdf") for _, name in files):
        raise RuntimeError("RGL catalog did not contain RGB (_rgb.bsdf) files")
    return files


def download_merl() -> None:
    out_dir = RAW_DIR / "merl"
    for url, name in merl_files():
        download(url, out_dir / name)

    # Keep the dataset's reader instructions and research-use notice alongside it.
    for name in ("Readme.txt", "Copyright_Notice.txt"):
        download(urljoin(MERL_ROOT, name), out_dir / name)


def download_rgl() -> None:
    out_dir = RAW_DIR / "rgl"
    files = rgl_files()
    print(f"Found {len(files)} RGL files in the official catalog.")
    for url, name in files:
        download(url, out_dir / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "merl", "rgl"),
        default="all",
        help="Dataset to download (default: all).",
    )
    args = parser.parse_args()

    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        if args.dataset in ("all", "merl"):
            download_merl()
        if args.dataset in ("all", "rgl"):
            download_rgl()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Datasets are in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
