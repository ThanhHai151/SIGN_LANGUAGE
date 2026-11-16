"""Download the public skeleton archives referenced by the historical report.

Run from the repository root: python -m backend.training.fetch_sstcn_data --extract
Only the Python standard library is needed. Large datasets stay outside git.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "data" / "sstcn"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive, destination):
    """Reject traversal/symlinks and limit uncompressed size before extracting."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        if sum(m.file_size for m in members) > 12 * 1024**3:
            raise ValueError("Archive exceeds the 12 GiB extraction limit")
        for m in members:
            name = PurePosixPath(m.filename.replace("\\", "/"))
            target = (destination / str(name)).resolve()
            if (name.is_absolute() or ".." in name.parts or ":" in str(name)
                    or not target.is_relative_to(destination)
                    or stat.S_ISLNK(m.external_attr >> 16)):
                raise ValueError(f"Unsafe archive member: {m.filename}")
        for m in members:
            if "__MACOSX" in PurePosixPath(m.filename).parts or m.is_dir():
                continue
            target = destination / m.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == m.file_size:
                # Compare CRC as well, so an interrupted or changed file is repaired.
                import zlib
                with target.open("rb") as f:
                    crc = 0
                    for b in iter(lambda: f.read(1024 * 1024), b""):
                        crc = zlib.crc32(b, crc)
                if crc == m.CRC:
                    continue
            tmp = target.with_name(target.name + ".part")
            with z.open(m) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(target)


def download(url, tmp):
    offset = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": "VSL-SSTCN-dataset-recovery/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as response:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("Download returned HTML instead of an archive; check access/sharing/quota")
        resumed = offset and response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        if response.status == 206 and not resumed:
            raise RuntimeError("Unexpected Content-Range; refusing to combine different file ranges")
        with tmp.open("ab" if resumed else "wb") as dst:
            shutil.copyfileobj(response, dst, length=1024 * 1024)


def fetch(entry, directory, source="auto"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / entry["name"]
    expected = entry.get("sha256")
    if target.exists() and zipfile.is_zipfile(target):
        digest = sha256(target)
        if expected and digest != expected:
            raise ValueError(f"Checksum mismatch for existing {target}")
        return target, digest
    query = urllib.parse.urlencode({"id": entry["id"], "export": "download", "confirm": "t"})
    drive_url = "https://drive.usercontent.google.com/download?" + query
    github_url = entry.get("github_release_url")
    if source not in {"auto", "github", "drive"}:
        raise ValueError(f"Unknown dataset source: {source}")
    if source == "github" and not github_url:
        raise ValueError("No GitHub Release URL is configured for this archive")
    urls = ([github_url, drive_url] if github_url else [drive_url]) if source == "auto" else [github_url if source == "github" else drive_url]
    tmp = target.with_name(target.name + ".part")
    for i, url in enumerate(urls):
        print(f"Downloading {target.name} from {'GitHub Releases' if url == github_url else 'Google Drive'}", flush=True)
        try:
            download(url, tmp)
            break
        except (urllib.error.URLError, TimeoutError, RuntimeError):
            if i == len(urls) - 1:
                raise
            print("GitHub download unavailable; retrying the original Google Drive source", flush=True)
    if not zipfile.is_zipfile(tmp):
        raise ValueError(f"Invalid ZIP download: {tmp}")
    digest = sha256(tmp)
    if expected and digest != expected:
        raise ValueError(f"Checksum mismatch for {tmp}")
    tmp.replace(target)
    return target, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=ROOT / "sources.json")
    p.add_argument("--output", type=Path, default=ROOT)
    p.add_argument("--only", nargs="+", help="Archive filenames, e.g. test_set.zip")
    p.add_argument("--extract", action="store_true")
    p.add_argument("--source", choices=["auto", "github", "drive"], default="auto",
                   help="auto prefers GitHub Releases and falls back to the original Drive source")
    args = p.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    known = {e["name"] for e in sources["archives"]}
    if args.only and set(args.only) - known:
        p.error(f"Unknown archives: {set(args.only) - known}")
    receipt_path = args.output / "download_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for e in sources["archives"]:
        if args.only and e["name"] not in args.only:
            continue
        archive, digest = fetch(e, args.output / "archives", args.source)
        entry = {**e, "sha256": digest, "bytes": archive.stat().st_size,
                 "retrieved_at": datetime.now(timezone.utc).isoformat(), "source_policy": args.source}
        if args.extract:
            destination = args.output / "raw" / archive.stem
            safe_extract(archive, destination)
            entry["extracted_to"] = str(destination)
        receipt[e["name"]] = entry
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(f"Ready: {e['name']} ({entry['bytes']:,} bytes, sha256={digest})", flush=True)


if __name__ == "__main__":
    main()
