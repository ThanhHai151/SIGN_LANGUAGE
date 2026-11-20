"""Prepare the full VSL training corpus on a cloud runner, in restartable source steps."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

from backend.training.fetch_sstcn_data import safe_extract
from backend.training.ingest_videos import discover_videos, ingest_videos
from backend.training.ingest_voya_vsl import ingest_from_directory
from backend.training.prepare_unified_dataset import prepare_unified


VIDEO_SOURCES = {
    "visign": "visign",
    "qipedc": "qipedc",
    "qipedc-cropped": "qipedc_cropped",
    "vsl400": "vsl400",
}
VOYA_REPO = "Kateht/VOYA_VSL"
VISIGN_REPO = "namwu/ViSignLanguage-Video"


def download_voya_parts(destination):
    from huggingface_hub import HfApi, hf_hub_download

    destination.mkdir(parents=True, exist_ok=True)
    completion = destination / "parts.json"
    revision_file = destination / "revision.txt"
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    else:
        revision = HfApi().dataset_info(VOYA_REPO, token=False).sha
        revision_file.write_text(revision + "\n")
    remote = dict(repo_id=VOYA_REPO, repo_type="dataset", revision=revision, token=False)
    shards = sorted(
        f for f in HfApi().list_repo_files(**remote)
        if f.startswith("Merged/") and f.endswith(".npz")
    )
    if not shards:
        raise ValueError("No VOYA class archives found at the recorded revision.")

    with tempfile.TemporaryDirectory(dir=destination) as scratch:
        hf_hub_download(filename="labels.json", local_dir=scratch, **remote)
        for shard in shards:
            part = destination / Path(shard).stem
            summary_file = part / "summary.json"
            if summary_file.exists() and json.loads(summary_file.read_text())["ingested"]:
                continue
            archive = Path(hf_hub_download(filename=shard, local_dir=scratch, **remote))
            summary = ingest_from_directory(Path(scratch), part)
            if not summary["ingested"]:
                raise ValueError(f"No usable samples in {shard}. See {part / 'errors.json'}.")
            archive.unlink()
            print(f"{shard}: {summary['ingested']} samples, {summary['errors']} rejected", flush=True)
    completion.write_text(json.dumps([Path(shard).stem for shard in shards], indent=2))


def download_visign(destination):
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        VISIGN_REPO, repo_type="dataset", local_dir=destination, token=False,
        allow_patterns=["dataset.zip", "dataset_metadata.csv"],
    )
    safe_extract(destination / "dataset.zip", destination / "videos")
    return destination / "videos", destination / "dataset_metadata.csv"


def ingest_source(args):
    out = args.work_dir / "sources" / args.source
    if args.source == "voya":
        download_voya_parts(out)
        return

    root, metadata = args.input_dir, args.metadata
    if args.source == "visign" and root is None:
        root, metadata = download_visign(args.work_dir / "downloads" / "visign")
    if root is None or not root.is_dir():
        raise ValueError(f"Mount the {args.source} dataset and pass its extracted --input-dir.")
    if metadata is not None and not metadata.is_file():
        raise FileNotFoundError(metadata)
    if args.source == "vsl400":
        if metadata is None:
            raise ValueError("Provide VSL400 metadata CSV with path,label,signer_id,view,group.")
        with metadata.open(encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        expected = {"path", "label", "signer_id", "view", "group"}
        if not rows or any(not all(row.get(k) for k in expected) for row in rows):
            raise ValueError("VSL400 CSV must identify the signer, view and sync group for every clip.")
        discovered = discover_videos(root, "vsl400", metadata)
        if len({row["video_path"].resolve() for row in discovered}) != len(rows):
            raise ValueError("Each VSL400 metadata row must match a unique video under --input-dir.")

    summary = ingest_videos(root, out, VIDEO_SOURCES[args.source], metadata, workers=args.workers, resume=True)
    if not summary["ingested"]:
        raise ValueError(f"No usable samples in {args.source}. See {out / 'summary.json'}.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_sources(args):
    root = args.work_dir / "sources"
    voya = root / "voya"
    parts = json.loads((voya / "parts.json").read_text())
    sources = [*(voya / name for name in parts), *(root / name for name in VIDEO_SOURCES)]
    if not any(args.sstcn_raw.rglob("*.npy")):
        raise FileNotFoundError(f"No SSTCN archives have been extracted under {args.sstcn_raw}.")
    for source in sources:
        summary = source / "summary.json"
        if not summary.is_file() or not json.loads(summary.read_text())["ingested"]:
            raise ValueError(f"Ingestion is missing or empty for {source}.")
    report = prepare_unified(
        output_dir=args.work_dir / "prepared", source_dirs=sources,
        include_sstcn_words=True, sstcn_raw_root=args.sstcn_raw,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("source", choices=["voya", *VIDEO_SOURCES])
    ingest.add_argument("--input-dir", type=Path)
    ingest.add_argument("--metadata", type=Path)
    ingest.add_argument("--workers", type=int, default=2, choices=range(1, 9))
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--sstcn-raw", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "ingest":
        ingest_source(args)
    else:
        prepare_sources(args)


if __name__ == "__main__":
    main()
