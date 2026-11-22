"""Unified scope and QIPEDC video region filter for Camera `words` mode.

Enforces two simultaneous conditions for `mode == "words"`:
  Condition A (ALLOWED_WORD_IDS): `entry_id` belongs to the active allowed word
      list (never fingerspelling letters or digits 1-10, and respecting any topic
      or configured allowlist).
  Condition B (ELIGIBLE_VIDEO_ENTRIES): Recognition reference videos must come
      from verified QIPEDC `.mp4` sources matching `^([DW]\\d+)([BNT])?$` with:
        - No suffix: Nationwide (ACCEPT)
        - Suffix `N`: South (ACCEPT)
        - Suffix `B`: North (REJECT)
        - Suffix `T`: Central (REJECT)
        - Anything else / missing / GT Supabase: UNVERIFIED (REJECT for QIPEDC;
          active SSTCN GT classes remain separately audited & gated by flag).
"""

from functools import lru_cache
import json
import os
from pathlib import Path
import posixpath
import re
from typing import Any, Optional
from urllib.parse import urlparse

from backend.models.lexicon import load_lexicon, norm_key

# Verified across all 4,362 raw rows of backend/data/sources/qipedc_dictionary.json
# and all 3,318 QIPEDC entries in backend/data/vsl_lexicon.json.
QIPEDC_VIDEO_ID_RE = re.compile(r"^([DW]\d+)([BNT])?$")

ACCEPTED_QIPEDC_REGIONS = frozenset({"NATIONWIDE", "N"})
REJECTED_QIPEDC_REGIONS = frozenset({"B", "T"})

_DEFAULT_SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
_seed_provenance_cache: dict[str, Any] = {"key": None, "data": None}


@lru_cache(maxsize=8192)
def _parse_qipedc_video_region_cached(video_or_url: Optional[str]) -> tuple:
    if not isinstance(video_or_url, str):
        return (None, None, "missing", "UNKNOWN", False, "missing_video")
    raw = video_or_url.strip()
    if not raw:
        return (None, None, "missing", "UNKNOWN", False, "missing_video")

    is_url = "://" in raw
    if is_url:
        parsed = urlparse(raw)
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        fn = posixpath.basename(path)
        stem, ext = os.path.splitext(fn)

        if "supabase.co" in netloc:
            return (stem or None, None, "gt_curriculum", "GT", False, "unverified_gt_curriculum")

        is_qipedc_video = netloc.endswith("qipedc.moet.gov.vn") and "/videos/" in path
        if not is_qipedc_video:
            return (stem or None, None, "external", "UNKNOWN", False, "non_qipedc_source")

        if ext.lower() != ".mp4":
            return (stem or None, None, "qipedc", "UNKNOWN", False, "invalid_video_extension")

        source_type = "qipedc"
    else:
        fn = raw.replace("\\", "/").rsplit("/", 1)[-1]
        if "." in fn:
            stem, ext = os.path.splitext(fn)
            if ext.lower() != ".mp4":
                return (stem or None, None, "unknown", "UNKNOWN", False, "invalid_video_extension")
        else:
            stem = fn
        source_type = "qipedc"

    if not stem:
        return (None, None, source_type, "UNKNOWN", False, "missing_video")

    m = QIPEDC_VIDEO_ID_RE.fullmatch(stem)
    if not m:
        return (stem, None, source_type, "UNKNOWN", False, "unverified_region")

    base_id, suffix = m.group(1), m.group(2)
    if suffix is None:
        return (stem, base_id, "qipedc", "NATIONWIDE", True, "accepted_nationwide")
    if suffix == "N":
        return (stem, base_id, "qipedc", "N", True, "accepted_south")
    if suffix == "B":
        return (stem, base_id, "qipedc", "B", False, "rejected_region_north")
    return (stem, base_id, "qipedc", "T", False, "rejected_region_central")


def parse_qipedc_video_region(video_or_url: Optional[str]) -> dict[str, Any]:
    """Parse video filename/URL/ID and classify its QIPEDC region strictly.

    Never inspects the raw trailing character of an arbitrary URL and never treats
    unknown or unverified formats as nationwide.
    """
    video_id, base_id, source_type, region, eligible, reason = _parse_qipedc_video_region_cached(
        video_or_url if isinstance(video_or_url, str) else None
    )
    return {
        "video_id": video_id,
        "base_id": base_id,
        "source_type": source_type,
        "region": region,
        "eligible": eligible,
        "reason": reason,
    }


def _resolve_seed_dir(seed_dir: Optional[Path] = None) -> Path:
    if seed_dir is not None:
        return Path(seed_dir)
    env_dir = os.environ.get("VSL_QIPEDC_REFERENCE_DIR")
    if env_dir:
        return Path(env_dir)
    return _DEFAULT_SEED_DIR


def load_seed_provenance(seed_dir: Optional[Path] = None) -> dict[str, Any]:
    """Load verified seed provenance mappings from `sources.json`."""
    resolved_dir = _resolve_seed_dir(seed_dir)
    src_file = resolved_dir / "sources.json"
    if not src_file.is_file():
        return {"by_file": {}, "by_video_id": {}, "by_entry_id": {}}

    try:
        stat = src_file.stat()
        cache_key = (str(src_file.resolve()), stat.st_mtime_ns)
    except OSError:
        cache_key = None

    if cache_key is not None and _seed_provenance_cache["key"] == cache_key:
        return _seed_provenance_cache["data"]

    try:
        rows = json.loads(src_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"by_file": {}, "by_video_id": {}, "by_entry_id": {}}

    by_file: dict[str, dict[str, Any]] = {}
    by_video_id: dict[str, dict[str, Any]] = {}
    by_entry_id: dict[str, list[dict[str, Any]]] = {}

    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            fname = str(row.get("file") or "")
            source_url = row.get("source")
            entry_id = row.get("entry_id")
            parsed = parse_qipedc_video_region(source_url)
            record = {
                "entry_id": entry_id,
                "file": fname,
                "source": source_url,
                "sha256": row.get("sha256"),
                "parsed": parsed,
            }
            if fname:
                by_file[fname] = record
                stem = Path(fname).stem
                key_vid = stem[3:] if stem.startswith("v3_") else stem
                by_video_id[key_vid] = record
            if parsed["video_id"] and parsed["video_id"] not in by_video_id:
                by_video_id[parsed["video_id"]] = record
            if entry_id:
                by_entry_id.setdefault(str(entry_id), []).append(record)

    result = {"by_file": by_file, "by_video_id": by_video_id, "by_entry_id": by_entry_id}
    if cache_key is not None:
        _seed_provenance_cache["key"] = cache_key
        _seed_provenance_cache["data"] = result
    return result


def verify_reference_video_id(video_id: Optional[str], seed_dir: Optional[Path] = None) -> dict[str, Any]:
    """Verify a QIPEDC reference video ID or alias (such as `D0261_close`)."""
    direct = parse_qipedc_video_region(video_id)
    if direct["eligible"] or direct["region"] in REJECTED_QIPEDC_REGIONS:
        return direct

    if isinstance(video_id, str):
        key = video_id.strip()
        prov = load_seed_provenance(seed_dir=seed_dir)["by_video_id"].get(key)
        if prov is not None:
            parsed = dict(prov["parsed"])
            parsed["alias_of"] = key
            return parsed
    return direct


def is_non_word_entry(entry: Optional[dict[str, Any]]) -> Optional[str]:
    """Return rejection reason if a lexicon entry is an alphabet letter or digit 1-10."""
    if not entry:
        return "missing_entry"
    if entry.get("fingerspelling"):
        return "fingerspelling_not_word"
    sstcn_idx = entry.get("sstcn_index")
    if sstcn_idx is not None and int(sstcn_idx) < 39:
        return "digit_not_word"
    return None


def filter_entry_eligible_videos(entry: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate primary `video_url` and each item in `variants` independently.

    If an entry has both `N` and `B` (or `T` / `GT`) videos, ONLY eligible Nationwide
    and `N` QIPEDC videos are retained; `B`/`T`/unverified variants are rejected.
    """
    if not entry:
        return {
            "entry_id": None,
            "eligible": False,
            "eligible_videos": [],
            "rejected_videos": [],
            "selected_video_id": None,
            "selected_video_url": None,
            "selected_region": None,
            "reason": "missing_entry",
        }

    candidates: list[tuple[Optional[str], Optional[str], bool]] = []
    if entry.get("video_url"):
        candidates.append((entry.get("video_url"), entry.get("region"), True))
    for variant in entry.get("variants") or []:
        if isinstance(variant, dict):
            candidates.append((variant.get("url"), variant.get("region"), False))

    eligible_videos: list[dict[str, Any]] = []
    rejected_videos: list[dict[str, Any]] = []
    seen_eligible_vids: set[str] = set()
    seen_rejected_keys: set[tuple[Optional[str], str]] = set()

    for url, claimed_region, is_primary in candidates:
        parsed = parse_qipedc_video_region(url)
        # Never trust a variant if metadata claims North/Central even when URL is malformed.
        if parsed["eligible"] and claimed_region in REJECTED_QIPEDC_REGIONS:
            parsed["eligible"] = False
            parsed["region"] = claimed_region
            parsed["reason"] = (
                "rejected_region_north" if claimed_region == "B" else "rejected_region_central"
            )

        item = {**parsed, "url": url, "is_primary": is_primary}
        if item["eligible"] and item["video_id"]:
            if item["video_id"] not in seen_eligible_vids:
                seen_eligible_vids.add(item["video_id"])
                eligible_videos.append(item)
        else:
            rkey = (url, item["reason"])
            if rkey not in seen_rejected_keys:
                seen_rejected_keys.add(rkey)
                rejected_videos.append(item)

    if eligible_videos:
        chosen = eligible_videos[0]
        return {
            "entry_id": entry.get("id"),
            "eligible": True,
            "eligible_videos": eligible_videos,
            "rejected_videos": rejected_videos,
            "selected_video_id": chosen["video_id"],
            "selected_video_url": chosen["url"],
            "selected_region": chosen["region"],
            "reason": chosen["reason"],
        }

    reason = rejected_videos[0]["reason"] if rejected_videos else "missing_video"
    return {
        "entry_id": entry.get("id"),
        "eligible": False,
        "eligible_videos": [],
        "rejected_videos": rejected_videos,
        "selected_video_id": None,
        "selected_video_url": None,
        "selected_region": None,
        "reason": reason,
    }


def eligible_qipedc_word_entries(lex: Optional[dict[str, Any]] = None) -> dict[str, dict[str, Any]]:
    """Map every `entry_id` passing word-type check AND Condition B to its video info."""
    if lex is None:
        lex = load_lexicon()
    eligible_map: dict[str, dict[str, Any]] = {}
    for entry in lex.get("entries", []):
        if is_non_word_entry(entry) is not None:
            continue
        info = filter_entry_eligible_videos(entry)
        if info["eligible"]:
            eligible_map[entry["id"]] = info
    return eligible_map


def allowed_live_word_ids(
    lex: Optional[dict[str, Any]] = None,
    *,
    indexed_ids: Any = None,
    builtin_ids: Any = None,
    extra_ids: Any = None,
    topic_id: Optional[str] = None,
    allowed_override: Optional[Any] = None,
) -> frozenset[str]:
    """Compute `ALLOWED_WORD_IDS ∩ ELIGIBLE_VIDEO_ENTRIES` for QIPEDC Live words mode.

    When `topic_id` is supplied and matches 0 eligible words, returns an empty
    frozenset (never None) so QIPEDC Live searches 0 candidates rather than falling
    back to the entire dictionary.
    """
    if lex is None:
        lex = load_lexicon()
    eligible_map = eligible_qipedc_word_entries(lex)

    if allowed_override is not None:
        allowed_pool = set(allowed_override)
    else:
        allowed_pool = set(builtin_ids or ()) | set(indexed_ids or ()) | set(extra_ids or ())

    valid = allowed_pool & set(eligible_map.keys())
    if topic_id:
        topic_ids = {
            e["id"]
            for e in lex.get("entries", [])
            if e.get("category") == topic_id and e.get("id")
        }
        valid &= topic_ids
    return frozenset(valid)


def approved_live_word_ids(
    lex: Optional[dict[str, Any]] = None,
    *,
    indexed_ids: Any = None,
    builtin_ids: Any = None,
) -> frozenset[str]:
    """Return the approved QIPEDC Live allowlist (`live_index ∪ builtin` ∩ eligible QIPEDC words)."""
    return allowed_live_word_ids(lex, indexed_ids=indexed_ids, builtin_ids=builtin_ids)


def validate_extra_qipedc_word(
    entry_id: Optional[str],
    lex: Optional[dict[str, Any]] = None,
    *,
    allowed_override: Optional[Any] = None,
) -> dict[str, Any]:
    """Validate whether an entry may enter `BUILTIN_WORDS`, `extra_words`, or `live/prepare`."""
    if lex is None:
        lex = load_lexicon()
    if not entry_id or entry_id not in lex.get("by_id", {}):
        return {
            "allowed": False,
            "entry_id": entry_id,
            "entry": None,
            "reason": "entry_not_in_lexicon",
            "video_id": None,
            "region": None,
        }

    entry = lex["by_id"][entry_id]
    if allowed_override is not None and entry_id not in allowed_override:
        return {
            "allowed": False,
            "entry_id": entry_id,
            "entry": entry,
            "reason": "entry_not_in_allowed_scope",
            "video_id": None,
            "region": None,
        }

    non_word = is_non_word_entry(entry)
    if non_word is not None:
        return {
            "allowed": False,
            "entry_id": entry_id,
            "entry": entry,
            "reason": non_word,
            "video_id": None,
            "region": None,
        }

    vid_info = filter_entry_eligible_videos(entry)
    if not vid_info["eligible"]:
        return {
            "allowed": False,
            "entry_id": entry_id,
            "entry": entry,
            "reason": vid_info["reason"],
            "video_id": None,
            "region": entry.get("region"),
        }

    return {
        "allowed": True,
        "entry_id": entry_id,
        "entry": entry,
        "reason": vid_info["reason"],
        "video_id": vid_info["selected_video_id"],
        "region": vid_info["selected_region"],
    }


def _resolve_sstcn_entry(model: Any, idx: int, lex: dict[str, Any]) -> Optional[dict[str, Any]]:
    entry = lex.get("by_sstcn", {}).get(idx)
    if entry is not None:
        return entry
    entry_ids = getattr(model, "entry_ids", None)
    if entry_ids and 0 <= idx < len(entry_ids) and entry_ids[idx]:
        entry = lex.get("by_id", {}).get(entry_ids[idx])
        if entry is not None:
            return entry
    class_names = getattr(model, "class_names", None)
    if class_names and 0 <= idx < len(class_names):
        return lex.get("by_key", {}).get(norm_key(class_names[idx]))
    return None


def allowed_sstcn_word_indices(
    model: Any,
    lex: Optional[dict[str, Any]] = None,
    *,
    topic_id: Optional[str] = None,
    include_pending_gt_sstcn: bool = False,
    allowed_override: Optional[Any] = None,
) -> tuple[list[int], dict[str, Any]]:
    """Return allowed SSTCN word class indices for `mode == "words"` and audit metadata.

    Strictly excludes class indices `< 39` (alphabet letters & digits 1-10) even if
    a caller passes an alphabet/digit `topic_id`. Unverified `GT` curriculum classes
    are NEVER treated as QIPEDC Nationwide/South and are excluded by default
    (`include_pending_gt_sstcn=False`).
    """
    if lex is None:
        lex = load_lexicon()

    num_classes = int(getattr(model, "num_classes", 102))
    raw_word_indices = list(getattr(model, "word_class_indices", list(range(39, num_classes))))
    word_indices = [int(i) for i in raw_word_indices if 39 <= int(i) < num_classes]
    topic_map = getattr(model, "topic_map", {}) or {}

    allowed_indices: list[int] = []
    excluded: list[dict[str, Any]] = []
    pending_gt_indices: list[int] = []

    for idx in word_indices:
        entry = _resolve_sstcn_entry(model, idx, lex)
        if entry is None:
            excluded.append({"index": idx, "entry_id": None, "reason": "missing_lexicon_entry"})
            continue

        eid = entry.get("id")
        if allowed_override is not None and eid not in allowed_override:
            excluded.append({"index": idx, "entry_id": eid, "reason": "entry_not_in_allowed_scope"})
            continue

        non_word = is_non_word_entry(entry)
        if non_word is not None:
            excluded.append({"index": idx, "entry_id": eid, "reason": non_word})
            continue

        vid_info = filter_entry_eligible_videos(entry)
        if vid_info["eligible"]:
            allowed_indices.append(idx)
        elif entry.get("region") == "GT":
            if include_pending_gt_sstcn:
                allowed_indices.append(idx)
                pending_gt_indices.append(idx)
            else:
                excluded.append({"index": idx, "entry_id": eid, "reason": "unverified_gt_curriculum"})
        else:
            excluded.append({"index": idx, "entry_id": eid, "reason": vid_info["reason"]})

    reject_reason = None
    if topic_id:
        if topic_id in topic_map:
            topic_set = {int(i) for i in topic_map[topic_id] if 0 <= int(i) < num_classes}
            allowed_indices = [i for i in allowed_indices if i in topic_set]
            if not allowed_indices:
                reject_reason = "topic_no_eligible_words"
        else:
            topic_entry_ids = {
                e["id"]
                for e in lex.get("entries", [])
                if e.get("category") == topic_id and e.get("id")
            }
            if topic_entry_ids:
                allowed_indices = [
                    i
                    for i in allowed_indices
                    if (e := _resolve_sstcn_entry(model, i, lex)) and e.get("id") in topic_entry_ids
                ]
            else:
                allowed_indices = []
            if not allowed_indices:
                reject_reason = "topic_no_eligible_words"
    elif not allowed_indices:
        reject_reason = excluded[0]["reason"] if excluded else "no_eligible_word_classes"

    return allowed_indices, {
        "reject_reason": reject_reason,
        "excluded": excluded,
        "pending_gt_indices": pending_gt_indices,
    }


def validate_word_result_scope(
    entry_id: Optional[str],
    *,
    source: str,
    lex: Optional[dict[str, Any]] = None,
    allowed_ids: Optional[Any] = None,
    include_pending_gt_sstcn: bool = False,
    sign_label: Optional[str] = None,
) -> dict[str, Any]:
    """Final validation gate for any `mode == "words"` prediction before response emission."""
    if lex is None:
        lex = load_lexicon()

    entry = lex.get("by_id", {}).get(entry_id) if entry_id else None
    if entry is None and entry_id:
        return {
            "valid": False,
            "entry_id": entry_id,
            "reject_reason": "entry_not_in_lexicon",
            "region": "UNKNOWN",
            "selected_video_id": None,
            "selected_video_url": None,
            "verification_status": "rejected_unknown_entry",
        }
    if entry is None and sign_label:
        entry = lex.get("by_key", {}).get(norm_key(sign_label))
    if entry is None:
        return {
            "valid": False,
            "entry_id": entry_id,
            "reject_reason": "missing_entry_id" if not sign_label else "unknown_sign_label",
            "region": "UNKNOWN",
            "selected_video_id": None,
            "selected_video_url": None,
            "verification_status": "rejected_unknown_entry",
        }

    resolved_id = entry["id"]
    non_word = is_non_word_entry(entry)
    if non_word is not None:
        return {
            "valid": False,
            "entry_id": resolved_id,
            "reject_reason": non_word,
            "region": entry.get("region") or "UNKNOWN",
            "selected_video_id": None,
            "selected_video_url": None,
            "verification_status": "rejected_non_word",
        }

    vid_info = filter_entry_eligible_videos(entry)
    if not vid_info["eligible"]:
        if entry.get("region") == "GT" or vid_info["reason"] == "unverified_gt_curriculum":
            if not (source == "sstcn" and include_pending_gt_sstcn):
                return {
                    "valid": False,
                    "entry_id": resolved_id,
                    "reject_reason": "unverified_gt_curriculum",
                    "region": "GT",
                    "selected_video_id": None,
                    "selected_video_url": None,
                    "verification_status": "rejected_gt_unverified",
                }
        else:
            return {
                "valid": False,
                "entry_id": resolved_id,
                "reject_reason": vid_info["reason"],
                "region": entry.get("region") or "UNKNOWN",
                "selected_video_id": None,
                "selected_video_url": None,
                "verification_status": "rejected_video_region",
            }

    if allowed_ids is not None and resolved_id not in allowed_ids:
        return {
            "valid": False,
            "entry_id": resolved_id,
            "reject_reason": "entry_not_in_allowed_scope",
            "region": entry.get("region") or "UNKNOWN",
            "selected_video_id": None,
            "selected_video_url": None,
            "verification_status": "rejected_out_of_scope",
        }

    if vid_info["eligible"]:
        return {
            "valid": True,
            "entry_id": resolved_id,
            "reject_reason": None,
            "region": vid_info["selected_region"],
            "selected_video_id": vid_info["selected_video_id"],
            "selected_video_url": vid_info["selected_video_url"],
            "verification_status": "verified_qipedc",
        }

    return {
        "valid": True,
        "entry_id": resolved_id,
        "reject_reason": None,
        "region": "GT",
        "selected_video_id": None,
        "selected_video_url": entry.get("video_url"),
        "verification_status": "pending_approval_gt_curriculum",
    }
