"""External release manifests; hidden annotations remain evaluator-side."""

import hashlib
import json
from collections import Counter
from pathlib import Path

FAMILIES = ("l2_passive", "l2_dynamic", "l2_interaction", "l3_owner_habit")


def contained_path(root, value):
    root = Path(root).expanduser().resolve()
    value = Path(value)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Expected a relative path within the dataset: {value}")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {value}")
    return path


def load_manifest(manifest, data_root, *, per_family=None, families=None, verify=True):
    with Path(manifest).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    seen = set()
    counts = Counter()
    selected = []
    wanted = set(families or FAMILIES)
    if wanted - set(FAMILIES):
        raise ValueError(f"Unknown task families: {wanted - set(FAMILIES)}")
    if per_family is not None and per_family < 1:
        raise ValueError("per_family must be positive")
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Duplicate manifest ID: {row['id']}")
        seen.add(row["id"])
        family = row["family"]
        if family not in wanted or (
            per_family is not None and counts[family] >= per_family
        ):
            continue
        source = row["source"]
        path = contained_path(data_root, source["episode_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify and source.get("episode_sha256"):
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != source["episode_sha256"]
            ):
                raise ValueError(
                    f"Episode hash differs from release manifest: {row['id']}"
                )
        selected.append(row)
        counts[family] += 1
    if not selected:
        raise ValueError("Selection is empty")
    return selected


def resolve_image(value, *, episode_path, data_root):
    """Resolve portable image references without reaching outside the data release."""
    if not value:
        return None
    root = Path(data_root).resolve()
    value = Path(value)
    candidates = (
        [value]
        if value.is_absolute()
        else [Path(episode_path).parent / value, root / value]
    )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Image is missing or outside the dataset: {value}")


def resolve_history_image(
    value, *, episode_path, data_root, session_index, num_sessions
):
    """Disambiguate legacy per-session basenames; never choose an arbitrary match."""
    if not value:
        raise FileNotFoundError("A history step has no RGB reference")
    value = Path(value)
    base = Path(episode_path).parent
    if value.is_absolute() or value.parent != Path("."):
        return resolve_image(value, episode_path=episode_path, data_root=data_root)
    if num_sessions == 2 and (base / "context").is_dir() and (base / "noise").is_dir():
        folder = ["context", "noise"][session_index]
    else:
        import re

        def numbered(prefix):
            return sorted(
                (
                    p
                    for p in base.iterdir()
                    if p.is_dir() and re.fullmatch(prefix + r"\d+", p.name)
                ),
                key=lambda p: int(re.search(r"\d+$", p.name)[0]),
            )

        folders = numbered("session_") + numbered("ep_l3_noise_s")
        if len(folders) != num_sessions:
            raise ValueError(
                "Ambiguous legacy image layout; provide session-relative image_path fields"
            )
        folder = folders[session_index].name
    return resolve_image(
        Path(folder) / value, episode_path=episode_path, data_root=data_root
    )
