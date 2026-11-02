"""Pure file-export operations shared by the Streamlit UI and tests.

Only regular files under the configured data root are exported by default.
The caller must separately authorize any source paths outside that root.
"""

from __future__ import annotations

import io
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping


CHUNK_SIZE = 4 * 1024 * 1024
ONE_GB = 1_000_000_000
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class ExportEntry:
    source: Path
    relative_name: str
    size: int


@dataclass
class ExportPlan:
    entries: list[ExportEntry] = field(default_factory=list)
    candidates: int = 0
    skipped_missing: int = 0
    skipped_outside: int = 0
    skipped_unreadable: int = 0
    diagnostics: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.entries)


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _safe_dataset(value: object) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "dataset"))
    return value.strip("._") or "dataset"


def _unique_relative_name(base: str, used: set[str]) -> str:
    name = base
    suffix = Path(base).suffix
    stem = base[: -len(suffix)] if suffix else base
    n = 1
    while name in used:
        name = f"{stem}__{n}{suffix}"
        n += 1
    used.add(name)
    return name


def make_export_plan(
    rows: Iterable[Mapping[str, object]],
    data_root: Path,
    resolve_file: Callable[[Mapping[str, object]], Path | None],
    allow_outside: bool = False,
    max_files: int | None = None,
) -> ExportPlan:
    """Resolve files once, record skip reasons, and calculate source byte counts."""
    root = Path(data_root).resolve()
    plan = ExportPlan()
    used: set[str] = set()
    for row in rows:
        plan.candidates += 1
        label = f"{row.get('dataset', '')} | {row.get('file_path', '')}"
        try:
            source = resolve_file(row)
            if source is None:
                plan.skipped_missing += 1
                plan.diagnostics.append(f"MISSING | {label}")
                continue
            source = Path(source).resolve()
            if not source.is_file():
                plan.skipped_missing += 1
                plan.diagnostics.append(f"NOT_FILE | {label}")
                continue
            if not (_under(source, root) or allow_outside):
                plan.skipped_outside += 1
                plan.diagnostics.append(f"OUTSIDE | {label} | {source}")
                continue
            size = source.stat().st_size
            if max_files is not None and len(plan.entries) >= max_files:
                plan.diagnostics.append(f"LIMIT | {label}")
                continue
            if _under(source, root):
                relative = source.relative_to(root).as_posix()
            else:
                relative = f"{_safe_dataset(row.get('dataset'))}/{source.name}"
            relative = _unique_relative_name(relative, used)
            plan.entries.append(ExportEntry(source, relative, size))
            plan.diagnostics.append(f"COPY | {label} | {source} | {size} bytes")
        except (OSError, ValueError) as exc:
            plan.skipped_unreadable += 1
            plan.diagnostics.append(f"ERROR | {label} | {exc}")
    return plan


def _diagnostics(plan: ExportPlan) -> bytes:
    lines = [
        f"candidates={plan.candidates}",
        f"included={len(plan.entries)}",
        f"source_bytes={plan.total_bytes}",
        f"skipped_missing={plan.skipped_missing}",
        f"skipped_outside={plan.skipped_outside}",
        f"skipped_unreadable={plan.skipped_unreadable}",
        "",
        *plan.diagnostics,
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _transfer(source, target, processed: int, total: int,
              name: str, on_progress: ProgressCallback | None) -> int:
    while True:
        chunk = source.read(CHUNK_SIZE)
        if not chunk:
            break
        target.write(chunk)
        processed += len(chunk)
        if on_progress:
            on_progress(processed, total, name)
    return processed


def build_zip_bytes(plan: ExportPlan, manifest: bytes,
                    on_progress: ProgressCallback | None = None) -> bytes:
    """Compress directly from sources; report progress by bytes read."""
    buffer = io.BytesIO()
    processed = 0
    total = plan.total_bytes
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for entry in plan.entries:
            member = zipfile.ZipInfo.from_file(entry.source,
                                               f"files/{entry.relative_name}")
            member.compress_type = zipfile.ZIP_DEFLATED
            with entry.source.open("rb") as source, archive.open(
                member, "w", force_zip64=True
            ) as target:
                processed = _transfer(source, target, processed,
                                      total, entry.relative_name, on_progress)
        archive.writestr("manifest.csv", manifest)
        archive.writestr("zip_diagnostics.txt", _diagnostics(plan))
    if on_progress:
        on_progress(total, total, "Complete")
    return buffer.getvalue()


def copy_without_compression(
    plan: ExportPlan,
    manifest: bytes,
    selected_directory: Path,
    allowed_root: Path,
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Copy under a permitted host directory into a unique, non-overwriting folder."""
    allowed = Path(allowed_root).expanduser().resolve()
    chosen = Path(selected_directory).expanduser().resolve()
    if not _under(chosen, allowed):
        raise ValueError(f"Destination must be inside the allowed copy root: {allowed}")
    allowed.mkdir(parents=True, exist_ok=True)
    chosen.mkdir(parents=True, exist_ok=True)
    # mkdtemp creates a unique folder; no previous user export is overwritten.
    output = Path(tempfile.mkdtemp(prefix="rir_copy_", dir=chosen))
    processed = 0
    total = plan.total_bytes
    for entry in plan.entries:
        destination = output / "files" / entry.relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with entry.source.open("rb") as source, destination.open("xb") as target:
            processed = _transfer(source, target, processed,
                                  total, entry.relative_name, on_progress)
        try:
            shutil.copystat(entry.source, destination, follow_symlinks=False)
        except OSError:
            pass  # content was copied; optional timestamps/permissions may not be portable
    (output / "manifest.csv").write_bytes(manifest)
    (output / "zip_diagnostics.txt").write_bytes(_diagnostics(plan))
    if on_progress:
        on_progress(total, total, "Complete")
    return output
