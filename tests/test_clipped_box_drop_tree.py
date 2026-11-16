"""`scripts/clipped_box_drop_tree.py` -- the val tree with the too-loose boxes removed.

The audit CSV is written with `clipped_box_audit.write_csv` itself, so these tests break if the
two scripts stop agreeing on the CSV format. Each labelled val image has two boxes: line 0 gets a
retained fraction under every threshold, line 1 one above it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from scripts.clipped_box_audit import BoxRow, write_csv
from scripts.clipped_box_drop_tree import DropTreeError, main, write_drop_tree

from t2o.data.manifest import DatasetManifest
from t2o.data.mirror import PROVENANCE_FILENAME, mirror_labels_onto_infrared

LOOSE, TIGHT = 0.4, 0.9
THRESHOLD = 0.5


@pytest.fixture
def source(dataset_root: Path, tmp_path: Path) -> DatasetManifest:
    """The shared fixture with thermal labels and a stale val cache, like the frozen custom tree."""
    root = tmp_path / "source"
    shutil.copytree(dataset_root, root)
    manifest_path = root / "data.yaml"
    # `path:` is absolute, so an unrewritten copy would resolve back to the original tree.
    manifest_path.write_text(
        manifest_path.read_text().replace(f"path: {dataset_root}", f"path: {root}")
    )
    manifest = DatasetManifest.load(manifest_path)
    mirror_labels_onto_infrared(manifest)
    (root / "val" / "infrared" / "labels.cache").write_bytes(b"stale")
    return manifest


def _row(split: str, stem: str, index: int, fraction: float, frozen_line: int | None) -> BoxRow:
    return BoxRow(
        split=split,
        stem=stem,
        box_index=index,
        class_id=0,
        retained_fraction=fraction,
        cut_edges="left" if fraction < 1 else "",
        quad_fill=1.0,
        drop_reason="" if frozen_line is not None else "min_visibility",
        in_frozen_tree=True,
        frozen_line=frozen_line,
    )


def _audit_csv(manifest: DatasetManifest, path: Path) -> Path:
    """One kept row per val label line, plus rows that must be ignored."""
    rows: list[BoxRow] = []
    for label in sorted((manifest.root / "val" / "infrared" / "labels").glob("*.txt")):
        count = len([line for line in label.read_text().splitlines() if line.strip()])
        fractions = [LOOSE, TIGHT][:count]
        rows += [_row("val", label.stem, i, f, i) for i, f in enumerate(fractions)]
        # A box the build already dropped: not in the tree, so not counted against its lines.
        rows.append(_row("val", label.stem, count, 0.1, None))
    # A loose train box: train is never touched.
    train = next((manifest.root / "train" / "infrared" / "labels").glob("*.txt"))
    rows.append(_row("train", train.stem, 0, LOOSE, 0))
    write_csv(rows, path)
    return path


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"data.yaml", PROVENANCE_FILENAME}
        and path.suffix != ".cache"
    }


def test_only_the_loose_val_lines_are_gone(source: DatasetManifest, tmp_path: Path) -> None:
    audit = _audit_csv(source, tmp_path / "audit.csv")
    out = write_drop_tree(source, audit, THRESHOLD, tmp_path / "drop").parent

    before, after = _files(source.root), _files(out)
    assert after.keys() == before.keys()
    changed = {name for name in before if before[name] != after[name]}
    assert changed, "no file changed; the fixture would not catch a skipped drop"
    for name in changed:
        assert name.startswith(("val/infrared/labels/", "val/visible/labels/"))
        old_lines = before[name].decode().splitlines(keepends=True)
        # Line 0 was the loose box; every other line is the source's, byte for byte.
        assert after[name].decode() == "".join(old_lines[1:])


def test_a_threshold_under_every_fraction_copies_the_tree_unchanged(
    source: DatasetManifest, tmp_path: Path
) -> None:
    audit = _audit_csv(source, tmp_path / "audit.csv")
    out = write_drop_tree(source, audit, LOOSE, tmp_path / "drop").parent

    assert _files(out) == _files(source.root)


def test_no_cache_and_the_manifest_points_at_the_copy(
    source: DatasetManifest, tmp_path: Path
) -> None:
    audit = _audit_csv(source, tmp_path / "audit.csv")
    out = tmp_path / "drop"

    assert (
        main(
            [
                "--data",
                str(source.path),
                "--audit-csv",
                str(audit),
                "--threshold",
                str(THRESHOLD),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    assert not list(out.rglob("*.cache"))
    loaded = DatasetManifest.load(out / "data.yaml")
    assert loaded.root == out.resolve()
    assert loaded.val_images == out.resolve() / "val" / "visible" / "images"


def test_the_provenance_names_the_threshold_and_the_audit_digest(
    source: DatasetManifest, tmp_path: Path
) -> None:
    audit = _audit_csv(source, tmp_path / "audit.csv")
    out = write_drop_tree(source, audit, THRESHOLD, tmp_path / "drop").parent

    for side in ("infrared", "visible"):
        provenance = json.loads((out / "val" / side / PROVENANCE_FILENAME).read_text())
        assert provenance["min_retained_fraction"] == THRESHOLD
        assert provenance["audit_csv_sha256"] == hashlib.sha256(audit.read_bytes()).hexdigest()
        assert provenance["boxes_dropped"] == provenance["files_changed"] > 0


def test_an_audit_for_another_tree_is_refused(source: DatasetManifest, tmp_path: Path) -> None:
    audit = _audit_csv(source, tmp_path / "audit.csv")
    stem = max((source.root / "val" / "infrared" / "labels").glob("*.txt")).name
    # One extra box on both sides, so the tree no longer has the lines the audit counted.
    for side in ("infrared", "visible"):
        labelled = source.root / "val" / side / "labels" / stem
        labelled.write_text(labelled.read_text() + "0 0.5 0.5 0.1 0.1\n")

    with pytest.raises(DropTreeError, match="audit kept"):
        write_drop_tree(source, audit, THRESHOLD, tmp_path / "drop")


def test_a_non_empty_destination_is_refused(source: DatasetManifest, tmp_path: Path) -> None:
    audit = _audit_csv(source, tmp_path / "audit.csv")
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "x").write_text("")

    with pytest.raises(DropTreeError, match="not empty"):
        write_drop_tree(source, audit, THRESHOLD, tmp_path / "taken")
