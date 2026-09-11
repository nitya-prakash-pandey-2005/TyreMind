"""No experiment may read a lap table without the repair applied.

This is a structural test rather than a behavioural one, and it exists because
the bug it guards against has already happened once.

The fuel counter was fixed in the loader, and five experiments kept the old
behaviour because they called `pd.read_parquet` directly -- including the one
that produces the rate estimates two others consume. Nothing failed. The
results simply disagreed with each other, and the only symptom was a number
that refused to move after the thing it depended on had changed.

A half-applied repair is worse than none: uniformly wrong results at least agree
with each other, while a partial fix produces documents whose figures contradict
one another with nothing to indicate which half is which. So the rule is that a
cached lap table enters an experiment through `read_lap_table` or not at all,
and this test enforces it by reading the source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
SOURCE = ROOT / "src" / "tyremind"

#: Files allowed to call pd.read_parquet directly, and why.
ALLOWED = {
    # The one implementation of the repair. Everything else routes through it.
    "corpus.py",
}

#: A direct read is also fine when the file being read is an experiment's own
#: derived cache rather than a session lap table -- those already contain
#: repaired values, because they were produced from repaired ones.
CACHE_READ = re.compile(r"read_parquet\(\s*(?:STINT_)?CACHE\s*\)")

DIRECT_READ = re.compile(r"\bpd\.read_parquet\(")


def python_files(*roots: Path) -> list[Path]:
    return sorted(p for root in roots for p in root.rglob("*.py"))


@pytest.mark.parametrize("path", python_files(EXPERIMENTS, SOURCE), ids=lambda p: p.name)
def test_lap_tables_are_read_through_the_repairing_loader(path: Path) -> None:
    if path.name in ALLOWED:
        pytest.skip(f"{path.name} owns the repair")

    text = path.read_text(encoding="utf-8")
    offending = [
        line.strip()
        for line in text.splitlines()
        if DIRECT_READ.search(line) and not CACHE_READ.search(line)
    ]

    assert not offending, (
        f"{path.relative_to(ROOT)} reads a lap table directly:\n  "
        + "\n  ".join(offending)
        + "\n\nUse tyremind.data.corpus.read_lap_table, which recomputes the fuel "
        "counter. A direct read silently keeps the pre-fix behaviour, which is how "
        "five experiments came to disagree with the rest of the project."
    )


def test_the_repair_itself_still_exists() -> None:
    """Guards the guard. If read_lap_table were renamed away, every file above
    would pass this module while doing nothing at all."""
    from tyremind.data.corpus import read_lap_table

    assert callable(read_lap_table)


def test_the_api_store_repairs_what_it_serves() -> None:
    """The dashboard reads the committed demo parquet through its own path, so
    it needs the repair too -- otherwise the product shows different numbers
    from the experiments that validate it."""
    text = (SOURCE / "api" / "store.py").read_text(encoding="utf-8")
    assert "read_lap_table" in text
