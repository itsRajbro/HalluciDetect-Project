"""Tests for the freeze gate's git-cleanliness filter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.smoke_test import dirty_paths  # noqa: E402


def test_generated_files_do_not_count_as_dirty():
    porcelain = "\n".join([
        "?? outputs/detectors/length_only/chars/test.jsonl",
        "?? results/eval/length_only/chars.json",
        "?? results/foundation_smoke_report.json",
        "?? results/env_report_NitroV15.json",
    ])
    assert dirty_paths(porcelain) == []


def test_real_changes_are_reported():
    porcelain = "\n".join([
        " M src/common/config.py",
        "?? data/split_manifest.json",
        "R  old.py -> src/common/new.py",
        '?? "src/with space/a.py"',
        "?? outputs/x.jsonl",
    ])
    assert dirty_paths(porcelain) == ["src/common/config.py", "data/split_manifest.json",
                                      "src/common/new.py", "src/with space/a.py"]


def test_git_helper_keeps_leading_space_of_first_status_line(monkeypatch):
    """Regression: strip() used to eat the leading space of ' M file', so the first
    path lost its first character ('requirements.txt' -> 'equirements.txt')."""
    import subprocess
    from types import SimpleNamespace
    from src.common import smoke_test as st

    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=" M requirements.txt\n?? scripts/\n"))
    out = st._git("status", "--porcelain")
    assert out.startswith(" M")
    assert st.dirty_paths(out) == ["requirements.txt", "scripts/"]
