"""docs/NONIDENTIFIABILITY.md: every construction must reproduce the cross-spectra to rounding error, with
non-negative gains where gains are claimed."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_all_constructions_agree_to_rounding_error():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "nonident_check.py")], capture_output=True, text=True,
                         check=True).stdout
    lines = [ln for ln in out.splitlines() if "max rel" in ln]
    assert len(lines) == 5, out                                    # checks 1-3, and the two lines of check 4
    for ln in lines:
        assert float(re.search(r"max rel [^:]*:\s+([0-9.e+-]+)", ln).group(1)) < 1e-12, ln
    gains = [float(g) for g in re.findall(r"smallest gain ([0-9.e+-]+)", out)]
    assert len(gains) == 2 and min(gains) >= 0, out                # world B uses gains only, all non-negative
