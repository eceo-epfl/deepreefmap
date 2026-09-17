"""Render the e2e golden change as a pull request body.

    python tests/golden_diff.py before.json after.json --pr N --head BRANCH --sha SHA --run-url URL

Lists every value that moved beyond the tolerance test_reconstruct_e2e.py applies
to its block. Exits 1 when nothing did, so a caller can skip opening a PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runs are bitwise identical on one machine at a fixed thread count. Across
# GitHub's runner fleet float32 reductions group differently per CPU, measured
# at rel 1.6e-6 (mapping), rel 1.5e-4 (cloud) and abs 5.2e-4 (cover) over 12
# legs. Each tolerance sits an order of magnitude above that drift, so a runner
# swap cannot fail the test and a code change has to move a value well past
# hardware noise to be reported.
MAPPING_RTOL = 3e-4
CLOUD_RTOL = 2e-3
COVER_ATOL = 5e-3

GOLDEN_PATH = "tests/data/reconstruct_e2e_golden.json"


def _outside_rtol(old: float, new: float, rtol: float) -> bool:
    return abs(new - old) > rtol * abs(old)


def changed_rows(before: dict, after: dict) -> list[tuple[str, str, object, object]]:
    rows: list[tuple[str, str, object, object]] = []
    for scenario in sorted(set(before) | set(after)):
        if scenario not in before:
            rows.append((scenario, "new scenario", "", ""))
            continue
        if scenario not in after:
            rows.append((scenario, "removed", "", ""))
            continue
        old, new = before[scenario], after[scenario]
        for key in sorted(set(old["structure"]) | set(new["structure"])):
            if old["structure"].get(key) != new["structure"].get(key):
                rows.append((scenario, f"structure.{key}", old["structure"].get(key), new["structure"].get(key)))
        for block, rtol in (("mapping", MAPPING_RTOL), ("cloud", CLOUD_RTOL)):
            for key, value in new[block].items():
                prior = old[block].get(key)
                if prior is None or _outside_rtol(prior, value, rtol):
                    rows.append((scenario, f"{block}.{key}", prior, value))
        for name in sorted(set(old["cover"]) | set(new["cover"])):
            old_frac, new_frac = old["cover"].get(name, 0.0), new["cover"].get(name, 0.0)
            if abs(new_frac - old_frac) > COVER_ATOL:
                rows.append((scenario, f"cover.{name}", old_frac, new_frac))
    return rows


def _change(value: str, old: object, new: object) -> str:
    if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
        return ""
    if value.startswith("cover."):
        return f"{new - old:+.3f}"
    return f"{(new - old) / old:+.1%}" if old else ""


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else str(value)


def _quiet_changes(rows: list, before: dict, after: dict) -> tuple[int, int]:
    listed = {(scenario, value) for scenario, value, *_ in rows}
    quiet = [
        (scenario, f"{block}.{key}")
        for scenario in set(before) & set(after)
        for block in ("mapping", "cloud", "cover")
        for key, value in after[scenario][block].items()
        if before[scenario][block].get(key) != value and (scenario, f"{block}.{key}") not in listed
    ]
    return len(quiet), len({scenario for scenario, _ in quiet})


def render(rows: list, before: dict, after: dict, pr: int, head: str, sha: str, run_url: str) -> str:
    lines = [
        f"Regenerates `{GOLDEN_PATH}` for `{head}` (#{pr}) at `{sha}`, from the e2e run that failed: {run_url}.",
        "",
        "Merge if the values below are meant to move. Merging reruns the checks on the feature PR.",
        "If they are not, fix it on the feature branch instead and close this PR. Nothing else changes.",
        "",
        "| Scenario | Value | Before | After | Change |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {scenario} | {value} | {_fmt(old)} | {_fmt(new)} | {_change(value, old, new)} |"
        for scenario, value, old, new in rows
    ]
    values, scenarios = _quiet_changes(rows, before, after)
    lines += ["", f"{values} values across {scenarios} scenarios moved within tolerance and are not listed."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--run-url", required=True)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    rows = changed_rows(before, after)
    if not rows:
        print("golden unchanged beyond tolerance", file=sys.stderr)
        return 1
    sys.stdout.write(render(rows, before, after, args.pr, args.head, args.sha, args.run_url))
    return 0


if __name__ == "__main__":
    sys.exit(main())
