#!/usr/bin/env python3
"""Warns if committed docs contain raw AI-orchestration markers.

Non-blocking by default. Set DOCS_STRICT=1 env to fail commit.

Convention:
- docs/_drafts/  — raw planning transcripts (gitignored, local only).
- docs/designs/, docs/plans/, docs/pilot/  — cleaned artifacts (committed).

This hook warns when files in the committed paths contain markers typical
of raw planning transcripts (reviewer findings, sentiment scores, drift IDs,
revision histories). Override via env to make it block commits in the
pre-public-flip checklist.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PATTERNS = [
    r"reviewer agent",
    r"office-hours reviewer",
    r"grill-me said",
    r"sentiment score:\s*\d",
    r"quality score:\s*\d",
    r"(?:DRIFT|CON|CUT|BvB|MAINT|TEST|OSS|ADV|KAR)-\d",
    r"deepening pass",
    r"revision history",
]


def main(argv: list[str]) -> int:
    strict = os.environ.get("DOCS_STRICT") == "1"
    issues: list[tuple[str, str]] = []
    for path_str in argv[1:]:
        path = Path(path_str)
        if not path.exists():
            continue
        content = path.read_text()
        for pat in PATTERNS:
            if re.search(pat, content, re.IGNORECASE):
                issues.append((path_str, pat))
                break  # one hit per file is enough

    for path, pat in issues:
        print(f"⚠️  {path}: AI-process marker '{pat}' — clean before public commit")
        print("   Raw drafts belong in docs/_drafts/ (gitignored)")

    return 1 if (issues and strict) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
