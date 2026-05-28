#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段二 P0 阻断审计入口。"""

from __future__ import annotations

from blueblue_research.blocker_audit import parse_args, run


def main() -> int:
    result = run(parse_args())
    print(f"output_dir={result['output_dir']}")
    print(f"features_rows={result['features_rows']}, labels_rows={result['labels_rows']}")
    print(f"findings={result['findings']}, blocker_fail_count={result['blocker_fail_count']}")
    print(f"overall_status={result['overall_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
