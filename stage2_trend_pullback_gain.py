#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段二趋势回调做多增益测试入口。"""

from __future__ import annotations

from blueblue_research.trend_pullback import parse_args, run


def main() -> int:
    result = run(parse_args())
    print(f"output_dir={result['output_dir']}")
    print(f"contracts={result['contracts']}, rows={result['rows']}, combos={result['combos']}")
    print(f"train_end={result['train_end']}, test_start={result['test_start']}")
    print(f"trades={result['trades']}, audit_blockers={result['audit_blockers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

