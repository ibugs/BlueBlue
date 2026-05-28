#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段二增益曲线入口：Delta → POC → 位置。

默认读取 `stage2_outputs/SHFE.cu` 中已经生成的阶段二结果，不重新读取 tick。
"""

from __future__ import annotations

from blueblue_research.gain_curve import parse_args, run


def main() -> int:
    result = run(parse_args())
    print(f"output_dir={result['output_dir']}")
    print(f"contracts={result['contracts']}, rows={result['rows']}, combos={result['combos']}")
    print(f"train_end={result['train_end']}, test_start={result['test_start']}")
    print(f"trades={result['trades']}, audit_blockers={result['audit_blockers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
