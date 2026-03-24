"""
重置 IMM / SMM JSON 数据。

用途：测试前清空历史状态，避免不同测试用户互相影响。
默认行为：
1) 删除 jl/ 下所有 imm_*.json（保留 PDF）
2) 基于 .env 的 yh* 映射重建空白 imm_yh*.json
3) 重置 jl/smm_shared_models.json 为 {}

用法：
  conda run -n python_class python .\\reset_mm_json.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from config import settings  # noqa: F401  # 触发 .env 自动加载


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    jl_dir = root / "jl"
    jl_dir.mkdir(parents=True, exist_ok=True)

    removed = 0
    for path in jl_dir.glob("imm_*.json"):
        try:
            path.unlink()
            removed += 1
        except Exception as e:
            print(f"[reset_mm_json] 删除失败 {path.name}: {e}")

    mapping: dict[str, str] = {}
    for key, value in os.environ.items():
        k = str(key or "").strip().lower()
        v = str(value or "").strip()
        if re.fullmatch(r"yh\d+", k) and v:
            mapping[k] = v

    created = 0
    for alias, user_id in sorted(mapping.items(), key=lambda x: x[0]):
        path = jl_dir / f"imm_{alias}.json"
        payload = {
            "user_id": user_id,
            "user_name": alias,
            "professional_background": "",
            "expertise_domains": [],
            "familiar_terms": [],
            "project_understanding": "",
            "unknown_terms": [],
            "known_terms": [],
            "last_confirmed_ts": 0.0,
            "updated_at": 0.0,
        }
        _atomic_write_json(path, payload)
        created += 1

    _atomic_write_json(jl_dir / "smm_shared_models.json", {})

    print(f"[reset_mm_json] 删除IMM文件: {removed}")
    print(f"[reset_mm_json] 新建IMM文件: {created}")
    print("[reset_mm_json] 已重置 SMM: smm_shared_models.json")


if __name__ == "__main__":
    main()
