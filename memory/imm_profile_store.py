"""
IMM profile store.

直接基于 MentalModelMemory 读写画像视图，不保留旧 user_profile 兼容类。
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

from memory.mental_model_memory import MentalModelMemory


class ImmProfileStore:
    def __init__(self, *, mental_model_memory: MentalModelMemory, sql_password: str):
        self.mm = mental_model_memory
        self.sql_password = sql_password

    @staticmethod
    def _clean_list(items: Any, max_items: int = 120) -> list[str]:
        out: list[str] = []
        for item in items or []:
            text = str(item or "").strip()
            if not text or text in out:
                continue
            out.append(text)
            if len(out) >= max_items:
                break
        return out

    def load(self, user_id: str, channel_id: str = "") -> dict | None:
        uid = str(user_id or "").strip()
        if not uid:
            return None
        imm = self.mm.get_imm(user_id=uid)
        if not imm:
            return None
        return {
            "channel_id": str(channel_id or "").strip(),
            "user_id": uid,
            "user_name": str(imm.get("user_name") or "").strip(),
            "major": str(imm.get("professional_background") or "").strip(),
            "research_interests": self._clean_list(imm.get("expertise_domains") or [], max_items=16),
            "methodology": [],
            "keywords": self._clean_list(imm.get("familiar_terms") or [], max_items=16),
            "known_terms": self._clean_list(imm.get("known_terms") or [], max_items=120),
            "updated_at": str(date.today()),
            "last_confirmed_ts": float(imm.get("last_confirmed_ts") or 0.0),
        }

    def save(self, profile: dict, channel_id: str = "") -> None:
        uid = str((profile or {}).get("user_id") or "").strip()
        if not uid:
            raise ValueError("profile.user_id is required")

        patch = {
            "user_name": str((profile or {}).get("user_name") or "").strip(),
            "professional_background": str((profile or {}).get("major") or "").strip(),
            "expertise_domains": self._clean_list((profile or {}).get("research_interests") or [], max_items=80),
            "familiar_terms": self._clean_list((profile or {}).get("keywords") or [], max_items=120),
            "known_terms": self._clean_list((profile or {}).get("known_terms") or [], max_items=120),
            "last_confirmed_ts": float((profile or {}).get("last_confirmed_ts") or 0.0),
            "updated_at": time.time(),
        }
        self.mm.upsert_imm(user_id=uid, patch=patch, user_name=patch["user_name"])

    def load_all(self, channel_id: str = "") -> list[dict]:
        uids: list[str] = []
        for key, value in os.environ.items():
            k = str(key or "").strip().lower()
            v = str(value or "").strip()
            if k.startswith("yh") and v and v not in uids:
                uids.append(v)

        out: list[dict] = []
        for uid in uids:
            row = self.load(uid, channel_id)
            if row:
                out.append(row)
        return out

    def get_known_terms(self, user_id: str, channel_id: str = "") -> list[str]:
        imm = self.mm.get_imm(user_id=str(user_id or "").strip())
        terms = []
        for term in imm.get("known_terms") or []:
            clean = str(term or "").strip().lower()
            if clean and clean not in terms:
                terms.append(clean)
        return terms

    def add_known_term(self, user_id: str, channel_id: str = "", term: str = "") -> bool:
        clean = str(term or "").strip().lower()
        if not clean:
            return False
        existing = self.get_known_terms(user_id=user_id, channel_id=channel_id)
        if clean in existing:
            return False
        self.mm.update_known_terms(user_id=str(user_id or "").strip(), terms=[clean])
        return True

    def mark_profile_confirmed(self, user_id: str, channel_id: str = "", confirmed_ts: float | None = None) -> None:
        self.mm.mark_profile_confirmed(user_id=str(user_id or "").strip(), confirmed_ts=confirmed_ts)

    @staticmethod
    def format_for_prompt(profiles: list[dict]) -> str:
        lines = []
        for p in profiles or []:
            name = p.get("user_name") or p.get("user_id", "未知")
            major = p.get("major") or "未知专业"
            interests = "、".join(p.get("research_interests") or []) or "暂无"
            methods = "、".join(p.get("methodology") or []) or "暂无"
            lines.append(f"用户 {name}：专业为{major}，研究兴趣包括{interests}，擅长{methods}。")
        return "\n".join(lines)
