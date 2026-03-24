"""
mental_model_memory.py

新的心智模型持久化：
1) IMM（Individual Mental Model）：每个用户一个 JSON 文件。
2) SMM（Shared Mental Model）：频道共享一个 JSON 文件。

IMM 字段：
- professional_background
- expertise_domains
- familiar_terms
- project_understanding
- unknown_terms[{term,status,note,updated_at}]
- known_terms
- last_confirmed_ts

SMM 字段：
- current_phase: pre|选题|分工|执行|总结
- phase_status: unsolve|solving|solved
- common_goal
- conflicts[{topic,status,round,note,updated_at}]
- team_cognition
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import io
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except Exception:
    _HAS_PDFPLUMBER = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False


PHASES = ["pre", "选题", "分工", "执行", "总结"]
PHASE_STATUSES = {"unsolve", "solving", "solved"}
UNKNOWN_TERM_STATUSES = {"unsolved", "solving", "solved", "ignore"}
CONFLICT_STATUSES = {"unsolved", "solved"}


def _now() -> float:
    return time.time()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _deep_copy(data: Any) -> Any:
    return json.loads(json.dumps(data, ensure_ascii=False))


def _clean_list(items: Any, max_len: int = 50) -> list[str]:
    out: list[str] = []
    for item in items or []:
        text = _clean_text(item)
        if not text:
            continue
        if text in out:
            continue
        out.append(text)
        if len(out) >= max_len:
            break
    return out


class MentalModelMemory:
    def __init__(self, jl_dir: str = "jl"):
        root = Path(__file__).resolve().parents[1]
        self._jl_dir = root / jl_dir
        self._smm_path = self._jl_dir / "smm_shared_models.json"
        self._lock = threading.Lock()

        self._imm_by_uid: dict[str, dict] = {}
        self._imm_file_map: dict[str, Path] = {}
        self._smm_by_channel: dict[str, dict] = {}

        self._uid_alias_map = self._load_uid_alias_map()
        self._load_all()

    def _load_uid_alias_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for key, value in os.environ.items():
            k = _clean_text(key).lower()
            v = _clean_text(value)
            if re.fullmatch(r"yh\d+", k) and v:
                mapping[v] = k
        return mapping

    def _imm_file_for_user(self, user_id: str) -> Path:
        alias = self._uid_alias_map.get(user_id)
        if alias:
            return self._jl_dir / f"imm_{alias}.json"
        safe_uid = re.sub(r"[^0-9A-Za-z_-]+", "_", user_id)
        return self._jl_dir / f"imm_{safe_uid}.json"

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {}
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[DEBUG][mental_model] 读取JSON失败 {path.name}: {e}")
            return {}

    def _save_json(self, path: Path, data: dict) -> None:
        self._jl_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _default_imm(self, user_id: str, user_name: str = "") -> dict:
        return {
            "user_id": user_id,
            "user_name": user_name,
            "professional_background": "",
            "expertise_domains": [],
            "familiar_terms": [],
            "project_understanding": "",
            "unknown_terms": [],
            "known_terms": [],
            "last_confirmed_ts": 0.0,
            "updated_at": _now(),
        }

    def _default_smm(self, channel_id: str) -> dict:
        return {
            "channel_id": channel_id,
            "current_phase": "pre",
            "phase_status": "solving",
            "common_goal": "",
            "conflicts": [],
            "team_cognition": "两人正在彼此认识",
            "updated_at": _now(),
        }

    def _normalize_unknown_terms(self, items: Any) -> list[dict]:
        out: list[dict] = []
        for item in items or []:
            if isinstance(item, str):
                term = _clean_text(item)
                status = "unsolved"
                note = ""
            elif isinstance(item, dict):
                term = _clean_text(item.get("term"))
                status = _clean_text(item.get("status")).lower() or "unsolved"
                note = _clean_text(item.get("note"))
            else:
                continue

            if not term:
                continue
            if status not in UNKNOWN_TERM_STATUSES:
                status = "unsolved"

            normalized = {
                "term": term,
                "status": status,
                "note": note,
                "updated_at": float(item.get("updated_at") or _now()) if isinstance(item, dict) else _now(),
            }

            replaced = False
            for idx, old in enumerate(out):
                if _clean_text(old.get("term")).lower() == term.lower():
                    out[idx] = normalized
                    replaced = True
                    break
            if not replaced:
                out.append(normalized)
            if len(out) >= 120:
                break
        return out

    def _normalize_imm(self, user_id: str, imm: dict, user_name: str = "") -> dict:
        base = self._default_imm(user_id=user_id, user_name=user_name)
        src = imm if isinstance(imm, dict) else {}

        base["user_name"] = _clean_text(src.get("user_name") or user_name)
        base["professional_background"] = _clean_text(src.get("professional_background"))
        base["expertise_domains"] = _clean_list(src.get("expertise_domains"), max_len=80)
        base["familiar_terms"] = _clean_list(src.get("familiar_terms"), max_len=120)
        base["project_understanding"] = _clean_text(src.get("project_understanding"))
        base["unknown_terms"] = self._normalize_unknown_terms(src.get("unknown_terms") or [])
        base["known_terms"] = _clean_list(src.get("known_terms") or base["familiar_terms"], max_len=120)
        base["last_confirmed_ts"] = float(src.get("last_confirmed_ts") or 0.0)
        base["updated_at"] = float(src.get("updated_at") or _now())
        return base

    def _normalize_smm(self, channel_id: str, smm: dict) -> dict:
        base = self._default_smm(channel_id=channel_id)
        src = smm if isinstance(smm, dict) else {}

        phase = _clean_text(src.get("current_phase"))
        if phase not in PHASES:
            phase = "pre"
        base["current_phase"] = phase

        status = _clean_text(src.get("phase_status")).lower()
        if status not in PHASE_STATUSES:
            status = "solving"
        base["phase_status"] = status

        base["common_goal"] = _clean_text(src.get("common_goal"))
        base["team_cognition"] = _clean_text(src.get("team_cognition") or "两人正在彼此认识")

        conflicts: list[dict] = []
        for item in src.get("conflicts") or []:
            if not isinstance(item, dict):
                continue
            topic = _clean_text(item.get("topic") or item.get("content"))
            if not topic:
                continue
            c_status = _clean_text(item.get("status")).lower() or "unsolved"
            if c_status not in CONFLICT_STATUSES:
                c_status = "unsolved"
            try:
                round_num = int(item.get("round") or 0)
            except Exception:
                round_num = 0
            conflicts.append(
                {
                    "topic": topic,
                    "status": c_status,
                    "round": round_num,
                    "note": _clean_text(item.get("note")),
                    "updated_at": float(item.get("updated_at") or _now()),
                }
            )
            if len(conflicts) >= 120:
                break
        base["conflicts"] = conflicts
        base["updated_at"] = float(src.get("updated_at") or _now())
        return base

    def _flush_imm(self, user_id: str) -> None:
        uid = _clean_text(user_id)
        if not uid:
            return
        path = self._imm_file_map.get(uid) or self._imm_file_for_user(uid)
        self._imm_file_map[uid] = path
        self._save_json(path, self._imm_by_uid.get(uid) or self._default_imm(uid))

    def _flush_smm(self) -> None:
        self._save_json(self._smm_path, self._smm_by_channel)

    def _load_all(self) -> None:
        self._jl_dir.mkdir(parents=True, exist_ok=True)

        self._imm_by_uid = {}
        self._imm_file_map = {}

        for path in sorted(self._jl_dir.glob("imm_*.json")):
            if path.name == "imm_user_models.json":
                continue
            data = self._load_json(path)
            uid = _clean_text(data.get("user_id"))
            if not uid:
                continue
            self._imm_by_uid[uid] = self._normalize_imm(user_id=uid, imm=data)
            self._imm_file_map[uid] = path

        smm_data = self._load_json(self._smm_path)
        normalized_smm: dict[str, dict] = {}
        for cid, smm in smm_data.items():
            channel_id = _clean_text(cid or (smm or {}).get("channel_id"))
            if not channel_id:
                continue
            normalized_smm[channel_id] = self._normalize_smm(channel_id=channel_id, smm=smm)
        self._smm_by_channel = normalized_smm
        self._flush_smm()

    def get_imm(self, user_id: str, user_name: str = "") -> dict:
        uid = _clean_text(user_id)
        with self._lock:
            imm = self._imm_by_uid.get(uid)
            if not imm:
                imm = self._default_imm(user_id=uid, user_name=user_name)
                self._imm_by_uid[uid] = imm
                self._flush_imm(uid)
            elif user_name and not _clean_text(imm.get("user_name")):
                imm["user_name"] = user_name
                imm["updated_at"] = _now()
                self._flush_imm(uid)
            return _deep_copy(imm)

    def get_smm(self, channel_id: str) -> dict:
        cid = _clean_text(channel_id)
        with self._lock:
            smm = self._smm_by_channel.get(cid)
            if not smm:
                smm = self._default_smm(channel_id=cid)
                self._smm_by_channel[cid] = smm
                self._flush_smm()
            return _deep_copy(smm)

    def upsert_imm(self, user_id: str, patch: dict, user_name: str = "") -> dict:
        uid = _clean_text(user_id)
        with self._lock:
            curr = self._imm_by_uid.get(uid) or self._default_imm(uid, user_name=user_name)
            src = patch if isinstance(patch, dict) else {}

            name = _clean_text(src.get("user_name") or user_name)
            if name:
                curr["user_name"] = name

            pb = _clean_text(src.get("professional_background"))
            if pb:
                curr["professional_background"] = pb

            ex = _clean_list(src.get("expertise_domains") or [], max_len=80)
            if ex:
                curr["expertise_domains"] = _clean_list((curr.get("expertise_domains") or []) + ex, max_len=80)

            ft = _clean_list(src.get("familiar_terms") or [], max_len=120)
            if ft:
                curr["familiar_terms"] = _clean_list((curr.get("familiar_terms") or []) + ft, max_len=120)

            pu = _clean_text(src.get("project_understanding"))
            if pu:
                curr["project_understanding"] = pu

            if isinstance(src.get("unknown_terms"), list):
                curr["unknown_terms"] = self._normalize_unknown_terms((curr.get("unknown_terms") or []) + (src.get("unknown_terms") or []))

            kt = _clean_list(src.get("known_terms") or [], max_len=120)
            if kt:
                curr["known_terms"] = _clean_list((curr.get("known_terms") or []) + kt, max_len=120)

            if "last_confirmed_ts" in src:
                try:
                    curr["last_confirmed_ts"] = float(src.get("last_confirmed_ts") or 0.0)
                except Exception:
                    pass

            curr["updated_at"] = _now()
            self._imm_by_uid[uid] = self._normalize_imm(user_id=uid, imm=curr, user_name=user_name)
            self._imm_file_map[uid] = self._imm_file_for_user(uid)
            self._flush_imm(uid)
            return _deep_copy(self._imm_by_uid[uid])

    def update_known_terms(self, user_id: str, terms: list[str]) -> None:
        uid = _clean_text(user_id)
        clean_terms = _clean_list([_clean_text(t).lower() for t in (terms or []) if _clean_text(t)], max_len=120)
        if not clean_terms:
            return
        with self._lock:
            imm = self._imm_by_uid.get(uid) or self._default_imm(uid)
            imm["known_terms"] = _clean_list((imm.get("known_terms") or []) + clean_terms, max_len=120)
            imm["familiar_terms"] = _clean_list((imm.get("familiar_terms") or []) + clean_terms, max_len=120)
            imm["updated_at"] = _now()
            self._imm_by_uid[uid] = self._normalize_imm(user_id=uid, imm=imm)
            self._flush_imm(uid)

    def mark_profile_confirmed(self, user_id: str, confirmed_ts: float | None = None) -> None:
        uid = _clean_text(user_id)
        with self._lock:
            imm = self._imm_by_uid.get(uid) or self._default_imm(uid)
            imm["last_confirmed_ts"] = float(confirmed_ts or _now())
            imm["updated_at"] = _now()
            self._imm_by_uid[uid] = self._normalize_imm(user_id=uid, imm=imm)
            self._flush_imm(uid)

    def _extract_pdf_text(self, pdf_path: Path, max_pages: int = 25) -> str:
        print(
            f"[DEBUG][mental_model] 开始读取PDF {pdf_path.name} "
            f"fitz={_HAS_FITZ} pdfplumber={_HAS_PDFPLUMBER} "
            f"pil={_HAS_PIL} tesseract={_HAS_TESSERACT}"
        )

        if _HAS_FITZ:
            try:
                doc = fitz.open(str(pdf_path))
                count = min(len(doc), max_pages)
                parts = [doc[i].get_text("text") or "" for i in range(count)]
                doc.close()
                txt = "\n".join(parts).strip()
                if txt:
                    print(f"[DEBUG][mental_model] PyMuPDF抽取成功 {pdf_path.name}: chars={len(txt)}")
                    return txt
                print(f"[DEBUG][mental_model] PyMuPDF抽取为空 {pdf_path.name}")
            except Exception as e:
                print(f"[DEBUG][mental_model] PyMuPDF读取失败 {pdf_path.name}: {e}")

        if _HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    count = min(len(pdf.pages), max_pages)
                    parts = [(pdf.pages[i].extract_text() or "") for i in range(count)]
                txt = "\n".join(parts).strip()
                if txt:
                    print(f"[DEBUG][mental_model] pdfplumber抽取成功 {pdf_path.name}: chars={len(txt)}")
                    return txt
                print(f"[DEBUG][mental_model] pdfplumber抽取为空 {pdf_path.name}")
            except Exception as e:
                print(f"[DEBUG][mental_model] pdfplumber读取失败 {pdf_path.name}: {e}")

        # 图片型PDF（扫描件）兜底：若可用则尝试 OCR。
        ocr_txt = self._extract_pdf_text_with_ocr(pdf_path=pdf_path, max_pages=min(max_pages, 5))
        if ocr_txt:
            print(f"[DEBUG][mental_model] OCR抽取成功 {pdf_path.name}: chars={len(ocr_txt)}")
            return ocr_txt

        print(
            f"[DEBUG][mental_model] PDF抽取失败或为空 {pdf_path.name}，"
            "若为扫描件请安装 Pillow+pytesseract，并确认本机已安装 Tesseract OCR"
        )

        return ""

    def _extract_pdf_text_with_ocr(self, pdf_path: Path, max_pages: int = 5) -> str:
        if not (_HAS_FITZ and _HAS_PIL and _HAS_TESSERACT):
            return ""
        try:
            doc = fitz.open(str(pdf_path))
            count = min(len(doc), max_pages)
            parts: list[str] = []
            for i in range(count):
                page = doc[i]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                txt = (pytesseract.image_to_string(img, lang="chi_sim+eng") or "").strip()
                if txt:
                    parts.append(txt)
            doc.close()
            return "\n".join(parts).strip()
        except Exception as e:
            print(f"[DEBUG][mental_model] OCR读取失败 {pdf_path.name}: {e}")
            return ""

    def _split_terms(self, text: str, max_len: int = 20) -> list[str]:
        chunks = re.split(r"[、,，;；/|\n]", _clean_text(text))
        out: list[str] = []
        for chunk in chunks:
            clean = re.sub(r"\s+", "", chunk).strip("。.:：;；,，")
            if len(clean) < 2:
                continue
            if clean not in out:
                out.append(clean)
            if len(out) >= max_len:
                break
        return out

    def _infer_imm_from_resume_text(self, user_id: str, user_name: str, text: str) -> dict:
        compact = re.sub(r"\s+", " ", text or "")
        major = ""

        major_patterns = [
            r"(?:专业|学科|研究方向|研究领域|Major)\s*[:：]\s*([^，。；;\n]{2,48})",
            r"(?:本科|硕士|博士)(?:阶段)?(?:专业)?\s*[:：]?\s*([^，。；;\n]{2,48})",
        ]
        for pattern in major_patterns:
            m = re.search(pattern, compact, flags=re.IGNORECASE)
            if m:
                major = m.group(1).strip()
                break

        terms: list[str] = []
        for hit in re.findall(r"(?:课程|研究兴趣|项目经历|擅长领域)\s*[:：]\s*([^\n]{2,180})", compact, flags=re.IGNORECASE):
            terms.extend(self._split_terms(hit, max_len=20))

        imm = self._default_imm(user_id=user_id, user_name=user_name)
        imm["professional_background"] = major
        imm["expertise_domains"] = _clean_list(terms, max_len=60)
        imm["familiar_terms"] = _clean_list(terms, max_len=100)
        imm["known_terms"] = _clean_list(terms, max_len=120)
        imm["updated_at"] = _now()
        return imm

    def _is_imm_effectively_empty(self, imm: dict | None) -> bool:
        """判断IMM是否为空模板数据（仅在此场景下允许用本地PDF初始化）。"""
        if not isinstance(imm, dict):
            return True

        has_major = bool(_clean_text(imm.get("professional_background")))
        has_expertise = bool(_clean_list(imm.get("expertise_domains") or []))
        has_familiar = bool(_clean_list(imm.get("familiar_terms") or []))
        has_project = bool(_clean_text(imm.get("project_understanding")))
        has_unknown = bool(self._normalize_unknown_terms(imm.get("unknown_terms") or []))
        has_known = bool(_clean_list(imm.get("known_terms") or []))

        return not (has_major or has_expertise or has_familiar or has_project or has_unknown or has_known)

    def bootstrap_imm_from_jl(self, user_name_resolver=None) -> None:
        # 兼容旧逻辑：若提供了 yh* -> user_id 环境变量，优先使用。
        alias_to_uid: dict[str, str] = {}
        for key, value in os.environ.items():
            k = _clean_text(key).lower()
            v = _clean_text(value)
            if re.fullmatch(r"yh\d+", k) and v:
                alias_to_uid[k] = v

        # 启动时固定扫描本地 jl/yh*.pdf，不依赖 Slack 文件或事件。
        local_aliases: list[str] = []
        for pdf_path in sorted(self._jl_dir.glob("yh*.pdf")):
            alias = _clean_text(pdf_path.stem).lower()
            if re.fullmatch(r"yh\d+", alias):
                local_aliases.append(alias)

        if not local_aliases:
            print("[DEBUG][mental_model] 未找到本地 yh*.pdf，跳过IMM初始化")
            return

        # 刷新 user_id -> alias 映射，避免实例化时环境变量尚未就绪导致文件名退化为 imm_<uid>.json。
        for alias, uid in alias_to_uid.items():
            self._uid_alias_map[_clean_text(uid)] = _clean_text(alias)

        print(
            f"[DEBUG][mental_model] 开始IMM初始化，"
            f"本地PDF候选数={len(local_aliases)} env映射数={len(alias_to_uid)}"
        )

        with self._lock:
            for alias in local_aliases:
                file_path = self._jl_dir / f"imm_{alias}.json"
                raw = self._load_json(file_path)
                is_empty_file = self._is_imm_effectively_empty(raw)
                if not is_empty_file:
                    print(f"[DEBUG][mental_model] 跳过初始化 alias={alias}（JSON非空: {file_path.name}）")
                    continue

                # user_id优先取现有json，其次环境变量映射，最后退回alias。
                user_id = _clean_text(raw.get("user_id")) or _clean_text(alias_to_uid.get(alias)) or alias
                curr = self._imm_by_uid.get(user_id)
                if curr and not self._is_imm_effectively_empty(curr):
                    print(f"[DEBUG][mental_model] 跳过初始化 alias={alias} user_id={user_id}（内存中已有非空IMM）")
                    continue

                pdf_path = self._jl_dir / f"{alias}.pdf"
                if not pdf_path.exists():
                    print(f"[DEBUG][mental_model] 跳过初始化 alias={alias}（未找到PDF: {pdf_path.name}）")
                    continue

                text = self._extract_pdf_text(pdf_path)
                if not text:
                    print(f"[DEBUG][mental_model] 跳过初始化 alias={alias}（PDF无可用文本）")
                    continue

                print(f"[DEBUG][mental_model] PDF提取完成 alias={alias} chars={len(text)}")

                user_name = alias
                if callable(user_name_resolver):
                    try:
                        user_name = _clean_text(user_name_resolver(user_id)) or alias
                    except Exception:
                        user_name = alias

                inferred = self._infer_imm_from_resume_text(user_id=user_id, user_name=user_name, text=text)
                if curr:
                    inferred["project_understanding"] = _clean_text(curr.get("project_understanding"))
                    inferred["unknown_terms"] = self._normalize_unknown_terms(curr.get("unknown_terms") or [])
                    inferred["last_confirmed_ts"] = float(curr.get("last_confirmed_ts") or 0.0)

                self._imm_by_uid[user_id] = self._normalize_imm(user_id=user_id, imm=inferred, user_name=user_name)
                self._imm_file_map[user_id] = self._imm_file_for_user(user_id)
                self._flush_imm(user_id)
                major = _clean_text(self._imm_by_uid[user_id].get("professional_background"))
                keywords = len(self._imm_by_uid[user_id].get("familiar_terms") or [])
                print(
                    f"[DEBUG][mental_model] IMM初始化完成 alias={alias} user_id={user_id} "
                    f"major={major!r} familiar_terms={keywords}"
                )

    def _safe_json_obj(self, raw: str) -> dict:
        txt = _clean_text(raw)
        if not txt:
            return {}
        try:
            data = json.loads(txt)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _normalize_phase(self, value: str) -> str:
        s = _clean_text(value)
        alias = {
            "pre": "pre",
            "准备": "pre",
            "选题": "选题",
            "topic": "选题",
            "分工": "分工",
            "division": "分工",
            "执行": "执行",
            "execution": "执行",
            "总结": "总结",
            "summary": "总结",
        }
        phase = alias.get(s, "")
        return phase if phase in PHASES else ""

    def _normalize_phase_status(self, value: str) -> str:
        s = _clean_text(value).lower()
        alias = {
            "unsolve": "unsolve",
            "unsolved": "unsolve",
            "solving": "solving",
            "solved": "solved",
        }
        status = alias.get(s, "")
        return status if status in PHASE_STATUSES else ""

    def _conversation_round(self, convs: str) -> int:
        lines = [line for line in str(convs or "").splitlines() if _clean_text(line)]
        return max(1, len(lines))

    def analyze_and_update(
        self,
        agent: Any,
        channel_id: str,
        user_id: str,
        user_name: str,
        message_text: str,
        convs: str,
    ) -> dict:
        imm = self.get_imm(user_id=user_id, user_name=user_name)
        smm = self.get_smm(channel_id=channel_id)

        prompt = (
            "你是协作心智模型分析器。仅输出JSON。\n"
            "请根据当前消息与上下文，给出IMM增量、SMM增量和回复决策。\n"
            "约束：\n"
            "1) current_phase 只能是 pre|选题|分工|执行|总结\n"
            "2) phase_status 只能是 unsolve|solving|solved\n"
            "3) unknown_terms.status 只能是 unsolved|solving|solved|ignore\n"
            "4) conflicts.status 只能是 unsolved|solved\n"
            "5) response_type 只能是 professional_explain|judgment|knowledge|none\n\n"
            "输出JSON结构：\n"
            "{"
            "\"imm_update\":{"
            "\"professional_background\":\"\","
            "\"expertise_domains_delta\":[],"
            "\"familiar_terms_delta\":[],"
            "\"project_understanding\":\"\","
            "\"unknown_terms_delta\":[{\"term\":\"\",\"status\":\"unsolved|solving|solved|ignore\",\"note\":\"\"}],"
            "\"known_terms_delta\":[],"
            "\"user_name\":\"\""
            "},"
            "\"smm_update\":{"
            "\"current_phase\":\"pre|选题|分工|执行|总结\","
            "\"phase_status\":\"unsolve|solving|solved\","
            "\"common_goal\":\"\","
            "\"team_cognition\":\"\","
            "\"conflict\":{\"topic\":\"\",\"status\":\"unsolved|solved\",\"round\":0,\"note\":\"\"}"
            "},"
            "\"response_decision\":{"
            "\"should_respond\":true,"
            "\"response_type\":\"professional_explain|judgment|knowledge|none\","
            "\"query\":\"\","
            "\"reason\":\"\""
            "}"
            "}\n\n"
            f"当前用户: {user_name} ({user_id})\n"
            f"当前消息: {message_text}\n"
            f"最近对话:\n{convs}\n\n"
            f"IMM现状:\n{json.dumps(imm, ensure_ascii=False)}\n\n"
            f"SMM现状:\n{json.dumps(smm, ensure_ascii=False)}\n"
        )

        try:
            raw = agent.generate_openai_response(prompt)
            parsed = self._safe_json_obj(raw)
        except Exception as e:
            print(f"[DEBUG][mental_model] analyze_and_update LLM失败: {e}")
            parsed = {}

        imm_u = parsed.get("imm_update") if isinstance(parsed.get("imm_update"), dict) else {}
        smm_u = parsed.get("smm_update") if isinstance(parsed.get("smm_update"), dict) else {}
        decision_u = parsed.get("response_decision") if isinstance(parsed.get("response_decision"), dict) else {}

        with self._lock:
            curr_imm = self._imm_by_uid.get(user_id) or self._default_imm(user_id=user_id, user_name=user_name)

            if _clean_text(imm_u.get("user_name")):
                curr_imm["user_name"] = _clean_text(imm_u.get("user_name"))
            elif user_name and not _clean_text(curr_imm.get("user_name")):
                curr_imm["user_name"] = user_name

            pb = _clean_text(imm_u.get("professional_background"))
            if pb:
                curr_imm["professional_background"] = pb

            ex_delta = _clean_list(imm_u.get("expertise_domains_delta") or [], max_len=80)
            if ex_delta:
                curr_imm["expertise_domains"] = _clean_list((curr_imm.get("expertise_domains") or []) + ex_delta, max_len=80)

            ft_delta = _clean_list(imm_u.get("familiar_terms_delta") or [], max_len=120)
            if ft_delta:
                curr_imm["familiar_terms"] = _clean_list((curr_imm.get("familiar_terms") or []) + ft_delta, max_len=120)

            kt_delta = _clean_list(imm_u.get("known_terms_delta") or [], max_len=120)
            if kt_delta or ft_delta:
                curr_imm["known_terms"] = _clean_list((curr_imm.get("known_terms") or []) + kt_delta + ft_delta, max_len=120)

            pu = _clean_text(imm_u.get("project_understanding"))
            if pu:
                curr_imm["project_understanding"] = pu

            if isinstance(imm_u.get("unknown_terms_delta"), list):
                curr_imm["unknown_terms"] = self._normalize_unknown_terms(
                    (curr_imm.get("unknown_terms") or []) + list(imm_u.get("unknown_terms_delta") or [])
                )

            curr_imm["updated_at"] = _now()
            self._imm_by_uid[user_id] = self._normalize_imm(user_id=user_id, imm=curr_imm, user_name=user_name)
            self._imm_file_map[user_id] = self._imm_file_for_user(user_id)
            self._flush_imm(user_id)

            curr_smm = self._smm_by_channel.get(channel_id) or self._default_smm(channel_id=channel_id)
            prev_phase = _clean_text(curr_smm.get("current_phase") or "pre")

            phase = self._normalize_phase(smm_u.get("current_phase")) or prev_phase
            phase_status = self._normalize_phase_status(smm_u.get("phase_status")) or _clean_text(curr_smm.get("phase_status") or "solving")
            if phase_status not in PHASE_STATUSES:
                phase_status = "solving"

            goal = _clean_text(smm_u.get("common_goal"))
            if goal:
                curr_smm["common_goal"] = goal

            cognition = _clean_text(smm_u.get("team_cognition"))
            if cognition:
                curr_smm["team_cognition"] = cognition

            conflict = smm_u.get("conflict") if isinstance(smm_u.get("conflict"), dict) else {}
            topic = _clean_text(conflict.get("topic"))
            if topic:
                c_status = _clean_text(conflict.get("status")).lower() or "unsolved"
                if c_status not in CONFLICT_STATUSES:
                    c_status = "unsolved"
                try:
                    round_num = int(conflict.get("round") or self._conversation_round(convs))
                except Exception:
                    round_num = self._conversation_round(convs)
                entry = {
                    "topic": topic,
                    "status": c_status,
                    "round": round_num,
                    "note": _clean_text(conflict.get("note")),
                    "updated_at": _now(),
                }
                conflicts = list(curr_smm.get("conflicts") or [])
                replaced = False
                for idx, old in enumerate(conflicts):
                    if _clean_text(old.get("topic")).lower() == topic.lower():
                        conflicts[idx] = entry
                        replaced = True
                        break
                if not replaced:
                    conflicts.append(entry)
                curr_smm["conflicts"] = conflicts[:120]

            curr_smm["current_phase"] = phase
            curr_smm["phase_status"] = phase_status
            curr_smm["updated_at"] = _now()
            self._smm_by_channel[channel_id] = self._normalize_smm(channel_id=channel_id, smm=curr_smm)
            self._flush_smm()

            new_phase = _clean_text(self._smm_by_channel[channel_id].get("current_phase") or phase)
            transition = {
                "changed": bool(new_phase != prev_phase),
                "from_phase": prev_phase,
                "to_phase": new_phase,
            }

        decision = {
            "should_respond": bool(decision_u.get("should_respond")),
            "response_type": _clean_text(decision_u.get("response_type")) or "none",
            "query": _clean_text(decision_u.get("query")),
            "reason": _clean_text(decision_u.get("reason")),
        }
        if decision["response_type"] not in {"professional_explain", "judgment", "knowledge", "none"}:
            decision["response_type"] = "none"
        if decision["response_type"] == "none":
            decision["should_respond"] = False

        return {
            "imm": self.get_imm(user_id=user_id, user_name=user_name),
            "smm": self.get_smm(channel_id=channel_id),
            "decision": decision,
            "smm_transition": transition,
        }
