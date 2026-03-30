"""
mental_model_memory.py

新的心智模型持久化：
1) IMM（Individual Mental Model）：每个用户一个 JSON 文件，采用中文分区结构。
2) SMM（Shared Mental Model）：频道共享一个 JSON 文件，采用任务生命周期/共识区/冲突区结构。

注意：
- 运行时返回数据保留旧字段投影（兼容既有调用方），落盘以新结构为准。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
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


PHASES = ["破冰", "选题", "分工", "实践", "写作"]
PHASE_STATUSES = {"未解决", "解决中", "已解决"}
UNKNOWN_TERM_STATUSES = {"未解决", "解决中", "已解决"}
CONFLICT_STATUSES = {"未解决", "解决中", "已解决"}

UNKNOWN_TERM_TIMEOUT_SECONDS = 10
CONFLICT_TIMEOUT_SECONDS = 10
TOPIC_STAGE_TIMEOUT_MINUTES = 5
DIVISION_STAGE_TIMEOUT_MINUTES = 10


def _now() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _seconds_since_iso(ts: str) -> int:
    text = _clean_text(ts)
    if not text:
        return 0
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0, int((now - dt).total_seconds()))
    except Exception:
        return 0


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


def _preview_text(text: str, max_len: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def _looks_like_confusion(text: str) -> bool:
    content = _normalize_ocr_spacing(text).lower()
    if not content:
        return False
    cues = (
        "什么意思", "是什么", "不懂", "看不懂", "怎么理解", "解释", "不太明白",
        "what is", "what's", "mean", "confused",
    )
    return any(c in content for c in cues)


def _looks_like_summary_request(text: str) -> bool:
    content = _normalize_ocr_spacing(text)
    if not content:
        return False
    cues = ("总结", "归纳", "复盘", "梳理", "回顾")
    return any(c in content for c in cues)


def _looks_like_topic_stall(text: str) -> bool:
    content = _normalize_ocr_spacing(text)
    if not content:
        return False
    topic_cues = ("选题", "题目", "方向", "研究什么", "做什么")
    stall_cues = ("没思路", "不知道", "想不出", "卡住", "没进展", "没有方向", "没想法")
    return any(c in content for c in topic_cues) and any(c in content for c in stall_cues)


def _looks_like_progress_stall(text: str) -> bool:
    """通用推进停滞信号：用于把 SMM.phase_status 标记为 unsolve。"""
    content = _normalize_ocr_spacing(text)
    if not content:
        return False
    stall_cues = (
        "没思路", "没有思路", "不知道", "想不出", "卡住", "没进展", "推进不动", "没有方向", "没想法", "没有想法", "不清楚", "不会",
    )
    return any(c in content for c in stall_cues)


def _normalize_ocr_spacing(text: str) -> str:
    value = str(text or "")
    # OCR 常把中文逐字断开，这里先合并中文字符间空格，再规整其他空白。
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _strip_score_chunks(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"GPA\s*[:：]?\s*\d(?:\.\d+)?\s*/\s*\d(?:\.\d+)?", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[（(]?\s*\d{2,3}(?:\.\d+)?\s*(?:/\s*100)?\s*[)）]?", "", value)
    value = re.sub(r"\b\d{4}[.-]\d{1,2}\s*[-~至]\s*\d{4}[.-]\d{1,2}\b", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_major_candidate(text: str) -> str:
    value = _normalize_ocr_spacing(_strip_score_chunks(text))
    for cue in ("加权", "平均分", "核心课程", "荣誉", "获奖", "实习", "学术成果", "论文", "课程"):
        if cue in value:
            value = value.split(cue, 1)[0].strip()
    value = re.sub(r"^[^\u4e00-\u9fffA-Za-z]+", "", value)
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z]+$", "", value)
    major_cues = ("专业", "法学", "科学", "工程", "管理", "经济", "信息", "智能", "计算机", "地理")
    if not any(c in value for c in major_cues):
        return ""
    if len(value) > 36:
        value = value[:36].strip()
    digits = len(re.findall(r"\d", value))
    if digits >= 3:
        return ""
    return value


def _clean_term_candidate(text: str) -> str:
    value = _normalize_ocr_spacing(_strip_score_chunks(text))
    value = re.sub(r"\bAl\b", "AI", value, flags=re.IGNORECASE)
    for cue in ("荣誉", "获奖", "奖学金", "平均分", "GPA", "实习", "项目经历", "学术成果", "论文"):
        if cue in value and len(value) > 8:
            value = value.split(cue, 1)[0].strip()
    value = re.sub(r"[\"'`<>《》【】\[\]{}]", "", value)
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9+\-/ ]+", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .,:;，。；：、-_")
    if len(value) < 2 or len(value) > 32:
        return ""
    if len(re.findall(r"\d", value)) >= max(3, int(len(value) * 0.4)):
        return ""
    return value


def _is_non_academic_term(term: str) -> bool:
    t = str(term or "")
    noise_cues = (
        "荣誉", "获奖", "优秀", "志愿者", "三好学生", "暑期", "公众号", "小红书", "篮球", "办公软件",
        "Office", "CET", "运营", "秀米", "可画", "语言水平", "学术成果", "SCI", "一作", "送审",
        "帮助用户", "围绕真实场景", "梳理核心链路", "彻底打通", "一键分析", "守住产品", "放弃激进",
        "设计极简交互", "构建规则", "混合架构", "确保AI指令",
        "支持多平台", "需求与场景设计", "和矩阵",
    )
    return any(c in t for c in noise_cues)


def _is_academic_term(term: str) -> bool:
    t = str(term or "")
    if not t or _is_non_academic_term(t):
        return False
    cues = (
        "法", "人工智能", "机器学习", "深度学习", "计算机", "数据", "算法", "信息", "检索", "推荐", "模型",
        "科学", "工程", "地理", "GIS", "NLP",
    )
    return any(c in t for c in cues)


def _is_skill_term(term: str) -> bool:
    t = str(term or "")
    skills = (
        "Python", "LangChain", "RAG", "模型微调", "Prompt Engineering", "Agent Skills",
        "n8n", "Coze", "Dify", "Figma", "Xmind",
    )
    return any(s in t for s in skills)


def _is_domain_like(term: str) -> bool:
    t = str(term or "")
    cues = (
        "法学", "法律", "人工智能", "机器学习", "深度学习", "计算机", "数据", "算法", "信息",
        "自然语言", "检索", "推荐", "风控", "金融", "管理", "地理", "科学", "工程", "系统",
    )
    if any(c in t for c in cues):
        return True
    return t.endswith(("学", "科学", "工程", "方向", "技术"))


def _sanitize_terms(items: Any, max_len: int, academic_only: bool = False) -> list[str]:
    out: list[str] = []
    for item in items or []:
        clean = _clean_term_candidate(item)
        if _is_non_academic_term(clean):
            continue
        if len(clean) > 16 and not (_is_domain_like(clean) or _is_skill_term(clean)):
            continue
        if academic_only and not _is_academic_term(clean):
            continue
        if not clean or clean in out:
            continue
        out.append(clean)
        if len(out) >= max_len:
            break
    return out


def _guess_major_from_terms(terms: list[str]) -> str:
    joined = " ".join(terms or [])
    if any(k in joined for k in ("法学", "公司法", "经济法", "财税法", "证券法", "法律")):
        return "法学"
    if any(k in joined for k in ("人工智能", "机器学习", "深度学习", "计算机", "算法", "NLP", "AI")):
        return "人工智能/计算机科学"
    if any(k in joined for k in ("地理信息", "GIS", "遥感", "空间数据")):
        return "地理信息科学"
    return ""


def _clean_project_name_candidate(text: str) -> str:
    value = _normalize_ocr_spacing(_strip_score_chunks(text))
    value = re.sub(r"^[^\u4e00-\u9fffA-Za-z0-9]+", "", value)
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+$", "", value)
    value = re.sub(r"\d+(?:[.\-]+\d*)+$", "", value).strip()
    value = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z])\d+$", "", value).strip()
    value = re.sub(r"\s+", " ", value).strip("。；;，,：:-")
    if not value:
        return ""

    reject_exact = {"项目经历", "项目描述", "研究经历", "课题经历"}
    if value in reject_exact:
        return ""

    reject_cues = (
        "项目描述", "核心聚焦", "功能与数据方案", "主导", "负责", "协助", "深入检索",
        "研究报告", "应用程序和网站", "业务闭环设计", "梳理", "定义", "两个课题",
        "本科", "研究生", "博士", "硕士", "荣誉", "奖学金", "英语水平", "CET", "GPA",
        "武汉大学", "复旦大学", "专业", "学年", "三好学生", "优秀学生", "志愿者",
    )
    if any(c in value for c in reject_cues):
        return ""

    project_cues = (
        "系统", "平台", "项目", "课题", "实习", "案例",
        "法官助理", "算法工程师", "开发者", "产品", "决策助手",
        "人民法院", "检察院", "美的集团", "赛博淘客",
    )
    has_project_cue = any(c in value for c in project_cues)
    has_year = bool(re.search(r"20\d{2}", value))
    has_sep = ("-" in value) or ("—" in value)
    if not has_project_cue:
        return ""
    if has_year and not has_project_cue:
        return ""
    if re.match(r"^[0-9Oo]+", value):
        return ""

    if len(value) < 4:
        return ""
    if len(value) > 46:
        value = _preview_text(value, max_len=46)
    return value


def _sanitize_project_items(items: Any, max_len: int = 12) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        raw_name = _clean_text(item.get("项目名称"))
        clean_name = _clean_project_name_candidate(raw_name)
        if not clean_name:
            continue
        if any(c in clean_name for c in ("本科", "研究生", "荣誉", "奖学金", "英语水平", "专业")):
            continue
        key = clean_name.lower()
        if key in seen:
            continue
        seen.add(key)
        effect = _clean_text(item.get("实现效果"))
        if any(c in effect for c in ("本科", "研究生", "荣誉", "奖学金", "英语水平", "专业")):
            effect = ""
        effect = re.sub(r"[©”“]+", "", effect).strip()
        if len(effect) > 220:
            effect = effect[:220].rstrip("，,；;")
        out.append(
            {
                "项目名称": clean_name,
                "采用技术": _clean_list(item.get("采用技术") or [], max_len=20),
                "实现效果": effect or "参与项目实施与优化，具体量化效果待补充",
            }
        )
        if len(out) >= max_len:
            break
    return [_polish_project_item(x) for x in out]


def _polish_project_name(name: str) -> str:
    n = _clean_text(name)
    if not n:
        return ""
    n = re.sub(r"\s+", " ", n)
    # 常见表达统一
    n = n.replace("智能决策系统", "智能决策支持系统")
    n = n.replace("独立开发", "独立研发")
    # 去除冗余后缀
    n = re.sub(r"(?:项目核心设计者|产品核心设计者)$", "产品核心设计", n).strip(" -")
    return n


def _polish_project_techniques(techniques: list[str], project_name: str, effect: str) -> list[str]:
    base = _clean_list(techniques or [], max_len=12)
    corpus = f"{project_name} {effect}".lower()
    add_map = [
        ("rag", "RAG"),
        ("embedding", "Embedding模型微调"),
        ("langchain", "LangChain"),
        ("python", "Python"),
        ("检索", "检索增强生成"),
        ("法官", "法律检索与案例分析"),
        ("审理", "法律文书撰写"),
        ("法院", "争议焦点提取"),
    ]
    for cue, label in add_map:
        if cue in corpus and label not in base:
            base.append(label)
    return _clean_list(base, max_len=8)


def _polish_project_effect(effect: str, project_name: str) -> str:
    e = _clean_text(effect)
    if not e:
        # 基于项目类型给出可接受的通用结果句。
        if any(k in project_name for k in ("法官助理", "法院", "检察")):
            return "参与案件材料梳理与法律文书撰写，支持类案检索与审理分析"
        if any(k in project_name.lower() for k in ("rag", "算法", "决策", "系统", "研发")):
            return "完成方案设计与实现验证，支持业务场景的智能分析与决策优化"
        return "参与项目实施与优化，具体量化效果待补充"

    e = re.sub(r"\s+", " ", e).strip("，,；;")
    if len(e) > 180:
        e = e[:180].rstrip("，,；;")
    return e


def _polish_project_item(item: dict) -> dict:
    name = _polish_project_name(_clean_text(item.get("项目名称")))
    effect = _polish_project_effect(_clean_text(item.get("实现效果")), name)
    techniques = _polish_project_techniques(item.get("采用技术") or [], project_name=name, effect=effect)
    return {
        "项目名称": name,
        "采用技术": techniques,
        "实现效果": effect,
    }


def _extract_techniques_from_block(block: str, familiar_terms: list[str], max_len: int = 8) -> list[str]:
    text = _normalize_ocr_spacing(block)
    out: list[str] = []

    # 1) 优先从“技术/方法/工具”显式字段中抓取。
    explicit_hits = re.findall(r"(?:技术栈|采用技术|方法|工具|能力)\s*[:：]\s*([^\n]{2,120})", text)
    for hit in explicit_hits:
        for chunk in re.split(r"[、,，;；/|]", hit):
            cand = _clean_term_candidate(chunk)
            if not cand:
                continue
            if cand not in out:
                out.append(cand)
            if len(out) >= max_len:
                return out

    # 2) 再从已识别术语中按出现匹配。
    lower_block = text.lower()
    for term in familiar_terms or []:
        t = _clean_text(term)
        if not t:
            continue
        if t.lower() in lower_block and t not in out:
            out.append(t)
        if len(out) >= max_len:
            return out

    # 3) 最后补充通用技术词。
    generic_terms = [
        "法律文本分析", "案例分析", "数据整理与统计分析", "法律文书撰写",
        "法律检索与案例分析", "争议焦点提取", "复杂案件分析",
        "国际法分析", "法律推理", "团队协作与策略设计",
        "政策分析", "制度设计", "学术研究方法",
        "Python", "LangChain", "RAG", "向量数据库", "评测", "模型微调",
        "Embedding", "Embedding模型微调", "检索增强生成", "提示词工程",
    ]
    for t in generic_terms:
        if t.lower() in lower_block and t not in out:
            out.append(t)
        if len(out) >= max_len:
            break
    return out


def _extract_effect_from_block(block: str, max_len: int = 160) -> str:
    text = _normalize_ocr_spacing(block)
    if not text:
        return ""
    sentences = re.split(r"[。；;\n]", text)
    keep: list[str] = []
    cues = ("提升", "完成", "参与", "负责", "实现", "形成", "输出", "获得", "落地", "优化", "支持")
    for s in sentences:
        sent = _clean_text(s)
        if len(sent) < 6:
            continue
        if any(c in sent for c in cues) or re.search(r"\d+", sent):
            keep.append(sent)
        if len(keep) >= 2:
            break
    merged = "；".join(keep)
    if len(merged) > max_len:
        merged = merged[:max_len].rstrip("，,；;")
    return merged


def _extract_project_items_from_resume(text: str, familiar_terms: list[str], max_len: int = 8) -> list[dict]:
    lines = [_normalize_ocr_spacing(x) for x in re.split(r"[\r\n]+", str(text or ""))]
    lines = [x for x in lines if x]

    items: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not any(c in line for c in (
            "项目", "系统", "平台", "课题", "实习", "案例",
            "法官助理", "算法工程师", "开发者", "人民法院", "美的集团", "赛博淘客",
        )):
            i += 1
            continue
        name = _clean_project_name_candidate(line)
        if not name:
            i += 1
            continue

        block_lines = [line]
        j = i + 1
        while j < len(lines) and len(block_lines) < 5:
            nxt = lines[j]
            if _clean_project_name_candidate(nxt):
                break
            block_lines.append(nxt)
            j += 1

        block = "\n".join(block_lines)
        items.append(
            {
                "项目名称": name,
                "采用技术": _extract_techniques_from_block(block, familiar_terms=familiar_terms, max_len=8),
                "实现效果": _extract_effect_from_block(block, max_len=180),
            }
        )
        if len(items) >= max_len:
            break
        i = j

    return _sanitize_project_items(items, max_len=max_len)


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
            "个人画像": {
                "姓名": user_name,
                "专业领域": "",
                "核心专长": [],
                "历史项目经验": [],
            },
            "个人领域知识库": {
                "提取术语": [],
                "术语解释": [],
            },
            "认知盲区 (未涉及知识)": [],
            "个人任务认知 (Task Stance)": {
                "期望研究方向": "",
                "提议研究方法": "",
                "预期实验流程": "",
            },
            "last_confirmed_ts": 0.0,
            "updated_at": _now(),
        }

    def _default_smm(self, channel_id: str) -> dict:
        return {
            "channel_id": channel_id,
            "任务生命周期": {
                "当前所处阶段": "破冰",
                "阶段进入时间": "",
                "阶段停留时长_分钟": 0,
            },
            "团队共识区 (Shared Consensus)": {
                "已确认方向": "",
                "已确认方法": None,
                "已确认分工": [],
            },
            "团队冲突区 (Conflict Zone)": [],
            "updated_at": _now(),
        }

    def _legacy_phase_to_cn(self, value: Any) -> str:
        s = _clean_text(value).lower()
        mapping = {
            "pre": "破冰",
            "准备": "破冰",
            "破冰": "破冰",
            "选题": "选题",
            "topic": "选题",
            "分工": "分工",
            "division": "分工",
            "执行": "实践",
            "execution": "实践",
            "实践": "实践",
            "总结": "写作",
            "summary": "写作",
            "写作": "写作",
        }
        return mapping.get(s, "")

    def _legacy_status_to_cn(self, value: Any, fallback: str = "未解决") -> str:
        s = _clean_text(value).lower()
        mapping = {
            "unsolve": "未解决",
            "unsolved": "未解决",
            "未解决": "未解决",
            "solving": "解决中",
            "解决中": "解决中",
            "solved": "已解决",
            "已解决": "已解决",
            "ignore": "已解决",
        }
        return mapping.get(s, fallback)

    def _cn_status_to_legacy(self, value: Any, fallback: str = "unsolve") -> str:
        s = _clean_text(value)
        mapping = {
            "未解决": "unsolve",
            "解决中": "solving",
            "已解决": "solved",
            "unsolved": "unsolve",
            "unsolve": "unsolve",
            "solving": "solving",
            "solved": "solved",
        }
        return mapping.get(s, fallback)

    def _cn_phase_to_legacy(self, value: Any) -> str:
        s = _clean_text(value)
        mapping = {
            "破冰": "pre",
            "选题": "选题",
            "分工": "分工",
            "实践": "执行",
            "写作": "总结",
            "pre": "pre",
            "执行": "执行",
            "总结": "总结",
        }
        return mapping.get(s, "pre")

    def _normalize_unknown_terms(self, items: Any) -> list[dict]:
        out: list[dict] = []
        for item in items or []:
            if isinstance(item, str):
                term = _clean_text(item)
                status = "未解决"
                note = ""
            elif isinstance(item, dict):
                term = _clean_text(item.get("未知术语") or item.get("term"))
                status = self._legacy_status_to_cn(item.get("当前状态") or item.get("status"), fallback="未解决")
                note = _clean_text(item.get("note"))
            else:
                continue

            if not term:
                continue
            if status not in UNKNOWN_TERM_STATUSES:
                status = "未解决"

            normalized = {
                "未知术语": term,
                "当前状态": status,
                "触发时间戳": _clean_text(item.get("触发时间戳")) if isinstance(item, dict) else "",
                "持续时长_秒": int(float(item.get("持续时长_秒") or 0)) if isinstance(item, dict) else 0,
                "note": note,
            }

            replaced = False
            for idx, old in enumerate(out):
                if _clean_text(old.get("未知术语")).lower() == term.lower():
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

        name = _clean_text(src.get("user_name") or user_name)
        profile = src.get("个人画像") if isinstance(src.get("个人画像"), dict) else {}
        knowledge = src.get("个人领域知识库") if isinstance(src.get("个人领域知识库"), dict) else {}
        task_stance = src.get("个人任务认知 (Task Stance)") if isinstance(src.get("个人任务认知 (Task Stance)"), dict) else {}

        major = _clean_major_candidate(profile.get("专业领域") or src.get("professional_background"))
        core = _sanitize_terms(profile.get("核心专长") or src.get("expertise_domains"), max_len=80, academic_only=False)
        terms = _sanitize_terms(knowledge.get("提取术语") or src.get("familiar_terms"), max_len=120)
        term_desc = _clean_list(knowledge.get("术语解释") or [], max_len=120)
        unknown_terms = self._normalize_unknown_terms(src.get("认知盲区 (未涉及知识)") or src.get("unknown_terms") or [])

        projects = profile.get("历史项目经验") if isinstance(profile.get("历史项目经验"), list) else []
        clean_projects = _sanitize_project_items(projects, max_len=20)
        if clean_projects:
            fallback_terms = _clean_list(core + terms, max_len=8)
            for row in clean_projects:
                if not (row.get("采用技术") or []):
                    row["采用技术"] = fallback_terms[:4]

        if not major:
            major = _guess_major_from_terms(terms)

        base["user_name"] = name
        base["个人画像"] = {
            "姓名": _clean_text(profile.get("姓名") or name),
            "专业领域": major,
            "核心专长": core,
            "历史项目经验": clean_projects,
        }
        base["个人领域知识库"] = {
            "提取术语": terms,
            "术语解释": term_desc,
        }
        base["认知盲区 (未涉及知识)"] = unknown_terms
        base["个人任务认知 (Task Stance)"] = {
            "期望研究方向": _clean_text(task_stance.get("期望研究方向") or src.get("project_understanding")),
            "提议研究方法": _clean_text(task_stance.get("提议研究方法")),
            "预期实验流程": _clean_text(task_stance.get("预期实验流程")),
        }
        base["last_confirmed_ts"] = float(src.get("last_confirmed_ts") or 0.0)
        base["updated_at"] = float(src.get("updated_at") or _now())
        return base

    def _normalize_smm(self, channel_id: str, smm: dict) -> dict:
        base = self._default_smm(channel_id=channel_id)
        src = smm if isinstance(smm, dict) else {}

        life = src.get("任务生命周期") if isinstance(src.get("任务生命周期"), dict) else {}
        consensus = src.get("团队共识区 (Shared Consensus)") if isinstance(src.get("团队共识区 (Shared Consensus)"), dict) else {}

        phase = self._legacy_phase_to_cn(life.get("当前所处阶段") or src.get("current_phase")) or "破冰"
        if phase not in PHASES:
            phase = "破冰"

        phase_enter = _clean_text(life.get("阶段进入时间"))
        try:
            phase_stay = int(float(life.get("阶段停留时长_分钟") or 0))
        except Exception:
            phase_stay = 0

        confirmed_goal = _clean_text(consensus.get("已确认方向") or src.get("common_goal"))
        confirmed_method = consensus.get("已确认方法")
        if confirmed_method is not None:
            confirmed_method = _clean_text(confirmed_method) or None
        confirmed_division = _clean_list(consensus.get("已确认分工") or [], max_len=120)

        status = self._legacy_status_to_cn(src.get("phase_status"), fallback="解决中")
        if status not in PHASE_STATUSES:
            status = "解决中"

        conflicts: list[dict] = []
        for item in (src.get("团队冲突区 (Conflict Zone)") or src.get("conflicts") or []):
            if not isinstance(item, dict):
                continue
            topic = _clean_text(item.get("冲突描述") or item.get("topic") or item.get("content"))
            if not topic:
                continue
            c_status = self._legacy_status_to_cn(item.get("当前状态") or item.get("status"), fallback="未解决")
            if c_status not in CONFLICT_STATUSES:
                c_status = "未解决"
            conflicts.append(
                {
                    "冲突描述": topic,
                    "当前状态": c_status,
                    "触发时间戳": _clean_text(item.get("触发时间戳")),
                    "持续时长_秒": int(float(item.get("持续时长_秒") or 0)),
                    "note": _clean_text(item.get("note")),
                }
            )
            if len(conflicts) >= 120:
                break

        base["任务生命周期"] = {
            "当前所处阶段": phase,
            "阶段进入时间": phase_enter,
            "阶段停留时长_分钟": phase_stay,
        }
        base["团队共识区 (Shared Consensus)"] = {
            "已确认方向": confirmed_goal,
            "已确认方法": confirmed_method,
            "已确认分工": confirmed_division,
        }
        base["团队冲突区 (Conflict Zone)"] = conflicts
        base["phase_status"] = status
        base["updated_at"] = float(src.get("updated_at") or _now())
        return base

    def _imm_with_legacy_projection(self, imm: dict) -> dict:
        out = _deep_copy(imm)
        profile = out.get("个人画像") if isinstance(out.get("个人画像"), dict) else {}
        knowledge = out.get("个人领域知识库") if isinstance(out.get("个人领域知识库"), dict) else {}
        stance = out.get("个人任务认知 (Task Stance)") if isinstance(out.get("个人任务认知 (Task Stance)"), dict) else {}
        unknowns = out.get("认知盲区 (未涉及知识)") if isinstance(out.get("认知盲区 (未涉及知识)"), list) else []

        out["professional_background"] = _clean_text(profile.get("专业领域"))
        out["expertise_domains"] = _clean_list(profile.get("核心专长") or [], max_len=80)
        out["familiar_terms"] = _clean_list(knowledge.get("提取术语") or [], max_len=120)
        out["known_terms"] = _clean_list(knowledge.get("提取术语") or [], max_len=120)
        out["project_understanding"] = _clean_text(stance.get("期望研究方向"))
        out["unknown_terms"] = [
            {
                "term": _clean_text(x.get("未知术语")),
                "status": self._cn_status_to_legacy(x.get("当前状态"), fallback="unsolve"),
                "note": _clean_text(x.get("note")),
                "updated_at": _now(),
            }
            for x in unknowns if isinstance(x, dict)
        ]
        return out

    def _smm_with_legacy_projection(self, smm: dict) -> dict:
        out = _deep_copy(smm)
        life = out.get("任务生命周期") if isinstance(out.get("任务生命周期"), dict) else {}
        consensus = out.get("团队共识区 (Shared Consensus)") if isinstance(out.get("团队共识区 (Shared Consensus)"), dict) else {}
        conflicts = out.get("团队冲突区 (Conflict Zone)") if isinstance(out.get("团队冲突区 (Conflict Zone)"), list) else []

        out["current_phase"] = self._cn_phase_to_legacy(life.get("当前所处阶段"))
        out["phase_status"] = self._cn_status_to_legacy(out.get("phase_status"), fallback="solving")
        out["common_goal"] = _clean_text(consensus.get("已确认方向"))
        out["team_cognition"] = _clean_text(consensus.get("已确认方法") or "")
        out["conflicts"] = [
            {
                "topic": _clean_text(c.get("冲突描述")),
                "status": self._cn_status_to_legacy(c.get("当前状态"), fallback="unsolve"),
                "round": 0,
                "note": _clean_text(c.get("note")),
                "updated_at": _now(),
            }
            for c in conflicts if isinstance(c, dict)
        ]
        return out

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
            normalized = self._normalize_imm(user_id=uid, imm=data)
            self._imm_by_uid[uid] = normalized
            self._imm_file_map[uid] = path
            if normalized != data:
                self._save_json(path, normalized)

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
            return self._imm_with_legacy_projection(imm)

    def get_smm(self, channel_id: str) -> dict:
        cid = _clean_text(channel_id)
        with self._lock:
            smm = self._smm_by_channel.get(cid)
            if not smm:
                smm = self._default_smm(channel_id=cid)
                self._smm_by_channel[cid] = smm
                self._flush_smm()
            return self._smm_with_legacy_projection(smm)

    def upsert_imm(self, user_id: str, patch: dict, user_name: str = "") -> dict:
        uid = _clean_text(user_id)
        with self._lock:
            curr = self._imm_by_uid.get(uid) or self._default_imm(uid, user_name=user_name)
            src = patch if isinstance(patch, dict) else {}

            name = _clean_text(src.get("user_name") or user_name)
            if name:
                curr["user_name"] = name

            profile_u = src.get("个人画像") if isinstance(src.get("个人画像"), dict) else {}
            kb_u = src.get("个人领域知识库") if isinstance(src.get("个人领域知识库"), dict) else {}
            stance_u = src.get("个人任务认知 (Task Stance)") if isinstance(src.get("个人任务认知 (Task Stance)"), dict) else {}

            pb = _clean_text(profile_u.get("专业领域") or src.get("professional_background"))
            if pb:
                profile = curr.get("个人画像") if isinstance(curr.get("个人画像"), dict) else {}
                profile["专业领域"] = pb
                curr["个人画像"] = profile

            ex = _clean_list(profile_u.get("核心专长") or src.get("expertise_domains") or [], max_len=80)
            if ex:
                profile = curr.get("个人画像") if isinstance(curr.get("个人画像"), dict) else {}
                profile["核心专长"] = _clean_list((profile.get("核心专长") or []) + ex, max_len=80)
                curr["个人画像"] = profile

            ft = _clean_list(kb_u.get("提取术语") or src.get("familiar_terms") or [], max_len=120)
            if ft:
                kb = curr.get("个人领域知识库") if isinstance(curr.get("个人领域知识库"), dict) else {}
                kb["提取术语"] = _clean_list((kb.get("提取术语") or []) + ft, max_len=120)
                curr["个人领域知识库"] = kb

            pu = _clean_text(stance_u.get("期望研究方向") or src.get("project_understanding"))
            if pu:
                stance = curr.get("个人任务认知 (Task Stance)") if isinstance(curr.get("个人任务认知 (Task Stance)"), dict) else {}
                stance["期望研究方向"] = pu
                curr["个人任务认知 (Task Stance)"] = stance

            method = _clean_text(stance_u.get("提议研究方法"))
            if method:
                stance = curr.get("个人任务认知 (Task Stance)") if isinstance(curr.get("个人任务认知 (Task Stance)"), dict) else {}
                stance["提议研究方法"] = method
                curr["个人任务认知 (Task Stance)"] = stance

            pipeline = _clean_text(stance_u.get("预期实验流程"))
            if pipeline:
                stance = curr.get("个人任务认知 (Task Stance)") if isinstance(curr.get("个人任务认知 (Task Stance)"), dict) else {}
                stance["预期实验流程"] = pipeline
                curr["个人任务认知 (Task Stance)"] = stance

            if isinstance(src.get("认知盲区 (未涉及知识)"), list) or isinstance(src.get("unknown_terms"), list):
                curr["认知盲区 (未涉及知识)"] = self._normalize_unknown_terms(
                    (curr.get("认知盲区 (未涉及知识)") or []) + (src.get("认知盲区 (未涉及知识)") or []) + (src.get("unknown_terms") or [])
                )

            kt = _clean_list(src.get("known_terms") or kb_u.get("提取术语") or [], max_len=120)
            if kt:
                kb = curr.get("个人领域知识库") if isinstance(curr.get("个人领域知识库"), dict) else {}
                kb["提取术语"] = _clean_list((kb.get("提取术语") or []) + kt, max_len=120)
                curr["个人领域知识库"] = kb

            if "last_confirmed_ts" in src:
                try:
                    curr["last_confirmed_ts"] = float(src.get("last_confirmed_ts") or 0.0)
                except Exception:
                    pass

            curr["updated_at"] = _now()
            self._imm_by_uid[uid] = self._normalize_imm(user_id=uid, imm=curr, user_name=user_name)
            self._imm_file_map[uid] = self._imm_file_for_user(uid)
            self._flush_imm(uid)
            return self._imm_with_legacy_projection(self._imm_by_uid[uid])

    def update_known_terms(self, user_id: str, terms: list[str]) -> None:
        uid = _clean_text(user_id)
        clean_terms = _clean_list([_clean_text(t).lower() for t in (terms or []) if _clean_text(t)], max_len=120)
        if not clean_terms:
            return
        with self._lock:
            imm = self._imm_by_uid.get(uid) or self._default_imm(uid)
            kb = imm.get("个人领域知识库") if isinstance(imm.get("个人领域知识库"), dict) else {}
            kb["提取术语"] = _clean_list((kb.get("提取术语") or []) + clean_terms, max_len=120)
            imm["个人领域知识库"] = kb
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
        chunks = re.split(r"[、,，;；/|\n。:：]", _normalize_ocr_spacing(text))
        out: list[str] = []
        for chunk in chunks:
            clean = _clean_term_candidate(chunk)
            if not clean:
                continue
            if clean not in out:
                out.append(clean)
            if len(out) >= max_len:
                break
        return out

    def _infer_imm_from_resume_text(self, user_id: str, user_name: str, text: str) -> dict:
        compact = _normalize_ocr_spacing(text)
        major = ""
        major_pattern_used = ""

        print(
            f"[DEBUG][mental_model] 进入简历字段提取 user_id={user_id} chars={len(compact)} "
            f"preview={_preview_text(compact)!r}"
        )

        major_patterns = [
            r"(?:专业|主修|研究方向|研究领域|Major)\s*[:：]?\s*([^，。；;\n]{2,40})",
            r"(?:本科|硕士|博士)(?:阶段)?(?:专业)?\s*[:：]?\s*([^，。；;\n]{2,40})",
            r"([\u4e00-\u9fff]{2,18}(?:专业|法学|科学|工程|管理学|经济学))",
            r"(人工智能|计算机科学|地理信息科学|法学)\s*[（(]?(?:本科|硕士|博士)?[)）]?",
        ]
        for pattern in major_patterns:
            m = re.search(pattern, compact, flags=re.IGNORECASE)
            if m:
                major = _clean_major_candidate(m.group(1))
                major_pattern_used = pattern
                break

        if major:
            print(
                f"[DEBUG][mental_model] 专业匹配成功 user_id={user_id} major={major!r} "
                f"pattern={major_pattern_used!r}"
            )
        else:
            print(f"[DEBUG][mental_model] 专业匹配失败 user_id={user_id}（未命中现有正则）")

        terms: list[str] = []
        term_hits = re.findall(
            r"(?:课程|研究兴趣|项目经历|擅长领域|研究方向|技能|熟悉|掌握)\s*[:：]?\s*([^\n]{2,200})",
            compact,
            flags=re.IGNORECASE,
        )
        for hit in term_hits:
            terms.extend(self._split_terms(hit, max_len=20))

        # OCR 文本常缺冒号，补一轮行级关键词抽取。
        if not term_hits:
            for line in re.split(r"[\n\r]", str(text or "")):
                line_norm = _normalize_ocr_spacing(line)
                if not line_norm:
                    continue
                if any(k in line_norm for k in ("课程", "研究兴趣", "研究方向", "擅长", "技能", "熟悉", "掌握")):
                    terms.extend(self._split_terms(line_norm, max_len=20))

        if term_hits:
            hit_preview = [_preview_text(h, max_len=80) for h in term_hits[:3]]
            print(
                f"[DEBUG][mental_model] 术语块匹配命中 user_id={user_id} "
                f"hit_count={len(term_hits)} hit_preview={hit_preview}"
            )
        else:
            print(f"[DEBUG][mental_model] 术语块匹配失败 user_id={user_id}（未命中现有正则）")

        clean_terms = _sanitize_terms(terms, max_len=40)
        expertise = [t for t in clean_terms if _is_domain_like(t) and _is_academic_term(t)][:8]
        familiar_candidates = [
            t for t in clean_terms if (_is_academic_term(t) or _is_skill_term(t))
        ]
        familiar = familiar_candidates[:12]
        if not expertise:
            expertise = _sanitize_terms(familiar, max_len=12, academic_only=True)
        if not major:
            major = _guess_major_from_terms(familiar)

        project_items = _extract_project_items_from_resume(text=text, familiar_terms=familiar, max_len=8)

        imm = self._default_imm(user_id=user_id, user_name=user_name)
        imm["个人画像"]["专业领域"] = major
        imm["个人画像"]["核心专长"] = expertise
        imm["个人画像"]["历史项目经验"] = project_items
        imm["个人领域知识库"]["提取术语"] = familiar
        imm["个人领域知识库"]["术语解释"] = []
        imm["updated_at"] = _now()

        print(
            f"[DEBUG][mental_model] 简历字段提取结果 user_id={user_id} "
            f"major={imm['个人画像']['专业领域']!r} "
            f"expertise={len(imm['个人画像']['核心专长'])} "
            f"familiar={len(imm['个人领域知识库']['提取术语'])}"
        )
        return imm

    def _is_imm_effectively_empty(self, imm: dict | None) -> bool:
        """判断IMM是否为空模板数据（仅在此场景下允许用本地PDF初始化）。"""
        if not isinstance(imm, dict):
            return True

        profile = imm.get("个人画像") if isinstance(imm.get("个人画像"), dict) else {}
        kb = imm.get("个人领域知识库") if isinstance(imm.get("个人领域知识库"), dict) else {}
        stance = imm.get("个人任务认知 (Task Stance)") if isinstance(imm.get("个人任务认知 (Task Stance)"), dict) else {}

        has_major = bool(_clean_text(profile.get("专业领域") or imm.get("professional_background")))
        has_expertise = bool(_clean_list(profile.get("核心专长") or imm.get("expertise_domains") or []))
        has_familiar = bool(_clean_list(kb.get("提取术语") or imm.get("familiar_terms") or []))
        has_project = bool(_clean_text(stance.get("期望研究方向") or imm.get("project_understanding")))
        has_unknown = bool(self._normalize_unknown_terms(imm.get("认知盲区 (未涉及知识)") or imm.get("unknown_terms") or []))
        has_known = bool(_clean_list(kb.get("提取术语") or imm.get("known_terms") or []))

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
                    old_stance = curr.get("个人任务认知 (Task Stance)") if isinstance(curr.get("个人任务认知 (Task Stance)"), dict) else {}
                    inferred["个人任务认知 (Task Stance)"]["期望研究方向"] = _clean_text(old_stance.get("期望研究方向") or curr.get("project_understanding"))
                    inferred["认知盲区 (未涉及知识)"] = self._normalize_unknown_terms(curr.get("认知盲区 (未涉及知识)") or curr.get("unknown_terms") or [])
                    inferred["last_confirmed_ts"] = float(curr.get("last_confirmed_ts") or 0.0)

                self._imm_by_uid[user_id] = self._normalize_imm(user_id=user_id, imm=inferred, user_name=user_name)
                self._imm_file_map[user_id] = self._imm_file_for_user(user_id)
                self._flush_imm(user_id)
                major = _clean_text((self._imm_by_uid[user_id].get("个人画像") or {}).get("专业领域"))
                keywords = len((self._imm_by_uid[user_id].get("个人领域知识库") or {}).get("提取术语") or [])
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
        phase = self._legacy_phase_to_cn(value)
        return phase if phase in PHASES else ""

    def _normalize_phase_status(self, value: str) -> str:
        status = self._legacy_status_to_cn(value, fallback="")
        return status if status in PHASE_STATUSES else ""

    def _conversation_round(self, convs: str) -> int:
        lines = [line for line in str(convs or "").splitlines() if _clean_text(line)]
        return max(1, len(lines))

    def _first_unsolved_term(self, imm: dict) -> str:
        for item in imm.get("认知盲区 (未涉及知识)") or []:
            if not isinstance(item, dict):
                continue
            status = _clean_text(item.get("当前状态"))
            term = _clean_text(item.get("未知术语"))
            if term and status in {"未解决", "解决中"}:
                return term
        return ""

    def _find_stale_conflict(self, smm: dict, current_round: int, stale_rounds: int = 3) -> dict | None:
        for conflict in smm.get("团队冲突区 (Conflict Zone)") or []:
            if not isinstance(conflict, dict):
                continue
            status = _clean_text(conflict.get("当前状态"))
            if status != "未解决":
                continue
            try:
                born_round = int(conflict.get("round") or 0)
            except Exception:
                born_round = 0
            if born_round > 0 and (current_round - born_round) >= stale_rounds:
                return conflict
        return None

    def _extract_professional_terms(self, text: str, max_len: int = 8) -> list[str]:
        content = _normalize_ocr_spacing(text)
        if not content:
            return []
        candidates: list[str] = []
        patterns = [
            r"(?:什么是|不懂|看不懂|解释(?:一下)?|怎么理解)\s*([A-Za-z][A-Za-z0-9+\-]{1,30}|[\u4e00-\u9fffA-Za-z0-9+\-]{2,20})",
            r"[《\"“](.{2,24})[》\"”]",
            r"\b([A-Z]{2,}[A-Za-z0-9+\-]{0,20})\b",
        ]
        for pat in patterns:
            for m in re.findall(pat, content):
                t = _clean_term_candidate(m)
                if not t:
                    continue
                if t not in candidates:
                    candidates.append(t)
                if len(candidates) >= max_len:
                    return candidates
        return candidates

    def _refresh_imm_timers(self, imm: dict) -> dict:
        unknowns = list(imm.get("认知盲区 (未涉及知识)") or [])
        refreshed: list[dict] = []
        for item in unknowns:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            status = _clean_text(row.get("当前状态")) or "未解决"
            if status in {"未解决", "解决中"}:
                if not _clean_text(row.get("触发时间戳")):
                    row["触发时间戳"] = _now_iso()
                row["持续时长_秒"] = _seconds_since_iso(row.get("触发时间戳"))
            else:
                try:
                    row["持续时长_秒"] = int(float(row.get("持续时长_秒") or 0))
                except Exception:
                    row["持续时长_秒"] = 0
            refreshed.append(row)
        imm["认知盲区 (未涉及知识)"] = refreshed
        return imm

    def _refresh_smm_timers(self, smm: dict) -> dict:
        life = smm.get("任务生命周期") if isinstance(smm.get("任务生命周期"), dict) else {}
        if not _clean_text(life.get("阶段进入时间")):
            life["阶段进入时间"] = _now_iso()
            life["阶段停留时长_分钟"] = 0
        else:
            life["阶段停留时长_分钟"] = max(0, _seconds_since_iso(life.get("阶段进入时间")) // 60)
        smm["任务生命周期"] = life

        conflicts = list(smm.get("团队冲突区 (Conflict Zone)") or [])
        refreshed_conflicts: list[dict] = []
        for item in conflicts:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            status = _clean_text(row.get("当前状态")) or "未解决"
            if status in {"未解决", "解决中"}:
                if not _clean_text(row.get("触发时间戳")):
                    row["触发时间戳"] = _now_iso()
                row["持续时长_秒"] = _seconds_since_iso(row.get("触发时间戳"))
            else:
                try:
                    row["持续时长_秒"] = int(float(row.get("持续时长_秒") or 0))
                except Exception:
                    row["持续时长_秒"] = 0
            refreshed_conflicts.append(row)
        smm["团队冲突区 (Conflict Zone)"] = refreshed_conflicts
        return smm

    def _parse_unknown_terms_from_message(self, message_text: str) -> list[dict]:
        text = _normalize_ocr_spacing(message_text)
        if not text:
            return []
        if not any(c in text.lower() for c in ["不懂", "看不懂", "什么是", "解释", "不太明白", "啥", "什么意思"]):
            return []
        terms = self._extract_professional_terms(text, max_len=5)
        return [
            {
                "未知术语": t,
                "当前状态": "未解决",
                "触发时间戳": _now_iso(),
                "持续时长_秒": 0,
            }
            for t in terms
        ]

    def _parse_task_stance_from_message(self, message_text: str) -> dict:
        text = _normalize_ocr_spacing(message_text)
        if not text:
            return {}
        stance: dict[str, str] = {}
        direction_patterns = [
            r"(?:研究方向|选题方向|目标方向|想做)\s*(?:是|为|:|：)?\s*(.{2,60})",
            r"(?:我们(?:就)?做|我们研究)\s*(.{2,60})",
        ]
        method_patterns = [
            r"(?:研究方法|方法)\s*(?:是|为|:|：)?\s*(.{2,60})",
            r"(?:打算|计划|准备用)\s*(.{2,60})",
        ]
        flow_patterns = [
            r"(?:实验流程|流程)\s*(?:是|为|:|：)?\s*(.{2,80})",
            r"(.{4,80}(?:->|→|先.+再.+))",
        ]

        for pat in direction_patterns:
            m = re.search(pat, text)
            if m:
                stance["期望研究方向"] = _clean_text(m.group(1)).strip("。；;，,")
                break
        for pat in method_patterns:
            m = re.search(pat, text)
            if m:
                stance["提议研究方法"] = _clean_text(m.group(1)).strip("。；;，,")
                break
        for pat in flow_patterns:
            m = re.search(pat, text)
            if m:
                stance["预期实验流程"] = _clean_text(m.group(1)).strip("。；;，,")
                break
        return stance

    def _parse_phase_from_context(self, message_text: str, convs: str, current_phase: str) -> str:
        content = _normalize_ocr_spacing(f"{convs}\n{message_text}")
        rules = [
            ("写作", ("写作", "论文", "汇报", "总结稿", "摘要")),
            ("实践", ("实验", "实现", "编码", "跑模型", "评测", "部署", "数据集")),
            ("分工", ("分工", "任务分配", "谁负责", "我来做", "你负责")),
            ("选题", ("选题", "题目", "研究方向", "做什么", "方向")),
        ]
        for phase, cues in rules:
            if any(c in content for c in cues):
                return phase
        return current_phase or "破冰"

    def _parse_conflict_from_message(self, message_text: str) -> dict:
        text = _normalize_ocr_spacing(message_text)
        if not text:
            return {}
        cues = ("不同意", "冲突", "矛盾", "争论", "但我觉得", "但是", "vs", "还是")
        if not any(c in text.lower() for c in [x.lower() for x in cues]):
            return {}
        return {
            "冲突描述": _preview_text(text, max_len=120),
            "当前状态": "未解决",
            "触发时间戳": _now_iso(),
            "持续时长_秒": 0,
        }

    def _parse_shared_consensus_from_message(self, message_text: str) -> dict:
        text = _normalize_ocr_spacing(message_text)
        if not text:
            return {}
        out: dict[str, Any] = {"已确认方向": "", "已确认方法": "", "已确认分工_delta": []}

        direction_patterns = [
            r"(?:我们(?:就)?定为|选题定为|方向定为|确定方向为)\s*(.{2,60})",
            r"(?:最终选题|确认选题)\s*(?:是|为|:|：)?\s*(.{2,60})",
        ]
        method_patterns = [
            r"(?:方法定为|研究方法定为|确认方法为)\s*(.{2,60})",
            r"(?:我们采用|我们使用)\s*(.{2,60})",
        ]

        for pat in direction_patterns:
            m = re.search(pat, text)
            if m:
                out["已确认方向"] = _clean_text(m.group(1)).strip("。；;，,")
                break
        for pat in method_patterns:
            m = re.search(pat, text)
            if m:
                out["已确认方法"] = _clean_text(m.group(1)).strip("。；;，,")
                break

        division_cues = ["我负责", "你负责", "由我", "由你", "分工", "任务分配"]
        if any(c in text for c in division_cues):
            out["已确认分工_delta"] = [_preview_text(text, max_len=80)]

        if not out["已确认方向"] and not out["已确认方法"] and not out["已确认分工_delta"]:
            return {}
        return out

    def _run_parallel_parsers(self, message_text: str, convs: str, curr_imm: dict, curr_smm: dict) -> dict:
        current_phase = _clean_text(((curr_smm or {}).get("任务生命周期") or {}).get("当前所处阶段") or "破冰")
        with ThreadPoolExecutor(max_workers=4) as ex:
            future_terms = ex.submit(self._parse_unknown_terms_from_message, message_text)
            future_stance = ex.submit(self._parse_task_stance_from_message, message_text)
            future_phase = ex.submit(self._parse_phase_from_context, message_text, convs, current_phase)
            future_conflict = ex.submit(self._parse_conflict_from_message, message_text)
            future_consensus = ex.submit(self._parse_shared_consensus_from_message, message_text)

            terms = future_terms.result() or []
            stance = future_stance.result() or {}
            phase = _clean_text(future_phase.result() or current_phase)
            conflict = future_conflict.result() or {}
            consensus = future_consensus.result() or {}

        out = {
            "imm_update": {
                "认知盲区_delta": terms,
                "个人任务认知 (Task Stance)": stance,
            },
            "smm_update": {
                "任务生命周期": {
                    "当前所处阶段": phase,
                    "阶段进入时间": "",
                    "阶段停留时长_分钟": 0,
                },
                "团队共识区 (Shared Consensus)": consensus,
                "冲突": conflict,
                "phase_status": "未解决" if _looks_like_progress_stall(message_text) else "解决中",
            },
        }
        return out

    def _first_overtime_unknown_term(self, imm: dict, threshold_seconds: int = UNKNOWN_TERM_TIMEOUT_SECONDS) -> str:
        for item in imm.get("认知盲区 (未涉及知识)") or []:
            if not isinstance(item, dict):
                continue
            if _clean_text(item.get("当前状态")) != "未解决":
                continue
            if int(float(item.get("持续时长_秒") or 0)) > threshold_seconds:
                term = _clean_text(item.get("未知术语"))
                if term:
                    return term
        return ""

    def _first_overtime_conflict(self, smm: dict, threshold_seconds: int = CONFLICT_TIMEOUT_SECONDS) -> dict | None:
        for item in smm.get("团队冲突区 (Conflict Zone)") or []:
            if not isinstance(item, dict):
                continue
            if _clean_text(item.get("当前状态")) != "未解决":
                continue
            if int(float(item.get("持续时长_秒") or 0)) > threshold_seconds:
                return item
        return None

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
            "1) 当前所处阶段 只能是 破冰|选题|分工|实践|写作\n"
            "2) 状态字段 只能是 未解决|解决中|已解决\n"
            "5) response_type 只能是 professional_explain|judgment|knowledge|topic|division|summary|none\n\n"
            "输出JSON结构：\n"
            "{"
            "\"imm_update\":{"
            "\"个人画像\":{\"姓名\":\"\",\"专业领域\":\"\",\"核心专长_delta\":[],\"历史项目经验_delta\":[{\"项目名称\":\"\",\"采用技术\":[],\"实现效果\":\"\"}]},"
            "\"个人领域知识库\":{\"提取术语_delta\":[],\"术语解释_delta\":[]},"
            "\"认知盲区_delta\":[{\"未知术语\":\"\",\"当前状态\":\"未解决|解决中|已解决\",\"note\":\"\"}],"
            "\"个人任务认知 (Task Stance)\":{\"期望研究方向\":\"\",\"提议研究方法\":\"\",\"预期实验流程\":\"\"},"
            "\"user_name\":\"\""
            "},"
            "\"smm_update\":{"
            "\"任务生命周期\":{\"当前所处阶段\":\"破冰|选题|分工|实践|写作\",\"阶段进入时间\":\"\",\"阶段停留时长_分钟\":0},"
            "\"团队共识区 (Shared Consensus)\":{\"已确认方向\":\"\",\"已确认方法\":\"\",\"已确认分工_delta\":[]},"
            "\"冲突\":{\"冲突描述\":\"\",\"当前状态\":\"未解决|解决中|已解决\",\"触发时间戳\":\"\",\"持续时长_秒\":0,\"note\":\"\"},"
            "\"phase_status\":\"未解决|解决中|已解决\""
            "},"
            "\"response_decision\":{"
            "\"should_respond\":true,"
            "\"response_type\":\"professional_explain|judgment|knowledge|topic|division|summary|none\","
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
        parser_updates = self._run_parallel_parsers(message_text=message_text, convs=convs, curr_imm=imm, curr_smm=smm)
        parser_imm_u = parser_updates.get("imm_update") if isinstance(parser_updates.get("imm_update"), dict) else {}
        parser_smm_u = parser_updates.get("smm_update") if isinstance(parser_updates.get("smm_update"), dict) else {}

        with self._lock:
            curr_imm = self._imm_by_uid.get(user_id) or self._default_imm(user_id=user_id, user_name=user_name)

            if _clean_text(imm_u.get("user_name")):
                curr_imm["user_name"] = _clean_text(imm_u.get("user_name"))
            elif user_name and not _clean_text(curr_imm.get("user_name")):
                curr_imm["user_name"] = user_name

            profile_u = imm_u.get("个人画像") if isinstance(imm_u.get("个人画像"), dict) else {}
            kb_u = imm_u.get("个人领域知识库") if isinstance(imm_u.get("个人领域知识库"), dict) else {}
            stance_u = imm_u.get("个人任务认知 (Task Stance)") if isinstance(imm_u.get("个人任务认知 (Task Stance)"), dict) else {}
            parser_stance_u = parser_imm_u.get("个人任务认知 (Task Stance)") if isinstance(parser_imm_u.get("个人任务认知 (Task Stance)"), dict) else {}

            pb = _clean_text(profile_u.get("专业领域") or imm_u.get("professional_background"))
            if pb:
                profile = curr_imm.get("个人画像") if isinstance(curr_imm.get("个人画像"), dict) else {}
                profile["专业领域"] = pb
                curr_imm["个人画像"] = profile

            ex_delta = _clean_list(profile_u.get("核心专长_delta") or imm_u.get("expertise_domains_delta") or [], max_len=80)
            if ex_delta:
                profile = curr_imm.get("个人画像") if isinstance(curr_imm.get("个人画像"), dict) else {}
                profile["核心专长"] = _clean_list((profile.get("核心专长") or []) + ex_delta, max_len=80)
                curr_imm["个人画像"] = profile

            ft_delta = _clean_list(kb_u.get("提取术语_delta") or imm_u.get("familiar_terms_delta") or [], max_len=120)
            if ft_delta:
                kb = curr_imm.get("个人领域知识库") if isinstance(curr_imm.get("个人领域知识库"), dict) else {}
                kb["提取术语"] = _clean_list((kb.get("提取术语") or []) + ft_delta, max_len=120)
                curr_imm["个人领域知识库"] = kb

            explain_delta = _clean_list(kb_u.get("术语解释_delta") or [], max_len=120)
            if explain_delta:
                kb = curr_imm.get("个人领域知识库") if isinstance(curr_imm.get("个人领域知识库"), dict) else {}
                kb["术语解释"] = _clean_list((kb.get("术语解释") or []) + explain_delta, max_len=120)
                curr_imm["个人领域知识库"] = kb

            pu = _clean_text(stance_u.get("期望研究方向") or imm_u.get("project_understanding"))
            if pu:
                stance = curr_imm.get("个人任务认知 (Task Stance)") if isinstance(curr_imm.get("个人任务认知 (Task Stance)"), dict) else {}
                stance["期望研究方向"] = pu
                curr_imm["个人任务认知 (Task Stance)"] = stance

            method = _clean_text(stance_u.get("提议研究方法"))
            if method:
                stance = curr_imm.get("个人任务认知 (Task Stance)") if isinstance(curr_imm.get("个人任务认知 (Task Stance)"), dict) else {}
                stance["提议研究方法"] = method
                curr_imm["个人任务认知 (Task Stance)"] = stance

            pipeline = _clean_text(stance_u.get("预期实验流程"))
            if pipeline:
                stance = curr_imm.get("个人任务认知 (Task Stance)") if isinstance(curr_imm.get("个人任务认知 (Task Stance)"), dict) else {}
                stance["预期实验流程"] = pipeline
                curr_imm["个人任务认知 (Task Stance)"] = stance

            parser_direction = _clean_text(parser_stance_u.get("期望研究方向"))
            parser_method = _clean_text(parser_stance_u.get("提议研究方法"))
            parser_flow = _clean_text(parser_stance_u.get("预期实验流程"))
            if parser_direction or parser_method or parser_flow:
                stance = curr_imm.get("个人任务认知 (Task Stance)") if isinstance(curr_imm.get("个人任务认知 (Task Stance)"), dict) else {}
                if parser_direction:
                    stance["期望研究方向"] = parser_direction
                if parser_method:
                    stance["提议研究方法"] = parser_method
                if parser_flow:
                    stance["预期实验流程"] = parser_flow
                curr_imm["个人任务认知 (Task Stance)"] = stance

            if isinstance(imm_u.get("认知盲区_delta"), list) or isinstance(imm_u.get("unknown_terms_delta"), list):
                curr_imm["认知盲区 (未涉及知识)"] = self._normalize_unknown_terms(
                    (curr_imm.get("认知盲区 (未涉及知识)") or []) + list(imm_u.get("认知盲区_delta") or []) + list(imm_u.get("unknown_terms_delta") or [])
                )
            if isinstance(parser_imm_u.get("认知盲区_delta"), list):
                curr_imm["认知盲区 (未涉及知识)"] = self._normalize_unknown_terms(
                    (curr_imm.get("认知盲区 (未涉及知识)") or []) + list(parser_imm_u.get("认知盲区_delta") or [])
                )

            curr_imm["updated_at"] = _now()
            curr_imm = self._refresh_imm_timers(curr_imm)
            self._imm_by_uid[user_id] = self._normalize_imm(user_id=user_id, imm=curr_imm, user_name=user_name)
            self._imm_file_map[user_id] = self._imm_file_for_user(user_id)
            self._flush_imm(user_id)

            curr_smm = self._smm_by_channel.get(channel_id) or self._default_smm(channel_id=channel_id)
            prev_phase = _clean_text((curr_smm.get("任务生命周期") or {}).get("当前所处阶段") or "破冰")

            life_u = smm_u.get("任务生命周期") if isinstance(smm_u.get("任务生命周期"), dict) else {}
            cons_u = smm_u.get("团队共识区 (Shared Consensus)") if isinstance(smm_u.get("团队共识区 (Shared Consensus)"), dict) else {}
            parser_life_u = parser_smm_u.get("任务生命周期") if isinstance(parser_smm_u.get("任务生命周期"), dict) else {}
            parser_cons_u = parser_smm_u.get("团队共识区 (Shared Consensus)") if isinstance(parser_smm_u.get("团队共识区 (Shared Consensus)"), dict) else {}
            phase = self._normalize_phase(parser_life_u.get("当前所处阶段") or life_u.get("当前所处阶段") or smm_u.get("current_phase")) or prev_phase
            phase_status = self._normalize_phase_status(parser_smm_u.get("phase_status") or smm_u.get("phase_status")) or _clean_text(curr_smm.get("phase_status") or "解决中")
            if phase_status not in PHASE_STATUSES:
                phase_status = "解决中"

            goal = _clean_text(parser_cons_u.get("已确认方向") or cons_u.get("已确认方向") or smm_u.get("common_goal"))
            if goal:
                consensus = curr_smm.get("团队共识区 (Shared Consensus)") if isinstance(curr_smm.get("团队共识区 (Shared Consensus)"), dict) else {}
                consensus["已确认方向"] = goal
                curr_smm["团队共识区 (Shared Consensus)"] = consensus

            method = _clean_text(parser_cons_u.get("已确认方法") or cons_u.get("已确认方法") or smm_u.get("team_cognition"))
            if method:
                consensus = curr_smm.get("团队共识区 (Shared Consensus)") if isinstance(curr_smm.get("团队共识区 (Shared Consensus)"), dict) else {}
                consensus["已确认方法"] = method
                curr_smm["团队共识区 (Shared Consensus)"] = consensus

            division_delta = _clean_list((cons_u.get("已确认分工_delta") or []) + (parser_cons_u.get("已确认分工_delta") or []), max_len=120)
            if division_delta:
                consensus = curr_smm.get("团队共识区 (Shared Consensus)") if isinstance(curr_smm.get("团队共识区 (Shared Consensus)"), dict) else {}
                consensus["已确认分工"] = _clean_list((consensus.get("已确认分工") or []) + division_delta, max_len=120)
                curr_smm["团队共识区 (Shared Consensus)"] = consensus

            conflict = smm_u.get("冲突") if isinstance(smm_u.get("冲突"), dict) else (
                smm_u.get("conflict") if isinstance(smm_u.get("conflict"), dict) else {}
            )
            parser_conflict = parser_smm_u.get("冲突") if isinstance(parser_smm_u.get("冲突"), dict) else {}
            if parser_conflict and not conflict:
                conflict = parser_conflict
            topic = _clean_text(conflict.get("冲突描述") or conflict.get("topic"))
            if topic:
                c_status = self._legacy_status_to_cn(conflict.get("当前状态") or conflict.get("status"), fallback="未解决")
                if c_status not in CONFLICT_STATUSES:
                    c_status = "未解决"
                entry = {
                    "冲突描述": topic,
                    "当前状态": c_status,
                    "触发时间戳": _clean_text(conflict.get("触发时间戳")) or _now_iso(),
                    "持续时长_秒": int(float(conflict.get("持续时长_秒") or 0)),
                    "note": _clean_text(conflict.get("note")),
                }
                conflicts = list(curr_smm.get("团队冲突区 (Conflict Zone)") or [])
                replaced = False
                for idx, old in enumerate(conflicts):
                    if _clean_text(old.get("冲突描述")).lower() == topic.lower():
                        conflicts[idx] = entry
                        replaced = True
                        break
                if not replaced:
                    conflicts.append(entry)
                curr_smm["团队冲突区 (Conflict Zone)"] = conflicts[:120]

            life = curr_smm.get("任务生命周期") if isinstance(curr_smm.get("任务生命周期"), dict) else {}
            phase_changed = bool(phase and phase != prev_phase)
            life["当前所处阶段"] = phase
            if phase_changed:
                life["阶段进入时间"] = _now_iso()
                life["阶段停留时长_分钟"] = 0
            elif _clean_text(life_u.get("阶段进入时间")):
                life["阶段进入时间"] = _clean_text(life_u.get("阶段进入时间"))
            try:
                stay = int(float(life_u.get("阶段停留时长_分钟") or life.get("阶段停留时长_分钟") or 0))
            except Exception:
                stay = int(float(life.get("阶段停留时长_分钟") or 0))
            if not phase_changed:
                life["阶段停留时长_分钟"] = max(0, stay)
            curr_smm["任务生命周期"] = life
            curr_smm["phase_status"] = phase_status

            # SMM优先：在选题/分工阶段遇到明显停滞语句时，强制标记为未解决。
            if phase in {"选题", "分工"} and _looks_like_progress_stall(message_text):
                curr_smm["phase_status"] = "未解决"

            curr_smm["updated_at"] = _now()
            curr_smm = self._refresh_smm_timers(curr_smm)
            self._smm_by_channel[channel_id] = self._normalize_smm(channel_id=channel_id, smm=curr_smm)
            self._flush_smm()

            new_phase = _clean_text((self._smm_by_channel[channel_id].get("任务生命周期") or {}).get("当前所处阶段") or phase)
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

        # 规则型兜底：当模型不触发时，按 IMM/SMM 关键状态主动触发。
        if not decision["should_respond"]:
            current_round = self._conversation_round(convs)
            latest_imm = self.get_imm(user_id=user_id, user_name=user_name)
            latest_smm = self.get_smm(channel_id=channel_id)

            # 主动响应器1：IMM 未解决术语超过阈值，主动解释。
            unsolved_term_timeout = self._first_overtime_unknown_term(latest_imm, threshold_seconds=UNKNOWN_TERM_TIMEOUT_SECONDS)
            if unsolved_term_timeout:
                decision = {
                    "should_respond": True,
                    "response_type": "professional_explain",
                    "query": unsolved_term_timeout,
                    "reason": "active_unknown_term_timeout",
                }

            # 主动响应器2：SMM 未解决冲突超过阈值，主动介入判断。
            if not decision["should_respond"]:
                overtime_conflict = self._first_overtime_conflict(latest_smm, threshold_seconds=CONFLICT_TIMEOUT_SECONDS)
                if overtime_conflict:
                    decision = {
                        "should_respond": True,
                        "response_type": "judgment",
                        "query": _clean_text(overtime_conflict.get("冲突描述") or overtime_conflict.get("topic")) or _clean_text(message_text),
                        "reason": "active_conflict_timeout",
                    }

            # 主动响应器3：选题阶段超时，主动询问是否需要选题帮助。
            life = latest_smm.get("任务生命周期") if isinstance(latest_smm.get("任务生命周期"), dict) else {}
            phase_now = _clean_text(life.get("当前所处阶段") or latest_smm.get("current_phase"))
            phase_stay = int(float(life.get("阶段停留时长_分钟") or 0))
            phase_status_cn = self._legacy_status_to_cn(latest_smm.get("phase_status"), fallback="未解决")

            if (not decision["should_respond"]
                and phase_now == "选题"
                and phase_status_cn != "已解决"
                and phase_stay >= TOPIC_STAGE_TIMEOUT_MINUTES):
                decision = {
                    "should_respond": True,
                    "response_type": "topic",
                    "query": "选题阶段已停留较久，我可以帮你们快速收敛选题。是否需要我基于双方背景给出3个方向？",
                    "reason": "active_topic_stage_timeout",
                }

            # 主动响应器3：分工阶段超时，主动询问是否需要分工帮助。
            if (not decision["should_respond"]
                and phase_now == "分工"
                and phase_status_cn != "已解决"
                and phase_stay >= DIVISION_STAGE_TIMEOUT_MINUTES):
                decision = {
                    "should_respond": True,
                    "response_type": "division",
                    "query": "分工阶段已停留较久，我可以按成员优势生成可执行分工方案。是否需要我现在给出？",
                    "reason": "active_division_stage_timeout",
                }

            # 规则4：用户明确要求总结，或阶段进入总结时，主动生成总结。
            if not decision["should_respond"]:
                if _looks_like_summary_request(message_text) or _clean_text(((latest_smm.get("任务生命周期") or {}).get("当前所处阶段") or latest_smm.get("current_phase"))) in {"写作", "总结"}:
                    decision = {
                        "should_respond": True,
                        "response_type": "summary",
                        "query": _clean_text(message_text) or "请总结当前讨论进展",
                        "reason": "rule_summary",
                    }

            # 规则5：冲突长期未解决（轮次视角）兜底。
            if not decision["should_respond"]:
                stale_conflict = self._find_stale_conflict(latest_smm, current_round=current_round, stale_rounds=3)
                if stale_conflict:
                    decision = {
                        "should_respond": True,
                        "response_type": "judgment",
                        "query": _clean_text(stale_conflict.get("冲突描述") or stale_conflict.get("topic")) or _clean_text(message_text),
                        "reason": "rule_stale_conflict",
                    }

        if decision["response_type"] not in {"professional_explain", "judgment", "knowledge", "topic", "division", "summary", "none"}:
            decision["response_type"] = "none"
        if decision["response_type"] == "none":
            decision["should_respond"] = False

        return {
            "imm": self.get_imm(user_id=user_id, user_name=user_name),
            "smm": self.get_smm(channel_id=channel_id),
            "decision": decision,
            "smm_transition": transition,
        }
