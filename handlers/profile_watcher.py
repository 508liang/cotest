"""
handlers/profile_watcher.py

修改记录：
  1. [写库时机] 提炼完画像后不再静默写库，改为发送确认卡片，等用户确认后才写库。
  2. [冷却机制] 已删除冷却机制，每次用户发言都触发画像监控。
  3. [新用户初始化] 首次检测到新用户时在 DB 中创建空白画像记录。
    4. [画像唯一性] 画像按 channel_id + user_id 唯一，同一用户在不同频道独立维护。
"""

from __future__ import annotations
import threading
import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.cosearch_agent import CoSearchAgent
    from memory.cosearch_agent_memory import Memory
    from memory.user_profile_memory import UserProfileMemory


_PROFILE_CUES = (
    "我是", "我学", "我的专业", "我专业", "研究方向", "研究兴趣", "擅长", "背景", "本科", "硕士", "博士",
    "方向是", "专业是", "从事", "主修", "主要做", "做的是",
    "感兴趣", "兴趣", "关注", "想研究", "想做", "计划研究",
)

_TASK_REQUEST_CUES = (
    "总结", "归纳", "复盘", "回顾", "选题", "分工", "解释", "判断",
    "帮我", "请帮", "能不能", "可不可以", "怎么", "如何", "请问",
)

_SELF_STRONG_PROFILE_PATTERNS = (
    r"我(是|来自|学|读)(.{0,16})(专业|方向|学院)",
    r"(我的|我)(研究方向|研究兴趣|专业)是",
    r"我(主修|从事|主要做)",
    r"我对.{1,30}(感兴趣|有兴趣|更关注|关注|想研究|想做|计划研究)",
)


def _normalize_phrase(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^[\s,，。.!！？:：;；、~\-]+", "", value)
    value = re.sub(r"[\s,，。.!！？:：;；、~\-]+$", "", value)
    value = re.sub(r"\s+", "", value)
    return value


def _extract_interest_phrases_from_utterance(text: str) -> list[str]:
    """规则保底：从原句抽取显式兴趣短语，避免LLM泛化丢失细粒度兴趣词。"""
    content = (text or "").strip()
    if not content:
        return []

    patterns = [
        r"我对(.{1,30}?)(感兴趣|有兴趣|更关注|关注|想研究|想做|计划研究)",
        r"我最近在关注(.{1,30})",
        r"我想研究(.{1,30})",
    ]

    extracted: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, content):
            raw = match[0] if isinstance(match, tuple) else match
            phrase = _normalize_phrase(raw)
            if not phrase:
                continue
            if len(phrase) < 2 or len(phrase) > 20:
                continue
            if phrase not in extracted:
                extracted.append(phrase)
    return extracted


def _is_profile_relevant_utterance(text: str, *, is_self: bool) -> bool:
    """判断一条发言是否可能包含用户画像信息，过滤命令类/闲聊类噪声。"""
    content = (text or "").strip()
    if not content:
        return False

    # 任务请求（总结/选题/分工/解释等）默认不作为画像证据，除非命中强画像模式。
    looks_like_task_request = any(k in content for k in _TASK_REQUEST_CUES)

    if is_self:
        if any(re.search(p, content) for p in _SELF_STRONG_PROFILE_PATTERNS):
            return True
        if looks_like_task_request:
            return False
        # 自述证据要求同时包含“我”和画像线索，避免把泛讨论当成画像更新。
        if "我" in content and any(c in content for c in _PROFILE_CUES):
            return True
        return False

    if looks_like_task_request:
        return False

    # 他人描述某用户背景的弱模式
    if re.search(r"(专业|方向|背景|擅长|主修)", content):
        return True

    # 他人提及其兴趣/研究方向（如：X对Y感兴趣）
    if re.search(r"对.{1,30}(感兴趣|有兴趣|关注|想研究)", content):
        return True

    return False


def _is_target_background_mention(text: str, mention_tokens: list[str]) -> bool:
    """判断他人发言是否在描述目标用户背景，避免把普通@提及当成画像证据。"""
    content = (text or "").strip()
    if not content:
        return False
    if any(k in content for k in _TASK_REQUEST_CUES):
        return False

    valid_tokens = [t for t in (mention_tokens or []) if t]
    if not valid_tokens:
        return False
    if not any(token in content for token in valid_tokens):
        return False

    token_pattern = "|".join(re.escape(t) for t in valid_tokens)
    if not token_pattern:
        return False

    role_cues = r"专业|方向|背景|主修|研究兴趣|研究方向|擅长|做的是|从事"
    interest_cues = r"感兴趣|有兴趣|更关注|关注|想研究|想做|计划研究"

    patterns = (
        rf"(?:{token_pattern}).{{0,12}}(?:{role_cues})",
        rf"(?:{role_cues}).{{0,12}}(?:{token_pattern})",
        rf"(?:{token_pattern}).{{0,8}}对.{{1,30}}(?:{interest_cues})",
    )
    return any(re.search(p, content) for p in patterns)


def watch_profile_in_background(
    *,
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    agent: "CoSearchAgent",
    memory: "Memory",
    profile_memory: "UserProfileMemory",
) -> None:
    """
    在后台线程中执行画像监控，调用方无需等待。
    每次用户发言都触发（无冷却机制）。
    """
    t = threading.Thread(
        target=_watch,
        kwargs=dict(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            agent=agent,
            memory=memory,
            profile_memory=profile_memory,
        ),
        daemon=True,
    )
    t.start()


def _watch(
    *,
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    agent,
    memory,
    profile_memory,
) -> None:
    """
    实际执行逻辑（在子线程中运行）：
      1. 新用户检测：若 DB 中无记录，初始化空白画像
      2. 从 DB 读取该用户在本频道的全部自有发言（排除Bot回复）
      3. 读取已有画像，组合成"已有画像 + 新发言"传给 LLM 做增量提炼
      4. 有增量变化时，发送确认卡片（不直接写库）
    """
    from handlers.profile_utils import notify_profile_update_if_changed

    print(f"[DEBUG][profile_watcher] 开始监控 user={user_name!r} channel={channel_name!r}")

    # ── Step 0: 新用户初始化（DB 中无记录则创建空白画像）─────────────────────
    existing = profile_memory.load(user_id, channel_id)
    if existing is None:
        blank = {
            "channel_id": channel_id,
            "user_id": user_id,
            "user_name": user_name,
            "major": "",
            "research_interests": [],
            "methodology": [],
            "keywords": [],
        }
        profile_memory.save(blank, channel_id)
        existing = blank
        print(f"[DEBUG][profile_watcher] 新用户，已初始化空白画像 user_id={user_id!r}")

    # ── Step 1: 读取上次确认时间，按增量窗口提取证据 ──────────────────────────
    last_confirmed_ts = float(existing.get("last_confirmed_ts") or 0.0)
    # 历史数据兼容：老记录没有 last_confirmed_ts 时，尝试用 updated_at 作为基线
    if last_confirmed_ts <= 0:
        updated_at = (existing.get("updated_at") or "").strip()
        if updated_at:
            try:
                baseline_dt = datetime.strptime(updated_at, "%Y-%m-%d")
                last_confirmed_ts = baseline_dt.timestamp()
            except Exception:
                pass
    print(f"[DEBUG][profile_watcher] last_confirmed_ts={last_confirmed_ts}")

    try:
        all_rows = memory.load_all_utterances(table_name=channel_name)
    except Exception as e:
        print(f"[DEBUG][profile_watcher] ⚠ 读取DB失败: {e}")
        return

    def _to_float_ts(raw) -> float:
        try:
            return float(raw)
        except Exception:
            return 0.0

    # A) 目标用户本人在确认后新增的发言
    own_lines = []
    own_interest_phrases: list[str] = []
    own_seen_utterances: set[str] = set()
    # B) 他人在确认后提到目标用户背景的发言
    mention_lines = []

    mention_tokens = [
        user_name,
        f"<@{user_id}>",
        (user_name.split(" ")[0] if user_name else ""),
    ]
    mention_tokens = [t for t in mention_tokens if t]

    for row in all_rows:
        speaker = row.get("speaker", "")
        utterance = (row.get("utterance", "") or "").strip()
        row_ts = _to_float_ts(row.get("timestamp"))

        if not utterance:
            continue
        if row_ts <= last_confirmed_ts:
            continue

        if speaker == user_name:
            if _is_profile_relevant_utterance(utterance, is_self=True):
                # 去重：同一句重复入库时只保留一次，避免重复证据干扰提炼
                if utterance not in own_seen_utterances:
                    own_seen_utterances.add(utterance)
                    own_lines.append(f"{speaker}: {utterance}")
                    for phrase in _extract_interest_phrases_from_utterance(utterance):
                        if phrase not in own_interest_phrases:
                            own_interest_phrases.append(phrase)
            continue

        if speaker == "CoSearchAgent":
            continue

        if _is_target_background_mention(utterance, mention_tokens):
            mention_lines.append(f"{speaker}: {utterance}")

    if not own_lines and not mention_lines:
        print(f"[DEBUG][profile_watcher] 确认后无新增证据，跳过")
        return

    own_convs = "\n".join(own_lines)
    mention_convs = "\n".join(mention_lines)
    print(f"[DEBUG][profile_watcher] 新增本人发言={len(own_lines)} 条, 他人提及={len(mention_lines)} 条")

    # ── Step 2: 组合输入传给 LLM ─────────────────────────────────────────────
    existing_has_content = (
        existing.get("major")
        or existing.get("research_interests")
        or existing.get("methodology")
        or existing.get("keywords")
    )

    if existing_has_content:
        existing_summary = _format_existing_for_prompt(existing)
        combined_input = (
            f"【已确认的用户画像】\n{existing_summary}\n\n"
            f"【新增对话记录（仅该用户自己的发言）】\n{own_convs or '（无）'}\n\n"
            f"【新增对话记录（其他用户提及该用户背景的信息）】\n{mention_convs or '（无）'}"
        )
    else:
        combined_input = (
            f"【用户对话记录（仅该用户自己的发言）】\n{own_convs or '（无）'}\n\n"
            f"【其他用户提及该用户背景的信息】\n{mention_convs or '（无）'}\n\n"
            f"请从以上新增证据中提取该用户的学术背景信息。"
        )

    try:
        extracted, elapsed = agent.extract_user_profile(user=user_name, convs=combined_input)
    except Exception as e:
        print(f"[DEBUG][profile_watcher] ⚠ 画像提炼失败: {e}")
        return

    has_content = (
        extracted.get("major")
        or extracted.get("research_interests")
        or extracted.get("methodology")
        or extracted.get("keywords")
    )
    if not has_content:
        print(f"[DEBUG][profile_watcher] ({elapsed:.2f}s) 发言无新学术信息，跳过")
        return

    extracted["user_id"]   = user_id
    extracted["user_name"] = user_name

    # 规则保底：把显式兴趣短语强制并入，避免被LLM泛化成上位词（如“欧洲法律”->“国际法”）
    draft_interests = [
        _normalize_phrase(x) for x in (extracted.get("research_interests") or []) if _normalize_phrase(x)
    ]
    for phrase in own_interest_phrases:
        if phrase not in draft_interests:
            draft_interests.append(phrase)
    extracted["research_interests"] = draft_interests

    if own_interest_phrases:
        print(f"[DEBUG][profile_watcher] 规则补充兴趣: {own_interest_phrases}")

    print(f"[DEBUG][profile_watcher] ({elapsed:.2f}s) 提炼结果: "
          f"major={extracted.get('major')!r} "
          f"interests={extracted.get('research_interests')} "
          f"methods={extracted.get('methodology')} "
          f"keywords={extracted.get('keywords')}")

    # ── Step 3: 有增量则发确认卡片（不直接写库）─────────────────────────────
    notified = notify_profile_update_if_changed(
        client=client,
        channel_id=channel_id,
        user_id=user_id,
        existing=existing,
        new_draft=extracted,
        profile_memory=profile_memory,
    )
    if notified:
        try:
            from utils import send_status_message

            send_status_message(
                client,
                channel_id,
                user_id,
                "检测到画像更新，已发送确认卡片。",
            )
        except Exception as e:
            print(f"[DEBUG][profile_watcher] ⚠ 发送画像更新提示失败: {e}")
    print(f"[DEBUG][profile_watcher] 完成 notified={notified}")


def _format_existing_for_prompt(profile: dict) -> str:
    """将已有画像格式化为自然语言，供 LLM 理解已有信息。"""
    if not profile:
        return "（暂无已确认画像）"
    major     = profile.get("major") or "未知"
    interests = "、".join(profile.get("research_interests") or []) or "暂无"
    methods   = "、".join(profile.get("methodology") or []) or "暂无"
    keywords  = "、".join(profile.get("keywords") or []) or "暂无"
    return (
        f"专业：{major}\n"
        f"研究兴趣：{interests}\n"
        f"擅长方法：{methods}\n"
        f"关键词：{keywords}"
    )
