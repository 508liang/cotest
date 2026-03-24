"""
handlers/profile_utils.py

修改记录：
  1. [画像污染] filter_bot_utterances 仅保留作为文本兜底的降级路径，正常路径
     已由 app 层的 get_user_only_conversation_history() 在 Slack API 层面
     彻底跳过 Bot 消息，不再依赖文本格式匹配。
  2. [Debug]   draft_profiles_from_convs / profiles_have_changed /
     merge_profile_with_existing 均添加详细日志。
  3. [逻辑修复] extract_latest_topic：优先匹配 Bot 输出的"📌 题目"行，
     其次匹配用户口述确认的选题；避免把Bot问候语误识别为选题。
"""

import re
import time
import threading
from typing import Any


PROFILE_NOTIFY_COOLDOWN_SECONDS = 20
_recent_profile_notify_at: dict[tuple[str, str], float] = {}
_profile_notify_lock = threading.Lock()


def _profile_notify_in_cooldown(channel_id: str, user_id: str) -> bool:
    now = time.time()
    key = (channel_id, user_id)
    with _profile_notify_lock:
        last_ts = _recent_profile_notify_at.get(key, 0.0)
        if now - last_ts <= PROFILE_NOTIFY_COOLDOWN_SECONDS:
            print(
                f"[DEBUG][profile_utils] notify_profile_update: 命中冷却，跳过重复通知 "
                f"channel={channel_id!r} user_id={user_id!r} delta={now - last_ts:.2f}s"
            )
            return True
        _recent_profile_notify_at[key] = now
        return False


# ── Bot消息文本过滤（仅作降级兜底，正常路径应使用 get_user_only_conversation_history）

def filter_bot_utterances(convs: str, bot_name: str = "CoSearchAgent") -> str:
    """
    从文本格式的对话记录中过滤 Bot 多行发言块。
    ⚠ 此函数为降级兜底：若 Slack API 已在拉取阶段跳过 Bot 消息，
       则无需再调此函数。仅在无法直接调 API 的场景（如从 DB 恢复对话）时使用。
    """
    speaker_pattern = re.compile(r"^.+?:")
    lines = convs.split("\n")
    result = []
    in_bot_block = False

    for line in lines:
        if line.startswith(f"{bot_name}:"):
            in_bot_block = True
            print(f"[DEBUG][profile_utils] filter_bot: 进入Bot块: {line[:60]!r}")
            continue
        if in_bot_block and speaker_pattern.match(line) and not line.startswith(f"{bot_name}:"):
            in_bot_block = False
            print(f"[DEBUG][profile_utils] filter_bot: 离开Bot块: {line[:60]!r}")
        if not in_bot_block:
            result.append(line)

    filtered = "\n".join(result)
    original_lines = len(convs.splitlines())
    kept_lines = len(result)
    print(f"[DEBUG][profile_utils] filter_bot: 原{original_lines}行 → 保留{kept_lines}行")
    return filtered


# ── 明确确认选题的模式（必须含"确定/确认/选定/我们的选题是/选题定为"等强确认词）────
import re as _re

_TOPIC_CONFIRM_PATTERNS = [
    # 明确说"选题是/为/定为/确定为"
    _re.compile(r'(?:我们的?|团队的?|最终)?选题(?:是|为|定为|确定为|确定是)[：:]?\s*[""「]?(.+?)[""」]?(?:$|[，。！\n])'),
    # "确定/确认/选定 + 选题/题目/研究方向"
    _re.compile(r'(?:确定|确认|选定|敲定)(?:了)?(?:研究)?(?:选题|题目|课题|研究方向)[：:]?\s*[""「]?(.+?)[""」]?(?:$|[，。！\n])'),
]

# 排除词：含这些词的行不算"确认选题"
_TOPIC_EXCLUDE_KEYWORDS = [
    '帮我', '给我', '有什么', '可以做', '好做', '怎么选',
    '选题建议', '研究方向建议', '有哪些', '能否', '可否', '什么选题',
]

_BOT_SPEAKER_HINTS = (
    "coresearchagent",
    "cosearchagent",
    "assistant",
    "bot",
    "助手",
)


def _looks_like_bot_speaker(speaker: str) -> bool:
    s = (speaker or "").strip().lower()
    if not s:
        return False
    return any(h in s for h in _BOT_SPEAKER_HINTS)


def _split_speaker_content(line: str) -> tuple[str, str]:
    """
    仅在形如 "说话人: 内容" 时拆分，避免把 ":white_check_mark:" 误当成发言者前缀。
    """
    colon_idx = line.find(":")
    if colon_idx <= 0:
        return "", line.strip()

    speaker = line[:colon_idx].strip()
    # 发言者前缀应简短且不含空白，避免将正文中的冒号误判为前缀。
    if len(speaker) > 40 or any(ch.isspace() for ch in speaker):
        return "", line.strip()
    return speaker, line[colon_idx + 1 :].strip()


def extract_latest_topic(convs: str) -> str:
    """
    从对话记录中提取最近一次被明确确认的研究选题。
    
    规则：
    1. 只扫描非Bot发言行（不以 'CoSearchAgent:' 开头）
    2. 必须含明确确认词（确定/确认/选定/选题是/选题为等）
    3. 含"建议/推荐/帮我/给我"等词的行直接排除
    4. 返回最后一次匹配到的选题文本，无则返回空字符串
    """
    if not convs:
        return ""

    def _clean_candidate(text: str) -> str:
        c = (text or "").strip()
        # 去掉包裹引号和多余空白，避免“翻译”这类带引号内容被长度误判。
        c = c.strip('"\'“”‘’「」『』[]【】()（）<>《》')
        c = c.strip().rstrip('。，！.!?？；;：:')
        c = re.sub(r"\s+", " ", c)
        return c

    confirmed_topic = ""
    for line in convs.splitlines():
        line = line.strip()
        if not line:
            continue

        speaker, content = _split_speaker_content(line)

        # 跳过 Bot 发言（显式前缀或发言者名命中 bot 特征）
        if line.startswith("CoSearchAgent:") or line.startswith("助手:"):
            continue
        if _looks_like_bot_speaker(speaker):
            continue

        if not content:
            continue

        # 逐个正则尝试匹配：只要命中明确确认句式，即使包含"建议/推荐系统"等词也应保留。
        matched = False
        for pattern in _TOPIC_CONFIRM_PATTERNS:
            m = pattern.search(content)
            if m:
                candidate = _clean_candidate(m.group(1))
                if candidate and len(candidate) >= 4:  # 太短的不算
                    confirmed_topic = candidate
                    matched = True
                    break  # 本行已匹配，继续下一行

        # 未命中明确确认句式时，含模糊提问词的行直接忽略
        if not matched and any(kw in content for kw in _TOPIC_EXCLUDE_KEYWORDS):
            continue

    return confirmed_topic


# ── 画像变化检测 ──────────────────────────────────────────────────────────────

def profiles_have_changed(old: dict | None, draft: dict) -> bool:
    """
    判断画像是否有实质性变化。
    ★ 已确认 major 的用户，major 字段不参与变化判断（锁定）。

    修复说明：
      原实现先 merge_profile_with_existing 再比较，merge 内部 _dedupe_list 有
      max_count 截断（research_interests/methodology 上限5条，keywords 上限2条），
      导致 existing 已满时新内容进不去 merged，始终判断为无增量。
      修复后直接比较 draft 原始内容与 old 的差集，再用 _is_similar 过滤掉
      语义近似的重复项，只有真正新增的内容才触发变化。
    """
    if old is None:
        print(f"[DEBUG][profile_utils] profiles_have_changed: old=None → True（首次录入）")
        return True

    # major 变化判断：
    # 1) old.major 为空、draft.major 非空 => 首次录入
    # 2) old.major 非空、draft.major 非空且不同 => 方向变更（必须触发确认）
    old_major   = (old.get("major") or "").strip()
    draft_major = (draft.get("major") or "").strip()
    if not old_major and draft_major:
        print(f"[DEBUG][profile_utils] profiles_have_changed: major首次录入 {draft_major!r}")
        return True
    if old_major and draft_major and draft_major != old_major:
        print(f"[DEBUG][profile_utils] profiles_have_changed: major方向变更 {old_major!r} -> {draft_major!r}")
        return True

    # 数组字段：直接用 draft 原始内容与 old 比较，不经过 merge 截断
    for field_name in ("research_interests", "methodology", "keywords"):
        old_set   = set(old.get(field_name) or [])
        draft_set = set(draft.get(field_name) or [])
        raw_new   = draft_set - old_set
        if not raw_new:
            print(f"[DEBUG][profile_utils] profiles_have_changed: {field_name} "
                  f"old={old_set} draft={draft_set} → 无新增")
            continue
        # 过滤掉与已有项语义相似的条目，避免近义词重复触发确认卡片
        truly_new = [
            item for item in raw_new
            if not any(_is_similar(item, x) for x in old_set)
        ]
        print(f"[DEBUG][profile_utils] profiles_have_changed: {field_name} "
              f"raw_new={raw_new} truly_new={truly_new}")
        if truly_new:
            return True

    print(f"[DEBUG][profile_utils] profiles_have_changed: 无增量 → False")
    return False


# ── 按用户过滤发言行 ─────────────────────────────────────────────────────────

def _extract_user_lines(user_only_convs: str, uname: str) -> str:
    """
    从纯用户对话中只抽取目标用户自己发言的行。
    格式为 "说话人: 内容"，多行内容归属于同一说话人直到下一个"说话人:"出现。
    只把目标用户自己说过的话传给 LLM，彻底排除其他用户内容的干扰。
    """
    lines = user_only_convs.split("\n")
    result = []
    current_speaker = None
    current_lines = []

    for line in lines:
        colon_idx = line.find(": ")
        if colon_idx > 0:
            speaker_candidate = line[:colon_idx].strip()
            # 说话人字段：无空格或合理短名（排除URL/句子误判）
            if len(speaker_candidate) <= 30:
                if current_speaker is not None and current_speaker == uname:
                    result.extend(current_lines)
                current_speaker = speaker_candidate
                current_lines = [line]
                continue
        if current_speaker is not None:
            current_lines.append(line)

    # 处理最后一段
    if current_speaker == uname:
        result.extend(current_lines)

    filtered = "\n".join(result).strip()
    print(f"[DEBUG][profile_utils] _extract_user_lines [{uname}]: "
          f"总行数={len(lines)} 该用户行数={len(result)} "
          f"预览: {filtered[:150]!r}")
    return filtered


# ── 批量提炼画像草稿 ──────────────────────────────────────────────────────────

def draft_profiles_from_convs(agent: Any, user_only_convs: str,
                               active_users: list[tuple]) -> list[dict]:
    """
    从纯用户对话中为每个活跃用户提炼画像草稿，只提炼不写库。

    ★ 关键修复：每个用户提炼时只传入该用户自己的发言行，
      彻底避免 LLM 从其他用户的大段内容中误提炼画像。
    ⚠ user_only_convs 必须是不含 Bot 回复的纯用户对话。
    """
    print(f"[DEBUG][profile_utils] draft_profiles: 开始，用户数={len(active_users)}")
    print(f"[DEBUG][profile_utils] draft_profiles: 全部对话前300字:\n{user_only_convs[:300]}")

    profiles = []
    for uid, uname in active_users:
        # ★ 核心修复：只取目标用户自己的发言行
        user_lines = _extract_user_lines(user_only_convs, uname)

        if not user_lines:
            print(f"[DEBUG][profile_utils] draft_profiles: ✗ [{uname}] 无发言记录，跳过")
            continue

        print(f"[DEBUG][profile_utils] draft_profiles: 提炼 [{uname}]（仅本人发言）...")
        extracted, elapsed = agent.extract_user_profile(user=uname, convs=user_lines)

        has_content = (
            extracted.get("major")
            or extracted.get("research_interests")
            or extracted.get("methodology")
            or extracted.get("keywords")
        )

        if has_content:
            extracted.update({"user_id": uid, "user_name": uname})
            profiles.append(extracted)
            print(f"[DEBUG][profile_utils] draft_profiles: ✓ [{uname}]({elapsed:.2f}s)"
                  f" major={extracted.get('major')!r}"
                  f" interests={extracted.get('research_interests')}"
                  f" methods={extracted.get('methodology')}"
                  f" keywords={extracted.get('keywords')}")
        else:
            print(f"[DEBUG][profile_utils] draft_profiles: ✗ [{uname}]({elapsed:.2f}s) "
                  f"发言无学术信息，跳过")

    print(f"[DEBUG][profile_utils] draft_profiles: 完成，共 {len(profiles)} 份有效草稿")
    return profiles


# ── 字符串相似度判断（用于去重）──────────────────────────────────────────────

def _is_similar(a: str, b: str) -> bool:
    """
    判断两个字符串是否相似（用于去重）。
    规则：一方包含另一方，或者去除常见后缀后相同。
    """
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    
    if not a_lower or not b_lower:
        return False
    
    # 完全相同
    if a_lower == b_lower:
        return True
    
    # 一方包含另一方
    if a_lower in b_lower or b_lower in a_lower:
        return True
    
    # 去除常见后缀后比较（如"绘图"vs"绘画"）
    suffixes = ["设计", "研究", "分析", "方法", "技术", "能力"]
    a_core = a_lower
    b_core = b_lower
    for suffix in suffixes:
        if a_core.endswith(suffix) and len(a_core) > len(suffix):
            a_core = a_core[:-len(suffix)]
        if b_core.endswith(suffix) and len(b_core) > len(suffix):
            b_core = b_core[:-len(suffix)]
    
    if a_core == b_core:
        return True

    # 语义归一化：将常见同义表达折叠后再比较，减少“同主题不同说法”反复触发更新。
    def _normalize_semantic(text: str) -> str:
        t = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", (text or "").lower())
        replacements = {
            "人工智能": "ai",
            "chatgpt": "ai",
            "chatbot": "聊天助手",
            "聊天机器人": "聊天助手",
            "对话系统": "聊天助手",
            "人机交互": "交互",
            "用户心理": "心理",
        }
        for src, dst in replacements.items():
            t = t.replace(src, dst)

        # 去掉高频泛词，保留主题核心。
        stop_tokens = [
            "研究", "方向", "相关", "基于", "影响", "作用", "用户", "系统",
            "助手", "聊天", "对", "与", "和", "的", "中", "在", "交互",
        ]
        for s in stop_tokens:
            t = t.replace(s, "")
        return t.strip()

    a_norm = _normalize_semantic(a_lower)
    b_norm = _normalize_semantic(b_lower)
    if a_norm and b_norm and (a_norm == b_norm or a_norm in b_norm or b_norm in a_norm):
        return True

    # 字符二元组 Jaccard：兜底识别中文近义变体。
    def _char_bigrams(text: str) -> set[str]:
        if len(text) < 2:
            return {text} if text else set()
        return {text[i:i+2] for i in range(len(text) - 1)}

    if a_norm and b_norm:
        a_bg = _char_bigrams(a_norm)
        b_bg = _char_bigrams(b_norm)
        if a_bg and b_bg:
            overlap = len(a_bg & b_bg)
            union = len(a_bg | b_bg)
            if union > 0 and (overlap / union) >= 0.6:
                return True
    
    # 特殊相似词组
    similar_groups = [
        {"绘图", "绘画", "画图", "作图"},
        {"写作", "论文写作", "文章写作", "撰写"},
        {"实验", "做实验", "实验设计", "实验操作"},
        {"数据分析", "数据处理", "数据收集"},
        {"文献综述", "文献分析", "文献研究"},
    ]
    for group in similar_groups:
        if a_lower in group and b_lower in group:
            return True
        # 部分匹配
        a_in_group = any(a_lower in item or item in a_lower for item in group)
        b_in_group = any(b_lower in item or item in b_lower for item in group)
        if a_in_group and b_in_group:
            return True
    
    return False


def _dedupe_list(items: list[str], max_count: int = 5) -> list[str]:
    """
    对列表进行去重，保留更简洁的表达。
    max_count: 最多保留的条目数。
    """
    if not items:
        return []
    
    result = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        
        # 检查是否与已有项相似
        is_dup = False
        for i, existing in enumerate(result):
            if _is_similar(item, existing):
                # 保留更短的（更简洁的）
                if len(item) < len(existing):
                    result[i] = item
                is_dup = True
                break
        
        if not is_dup:
            result.append(item)
        
        if len(result) >= max_count:
            break
    
    return result


# ── 画像合并 ──────────────────────────────────────────────────────────────────

def merge_profile_with_existing(existing: dict | None, draft: dict) -> dict:
    """
    合并策略：以数据库已有值为基础，将草稿中的新内容补充进去。

    ★ 核心变更：已确认（major非空）的用户，major 字段锁定，
      只合并 research_interests / methodology / keywords 的新增项。
    """
    if existing is None:
        result = draft.copy()
        for field_name in ("research_interests", "methodology", "keywords"):
            result[field_name] = _dedupe_list(result.get(field_name) or [], max_count=5)
        print(f"[DEBUG][profile_utils] merge: existing=None，直接使用 draft（已去重）")
        return result

    merged = existing.copy()

    draft_major    = (draft.get("major") or "").strip()
    existing_major = (merged.get("major") or "").strip()
    direction_changed = bool(existing_major and draft_major and draft_major != existing_major)

    # ── Step A: draft.keywords 中像研究方向的词，在截断前先提升到 research_interests
    # 必须在 keywords 的 _dedupe_list 截断之前执行，否则排在第6位的词会被截掉。
    # 判断规则：keyword 包含学科/方向词根，且与已有 research_interests 不相似，则提升。
    _RESEARCH_DIRECTION_ROOTS = [
        "法", "律", "法学", "经济", "金融", "财税", "政策", "监管", "治理",
        "数据", "隐私", "权利", "权益", "制度", "规制", "合规", "诉讼", "仲裁",
        "国际", "比较", "跨境", "欧洲", "美国", "亚洲", "全球",
        "环境", "气候", "碳", "ESG", "可持续",
        "刑事", "民事", "行政", "宪法", "商法", "劳动", "知识产权",
        "人工智能", "算法", "区块链", "科技", "信息",
    ]

    # 用 existing 的 research_interests 作为基准（此时 merged 还未更新 ri）
    base_ri        = list(existing.get("research_interests") or [])
    draft_kws_raw  = list(draft.get("keywords") or [])
    promoted_kws   = []
    remaining_draft_kws = []

    for kw in draft_kws_raw:
        is_direction  = any(root in kw for root in _RESEARCH_DIRECTION_ROOTS)
        already_in_ri = any(_is_similar(kw, x) for x in base_ri)
        if is_direction and not already_in_ri:
            promoted_kws.append(kw)
            base_ri.append(kw)   # 防止同批次重复提升
        else:
            remaining_draft_kws.append(kw)

    if promoted_kws:
        print(f"[DEBUG][profile_utils] merge: keywords→research_interests 提升: {promoted_kws}")

    # 方向变更：优先采用新 major 与新证据，避免旧领域画像污染
    if direction_changed:
        merged["major"] = draft_major
        print(f"[DEBUG][profile_utils] merge: 检测到方向变更，major采用新值 {draft_major!r}")

        ri_new = list(draft.get("research_interests") or []) + promoted_kws
        meth_new = list(draft.get("methodology") or [])
        kw_new = remaining_draft_kws

        merged["research_interests"] = _dedupe_list(ri_new, max_count=8)
        merged["methodology"] = _dedupe_list(meth_new, max_count=5)
        merged["keywords"] = _dedupe_list(kw_new, max_count=5)

        # 若新证据过少，才回退保留少量旧值兜底
        if not merged["research_interests"]:
            merged["research_interests"] = _dedupe_list(list(existing.get("research_interests") or []), max_count=3)
        if not merged["methodology"]:
            merged["methodology"] = _dedupe_list(list(existing.get("methodology") or []), max_count=3)
        if not merged["keywords"]:
            merged["keywords"] = _dedupe_list(list(existing.get("keywords") or []), max_count=3)

        merged["user_id"]   = draft.get("user_id",   existing.get("user_id", ""))
        merged["user_name"] = draft.get("user_name", existing.get("user_name", ""))
        return merged

    # ★ 已有确认画像且非方向变更时，major 锁定不动
    if existing_major:

        # research_interests：合并 existing + draft.ri + 提升词，上限8条
        ri_combined = (list(existing.get("research_interests") or [])
                       + list(draft.get("research_interests") or [])
                       + promoted_kws)
        merged["research_interests"] = _dedupe_list(ri_combined, max_count=8)
        if merged["research_interests"] != list(existing.get("research_interests") or []):
            print(f"[DEBUG][profile_utils] merge: research_interests 合并 → "
                  f"{merged['research_interests']}")

        # methodology：正常合并，上限5条
        meth_combined = (list(existing.get("methodology") or [])
                         + list(draft.get("methodology") or []))
        merged["methodology"] = _dedupe_list(meth_combined, max_count=5)
        if merged["methodology"] != list(existing.get("methodology") or []):
            print(f"[DEBUG][profile_utils] merge: methodology 合并 → {merged['methodology']}")

        # keywords：只用 remaining_draft_kws（已排除提升词），过滤与 ri/major 重복的
        kw_combined = (list(existing.get("keywords") or []) + remaining_draft_kws)

    else:
        # 首次录入，major 可以被 draft 填充
        if draft_major:
            merged["major"] = draft_major
            print(f"[DEBUG][profile_utils] merge: major 首次录入 {draft_major!r}")

        ri_combined = (list(existing.get("research_interests") or [])
                       + list(draft.get("research_interests") or [])
                       + promoted_kws)
        merged["research_interests"] = _dedupe_list(ri_combined, max_count=8)
        if len(merged["research_interests"]) != len(existing.get("research_interests") or []):
            print(f"[DEBUG][profile_utils] merge: research_interests 合并 → "
                  f"{merged['research_interests']}")

        meth_combined = (list(existing.get("methodology") or [])
                         + list(draft.get("methodology") or []))
        merged["methodology"] = _dedupe_list(meth_combined, max_count=5)
        if len(merged["methodology"]) != len(existing.get("methodology") or []):
            print(f"[DEBUG][profile_utils] merge: methodology 合并 → {merged['methodology']}")

        kw_combined = (list(existing.get("keywords") or []) + remaining_draft_kws)

    print(f"[DEBUG][profile_utils] merge: research_interests最终={merged.get('research_interests')}")

    # ── Step B: keywords 用新监测到的替换旧的 ────────────────────────────────────
    # 策略：优先保留 remaining_draft_kws（本轮新监测），旧 keywords 补位填满上限。
    # 过滤掉与 research_interests / major 相似的冗余词，只保留真正独特的关键词。
    major_str          = (merged.get("major") or "").lower()
    final_ri           = merged.get("research_interests") or []
    existing_kws_clean = list(existing.get("keywords") or [])

    def _kw_is_valid(kw):
        if kw.lower() == major_str:
            return False
        if any(_is_similar(kw, x) for x in final_ri):
            return False
        return True

    # 先放新监测到的（remaining_draft_kws），再用旧 keywords 补位
    new_kws = [kw for kw in remaining_draft_kws if _kw_is_valid(kw)]
    old_kws = [kw for kw in existing_kws_clean  if _kw_is_valid(kw)]

    final_kws = _dedupe_list(new_kws + old_kws, max_count=5)
    print(f"[DEBUG][profile_utils] merge: keywords new={new_kws} old_fallback={old_kws} 最终={final_kws}")

    merged["keywords"] = final_kws

    merged["user_id"]   = draft.get("user_id",   existing.get("user_id",   ""))
    merged["user_name"] = draft.get("user_name", existing.get("user_name", ""))

    return merged


# ── 画像更新通知 ──────────────────────────────────────────────────────────────

def notify_profile_update_if_changed(
    client, channel_id: str, user_id: str,
    existing: dict | None, new_draft: dict,
    profile_memory,
) -> bool:
    """
    检测画像是否有实质增量，若有则：
      1. 【不】静默保存——必须等用户确认后才写库
      2. 发送确认卡片，让用户决定是否接受更新

    返回 True 表示已发出通知，False 表示无增量。
    """
    from handlers.profile_confirm import send_profile_confirm_card

    if not profiles_have_changed(existing, new_draft):
        return False

    # 判断是否发生方向转变（用于通知文案区分）
    draft_major    = (new_draft.get("major") or "").strip()
    existing_major = (existing.get("major") or "").strip() if existing else ""
    direction_changed = bool(draft_major and existing_major and draft_major != existing_major)

    # 合并草稿用于展示（但不写库，等用户确认）
    merged = merge_profile_with_existing(existing, new_draft)
    print(f"[DEBUG][profile_utils] notify_profile_update: 检测到增量，等待用户确认 "
          f"major={merged.get('major')!r} interests={merged.get('research_interests')} "
          f"direction_changed={direction_changed}")

    reason = "方向变更" if direction_changed else "画像更新"

    # 同一用户在极短时间内只发一张确认卡，避免并发路径造成双卡。
    if _profile_notify_in_cooldown(channel_id, user_id):
        return False

    send_profile_confirm_card(client, channel_id, user_id, merged, reason=reason)
    return True