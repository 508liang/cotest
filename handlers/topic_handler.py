"""
handlers/topic_handler.py

修改记录：
  1. [画像污染] TopicContext 新增 user_only_convs 字段；handle_topic_intent 内
     直接使用 ctx.user_only_convs 而非再次调 get_user_only_conversation_history，
     保证来源唯一、与 app 层一致。
  2. [Debug]   全链路添加 [DEBUG][topic_handler] 前缀日志，覆盖每个决策分支。
  3. [内容质量] _format_references 改为"标题+摘要+来源URL"格式；
               _execute_topic 里搜索词来源（LLM改写 vs 规则兜底）以及
               每条搜索词的命中结果数均有日志输出；
               keywords 使用画像中的 research_interests/keywords 字段补充，
               使 _rank_results 评分更精准；
               _format_references 上限从8提到12。
"""

import time
import http.client
from dataclasses import dataclass, field
from typing import Any, Optional

from memory.imm_profile_store import ImmProfileStore
from utils import resolve_user_name
from handlers.profile_utils import (
    draft_profiles_from_convs,
    profiles_have_changed,
    merge_profile_with_existing,
    notify_profile_update_if_changed,   # ⬅ 新增
)

# ── 权威来源加分域名 ──────────────────────────────────────────────────────────
AUTHORITATIVE_DOMAINS = [
    "arxiv.org", "scholar.google", "springer.com", "sciencedirect.com",
    "ieee.org", "acm.org", "nature.com", "wiley.com", "tandfonline.com",
    "cnki.net", "wanfangdata", "ssrn.com", "researchgate.net",
    "pubmed.ncbi.nlm.nih.gov", "link.springer.com", "dl.acm.org",
]

REVIEW_KEYWORDS = [
    "综述", "热点", "前沿", "survey", "review", "overview",
    "趋势", "进展", "meta-analysis", "systematic review",
]

RECENT_YEARS = ["2023", "2024", "2025", "2026"]


# ── 搜索结果评分 ───────────────────────────────────────────────────────────────

def _score_result(result: dict, keywords: list[str]) -> int:
    score = 0
    link    = result.get("link", "").lower()
    title   = result.get("title", "").lower()
    snippet = (result.get("snippet", "") or "").lower()
    if any(d in link for d in AUTHORITATIVE_DOMAINS):
        score += 3
    kw_hits = sum(1 for kw in keywords if kw.lower() in title or kw.lower() in snippet)
    score += min(kw_hits, 4)                          # 上限4分（原来3分）
    if len(snippet) > 100: score += 1                 # 摘要充实
    if any(s in title for s in REVIEW_KEYWORDS): score += 2
    if any(y in snippet for y in RECENT_YEARS): score += 1
    if result.get("link", "").startswith("https"): score += 1   # 安全链接轻加分
    return score


def _rank_results(results: list[dict], keywords: list[str], top_k: int = 12) -> list[dict]:
    valid = [r for r in results if r.get("title") and r.get("snippet")]
    scored = [(_score_result(r, keywords), r) for r in valid]
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"[DEBUG][topic_handler] 评分排序（共{len(scored)}条）：")
    for score, r in scored[:top_k]:
        print(f"[DEBUG][topic_handler]   [{score:2d}分] {r.get('title','')[:60]}")
    return [r for _, r in scored[:top_k]]


# ── 规则兜底查询词生成 ─────────────────────────────────────────────────────────

def _fallback_queries(majors: list[str], query: str, convs: str) -> list[str]:
    tech = [
        "大模型", "LLM", "NLP", "人工智能", "机器学习", "深度学习",
        "区块链", "知识图谱", "数字化", "算法", "数据驱动",
    ]
    extra = [kw for kw in tech if kw in f"{convs}\n{query}"]

    if len(majors) >= 2:
        cross = " ".join(majors[:2])
        qs = [
            f"{cross} 交叉研究 热点方向 2026",
            f"{cross} 跨学科 研究综述 2026",
            f"{majors[0]} {majors[1]} 结合 应用场景 案例",
            f"{majors[0]} 研究空白 未来方向",
        ]
        if extra:
            qs.append(f"{majors[0]} {extra[0]} 研究前沿 2025")
    elif len(majors) == 1:
        qs = [
            f"{majors[0]} 研究热点 2025",
            f"{majors[0]} 研究空白 未来方向",
            f"{majors[0]} 前沿问题 综述" if not extra else f"{majors[0]} {extra[0]} 应用研究",
        ]
    else:
        core = query[:40].strip()
        qs = [
            f"{core} 研究热点 综述",
            f"{core} 研究方向 2025",
        ]
    return qs


# ── 参考文献格式化（标题 + 摘要 + 来源，供 LLM 引用）────────────────────────────

def _format_references(results: list[dict], max_items: int = 12) -> str:
    lines = []
    for i, r in enumerate(results[:max_items], start=1):
        title   = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        link    = r.get("link", "").strip()
        lines.append(f"[{i}] 标题：{title}\n    摘要：{snippet}\n    来源：{link}")
    return "\n\n".join(lines)


# ── 从画像提取关键词用于评分 ──────────────────────────────────────────────────

def _extract_profile_keywords(profiles: list[dict]) -> list[str]:
    keywords = []
    for p in profiles:
        keywords.extend(p.get("research_interests") or [])
        keywords.extend(p.get("keywords") or [])
        if p.get("major"):
            keywords.append(p["major"])
    return list(dict.fromkeys(keywords))   # 去重保序


# ── DataClass ────────────────────────────────────────────────────────────────

@dataclass
class TopicContext:
    client: Any
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    ts: str
    query: str
    convs: str           # 含Bot回复的完整对话，用于上下文理解
    intent_time: float
    agent: Any
    search_engine: Any
    memory: Any
    search_memory: Any
    profile_memory: Any
    user_id2names: dict
    bot_id: str
    sql_password: str
    # ⬇ 新增：纯用户对话（不含Bot回复），专用于画像提炼
    # app层传入 get_user_only_conversation_history() 的结果
    user_only_convs: str = field(default="")
    is_new_channel: bool = field(default=False)   # ⬅ 新增
    # ★ 新增：当前频道实际发言的用户名列表（由 memory.get_speakers_in_channel 提供）
    channel_speakers: list = field(default_factory=list)
    # ★ 新增：当前频道参与用户 ID 列表
    active_user_ids: list = field(default_factory=list)
    # ★ 新增：统一输入上下文（IMM+SMM+近5轮+query）
    imm_smm_context: str = field(default="")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def handle_topic_intent(ctx: TopicContext, user_profiles_text: str = ""):
    """
    选题意图主处理函数。
    user_profiles_text: 若由 profile_confirm 回调传入，则跳过画像提炼直接使用。
    """
    from handlers.profile_confirm import send_profile_confirm_card
    from memory.pending_intent_memory import PendingIntentMemory

    print(f"[DEBUG][topic_handler] ── handle_topic_intent 开始 ──")
    print(f"[DEBUG][topic_handler] user={ctx.user_name!r} channel={ctx.channel_name!r}")
    print(f"[DEBUG][topic_handler] query={ctx.query!r}")

    # 新链路：若上层已提供 IMM+SMM 统一上下文，直接执行，不再走 profile 相关流程。
    if ctx.imm_smm_context:
        print("[DEBUG][topic_handler] 使用 IMM+SMM 统一上下文，跳过 profile 提炼/确认流程")
        _execute_topic(ctx, user_profiles_text=ctx.imm_smm_context)
        return

    # 若由确认回调传入 user_profiles_text，跳过画像提炼直接执行
    if user_profiles_text:
        print(f"[DEBUG][topic_handler] 由确认回调传入 user_profiles_text，直接执行选题")
        _execute_topic(ctx, user_profiles_text=user_profiles_text)
        return

    pending_memory = PendingIntentMemory(sql_password=ctx.sql_password)
    pending_memory.create_table_if_not_exists()

    # ── Step 1: 提炼画像草稿（严格使用纯用户对话）─────────────────────────────
    user_only = ctx.user_only_convs
    if not user_only:
        from utils import get_user_only_conversation_history
        print(f"[DEBUG][topic_handler] ⚠ ctx.user_only_convs 为空，降级调用 API（不推荐）")
        user_only = get_user_only_conversation_history(
            client=ctx.client, channel_id=ctx.channel_id,
            bot_id=ctx.bot_id, user_id2names=ctx.user_id2names, ts=ctx.ts, limit=50
        )

    profile_source = f"{user_only}\n{ctx.query}"
    print(f"[DEBUG][topic_handler] 画像提炼输入行数={len(profile_source.splitlines())} "
          f"前200字:\n{profile_source[:200]}")

    # ── 严格限定 active_users 为当前频道非Bot参与用户 ─────────────────────────
    allowed_ids = [
        uid for uid in (ctx.active_user_ids or [])
        if uid and uid not in (ctx.bot_id, "bot_id")
    ]
    active_users = []
    for uid in allowed_ids:
        uname = ctx.user_id2names.get(uid, uid)
        if uname == uid:
            try:
                uname = resolve_user_name(
                    client=ctx.client,
                    user_id=uid,
                    user_id2names=ctx.user_id2names,
                    sql_password=ctx.sql_password,
                )
            except Exception:
                pass
        active_users.append((uid, uname))
    if not active_users:
        active_users = [(ctx.user_id, ctx.user_name)]
    print(f"[DEBUG][topic_handler] 当前频道参与的非Bot用户: {[(u, n) for u, n in active_users]}")

    drafts = draft_profiles_from_convs(ctx.agent, profile_source, active_users)
    print(f"[DEBUG][topic_handler] 提炼到画像草稿 {len(drafts)} 份")

    # ── Step 2: 检测当前用户的画像是否有增量变化 ─────────────────────────────
    existing = ctx.profile_memory.load(ctx.user_id, ctx.channel_id)
    my_draft = next((p for p in drafts if p.get("user_id") == ctx.user_id), None)

    # ★ 核心变更：如果当前用户画像有增量，先挂起意图，让用户确认后再执行
    if my_draft and profiles_have_changed(existing, my_draft):
        merged_profile = merge_profile_with_existing(existing, my_draft)
        print(f"[DEBUG][topic_handler] ★ 检测到当前用户画像有增量，先挂起意图等待确认")
        print(f"[DEBUG][topic_handler]   existing.major={existing.get('major') if existing else None!r}")
        print(f"[DEBUG][topic_handler]   merged.major={merged_profile.get('major')!r}")

        # 保存挂起意图
        pending_memory.save(
            user_id=ctx.user_id,
            channel_id=ctx.channel_id,
            intent_label="【选题】",
            payload={
                "channel_id": ctx.channel_id, "channel_name": ctx.channel_name,
                "user_name": ctx.user_name, "ts": ctx.ts,
                "query": ctx.query, "convs": ctx.convs, "intent_time": ctx.intent_time,
                "bot_id": ctx.bot_id,
                "agent": "__global__", "search_engine": "__global__",
                "memory": "__global__", "search_memory": "__global__",
                "active_user_ids": ctx.active_user_ids or [],
            }
        )

        # 发送画像确认卡片
        reason = "方向变更" if (
            existing and existing.get("major") and merged_profile.get("major")
            and existing.get("major") != merged_profile.get("major")
        ) else "画像更新"
        send_profile_confirm_card(ctx.client, ctx.channel_id, ctx.user_id,
                                   merged_profile, reason=reason)
        print(f"[DEBUG][topic_handler] 已发送画像确认卡片，等待用户确认后继续执行选题")
        return

    # ── Step 3: 其他用户的画像增量 → 静默通知（不阻断当前用户流程）─────────────
    for d in drafts:
        uid = d.get("user_id")
        if not uid or uid == ctx.user_id:
            continue  # 当前用户已在上面处理
        existing_d = ctx.profile_memory.load(uid, ctx.channel_id)
        notified = notify_profile_update_if_changed(
            client=ctx.client,
            channel_id=ctx.channel_id,
            user_id=uid,
            existing=existing_d,
            new_draft=d,
            profile_memory=ctx.profile_memory,
        )
        print(f"[DEBUG][topic_handler] 其他用户画像通知 [{d.get('user_name')}]: notified={notified}")

    # ── Step 4: 组装所有用户画像，直接执行 ────────────────────────────────────
    all_profiles = []
    for uid, uname in active_users:
        p = ctx.profile_memory.load(uid, ctx.channel_id)
        if not p:
            p = next((d for d in drafts if d.get("user_id") == uid), None)
        if p and (p.get("major") or p.get("research_interests") or p.get("methodology")):
            all_profiles.append(p)
            print(f"[DEBUG][topic_handler] 纳入画像 [{uname}]:"
                  f" major={p.get('major')!r}"
                  f" interests={p.get('research_interests')}"
                  f" methods={p.get('methodology')}")
        else:
            print(f"[DEBUG][topic_handler] 跳过画像 [{uname}]（无实质内容）")

    profiles_text = (
        ImmProfileStore.format_for_prompt(all_profiles) if all_profiles
        else f"当前提问者：{ctx.user_name}。可结合对话内容推断学科背景。"
    )
    print(f"[DEBUG][topic_handler] 最终 user_profiles_text:\n{profiles_text[:300]}")
    print(f"[DEBUG][topic_handler] 直接执行选题，画像数量={len(all_profiles)}")
    _execute_topic(ctx, user_profiles_text=profiles_text)


# ── 选题执行（RAG 核心流程）─────────────────────────────────────────────────────

def _execute_topic(ctx: TopicContext, user_profiles_text: str):
    """
    实际执行选题RAG，供确认后回调和直接执行两条路径共用。
    流程：查询改写 → 多轮搜索 → 评分排序 → 格式化参考文献 → LLM生成选题
    """
    from utils import send_rag_answer, send_status_message, delete_status_message, slack_chat_update

    print(f"[DEBUG][topic_handler] ── _execute_topic 开始 ──")
    print(f"[DEBUG][topic_handler] user_profiles_text:\n{user_profiles_text[:300]}")

    # ── Step A: 查询改写（LLM生成多条学术检索词）────────────────────────────
    rewrite_time = 0.0
    search_queries: list[str] = []
    try:
        search_queries, rewrite_time = ctx.agent.rewrite_topic_query(
            query=ctx.query, convs=f"{ctx.user_only_convs}\n{ctx.query}".strip(), user_profiles=user_profiles_text
        )
        print(f"[DEBUG][topic_handler] LLM改写检索词({rewrite_time:.2f}s): {search_queries}")
    except Exception as e:
        print(f"[DEBUG][topic_handler] ⚠ LLM改写失败，规则兜底: {e}")

    # 规则兜底：若LLM改写失败或结果为空
    if not search_queries:
        import re as _re
        patterns = [r"我是(.{2,10}?)专业", r"他是(.{2,10}?)专业", r"她是(.{2,10}?)专业",
                    r"我研究(.{2,10}?)方向",   r"我做(.{2,10}?)研究"]
        majors = list(dict.fromkeys(
            m for p in patterns for m in _re.findall(p, f"{ctx.convs}\n{ctx.query}")
        ))
        search_queries = _fallback_queries(majors, ctx.query, ctx.convs)
        print(f"[DEBUG][topic_handler] 规则兜底检索词: {search_queries}")

    # ── Step B: 多轮搜索 + 实时进度更新 ─────────────────────────────────────
    seen: set[str] = set()
    raw: list[dict] = []
    search_start = time.time()
    search_status_lines: list[str] = []

    total = len(search_queries)

    # ★ 只发一条进度消息，后续只更新不删除
    search_progress_ts = send_status_message(
        ctx.client, ctx.channel_id, ctx.user_id,
        f"🔍 正在搜索第 1/{total} 条…"
    )

    for i, sq in enumerate(search_queries, start=1):
        # ★ 更新同一条消息，不新发不删除
        try:
            slack_chat_update(
                client=ctx.client,
                channel=ctx.channel_id,
                ts=search_progress_ts,
                text=f"<@{ctx.user_id}> 🔍 正在搜索第 {i}/{total} 条：{sq}…",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"<@{ctx.user_id}> 🔍 正在搜索第 {i}/{total} 条：`{sq}`…"
                    }
                }]
            )
        except Exception:
            pass
        print(f"[DEBUG][topic_handler] 🔍 正在搜索第 {i}/{total} 条：`{sq}`…")

        added = 0
        try:
            results = ctx.search_engine.get_search_results(sq)
            for r in results:
                if r.get("link") and r["link"] not in seen \
                        and r.get("title") and r.get("snippet"):
                    seen.add(r["link"])
                    raw.append(r)
                    added += 1
            print(f"[DEBUG][topic_handler] 搜索 {sq!r} → 新增 {added} 条（累计 {len(raw)} 条）")
        except Exception as e:
            print(f"[DEBUG][topic_handler] ⚠ 搜索失败 [{sq!r}]: {e}")

        search_status_lines.append(f"• `{sq}` → {added} 条结果")

    # ★ 所有搜索完成后才删除进度消息
    delete_status_message(ctx.client, ctx.channel_id, search_progress_ts)

    search_elapsed = time.time() - search_start
    print(f"[DEBUG][topic_handler] 搜索阶段完成({search_elapsed:.2f}s)，原始结果 {len(raw)} 条")

    if not raw:
        print(f"[DEBUG][topic_handler] ⚠ 所有搜索均无结果，直接用 query 兜底搜索")
        raw = ctx.search_engine.get_search_results(ctx.query)

    # ── Step C: 评分排序（利用画像关键词提升精准度）──────────────────────────
    # 从 user_profiles_text 不便二次解析，直接从检索词 + 画像关键词提取
    profile_kws = [kw for sq in search_queries for kw in sq.split()]
    search_results = _rank_results(raw, profile_kws, top_k=12)
    print(f"[DEBUG][topic_handler] 排序后保留 {len(search_results)} 条结果")

    # ── Step D: 格式化参考文献（含来源URL，让 LLM 引用更可信）──────────────
    refs_text = _format_references(search_results, max_items=12)
    print(f"[DEBUG][topic_handler] 参考文献文本（前500字）:\n{refs_text[:500]}")

    # ── Step E: 流式生成选题建议 ★ ───────────────────────────────────────────
    topics, topics_time, response_ts = ctx.agent.propose_topics_stream(
        query=ctx.query,
        convs=f"{ctx.user_only_convs}\n{ctx.query}".strip(),
        user_profile=user_profiles_text,
        references=refs_text,
        client_slack=ctx.client,
        channel_id=ctx.channel_id,
        user_id=ctx.user_id,
        search_status_lines=search_status_lines,
        search_results=search_results,   # ★ 传入原始结果供来源卡片使用
    )
    print(f"[DEBUG][topic_handler] 选题生成完成({topics_time:.2f}s)")

    # ── Step F: 写库（不再需要 send_rag_answer，ts 已由流式消息获得）─────────
    ctx.memory.write_into_memory(
        table_name=ctx.channel_name,
        utterance_info={
            "speaker":         "CoSearchAgent",
            "utterance":       topics,
            "convs":           ctx.convs,
            "query":           ctx.query,
            "rewrite_query":   " | ".join(search_queries),
            "rewrite_thought": "",
            "clarify":         "",
            "clarify_thought": "",
            "clarify_cnt":     0,
            "search_results":  str(search_results),
            "infer_time":      str({
                "intent":  ctx.intent_time,
                "rewrite": rewrite_time,
                "search":  search_elapsed,
                "topics":  topics_time,
            }),
            "reply_timestamp": ctx.ts,
            "reply_user":      ctx.user_name,
            "timestamp":       response_ts,
        },
    )
    ctx.search_memory.create_table_if_not_exists(table_name=f"{ctx.channel_name}_search")
    ctx.search_memory.write_into_memory(
        table_name=f"{ctx.channel_name}_search",
        search_info={
            "user_name":      ctx.user_name,
            "query":          ctx.query,
            "answer":         topics,
            "search_results": str(search_results),
            "start":          0,
            "end":            2,
            "click_time":     time.time(),
            "timestamp":      response_ts,
        },
    )
    print(f"[DEBUG][topic_handler] ── _execute_topic 完成 ──")