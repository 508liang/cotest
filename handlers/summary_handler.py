"""
handlers/summary_handler.py

【总结】意图处理器：提炼群聊完整对话的共识、行动项与讨论状态。

与选题/分工不同，总结不需要用户画像确认流程，直接读取全量对话记忆后生成。

修改记录：
  1. [选题感知]   在粗度总结生成前，利用 extract_latest_topic 从完整对话历史中检测已确认选题；
                 若检测到选题，将其以结构化前缀注入到 summarize_convs 的 convs 参数中，
                 使 LLM 能在明确的选题框架下生成更有针对性的总结，减少幻觉。
  2. [总结粒度]   新增粗度/细度总结分流：
                 - 触发总结意图后，先调用 agent.classify_summary_granularity(query)
                   判断本次请求是【粗度总结】还是【细度总结】。
                 - 【粗度总结】：读取 full_convs + 选题注入后调用 agent.summarize_convs()，
                   使用 summary.txt prompt。
                 - 【细度总结】：从 full_convs 中筛选与指定话题相关的行，调用
                   agent.summarize_focused()，使用 summary_focused.txt prompt。
                   若相关内容不足 2 行，直接回复提示，不调用 LLM。
  3. [筛选窗口]   关键词匹配大小写不敏感；上下文窗口改为向前 2 行 + 向后 5 行，
                 确保 bot 多行长回复能被完整捞入，不再只取 ±1 行。
  4. [Debug]     全链路添加 [DEBUG][summary_handler] 前缀日志，覆盖粒度识别、选题检测、
                 对话筛选各阶段的详细状态。
"""

import re
import time
import os
from dataclasses import dataclass
from typing import Any

from handlers.profile_utils import extract_latest_topic


def _summary_debug_verbose() -> bool:
    return os.getenv("SUMMARY_DEBUG_VERBOSE", "0").strip() in {"1", "true", "yes", "on"}


@dataclass
class SummaryContext:
    client: Any
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    ts: str
    query: str
    convs: str          # get_conversation_history 返回的近期对话（用于意图分类上下文）
    intent_time: float
    agent: Any
    memory: Any


# ── 主入口 ────────────────────────────────────────────────────────────────────

def handle_summary_intent(ctx: SummaryContext):
    """
    总结流程：
      1. 从数据库读取该频道的完整对话记忆
      2. 调用 LLM 判断本次 query 是【粗度总结】还是【细度总结】
      3a. 粗度总结：检测已确认选题并注入上下文 -> agent.summarize_convs()
      3b. 细度总结：从 full_convs 中筛选话题相关行 -> agent.summarize_focused()
      4. 发送结果并写入记忆
    """
    from utils import send_answer

    print(f"[DEBUG][summary_handler] -- handle_summary_intent 开始 --")
    print(f"[DEBUG][summary_handler] user={ctx.user_name!r} channel={ctx.channel_name!r}")
    print(f"[DEBUG][summary_handler] query={ctx.query!r}")

    # Step 1: 读取完整对话记忆
    full_convs = _load_full_convs(ctx)
    print(f"[DEBUG][summary_handler] 读取完整对话记忆，共 {len(full_convs.splitlines())} 行")

    # Step 2: 判断粒度（粗度 / 细度）
    granularity, topic, granularity_time = ctx.agent.classify_summary_granularity(ctx.query)
    print(f"[DEBUG][summary_handler] ★ 总结粒度识别结果: granularity={granularity!r} "
          f"topic={topic!r} 耗时={granularity_time:.2f}s")

    # Step 3: 按粒度分流
    if granularity == "focused":
        print(f"[DEBUG][summary_handler] -> 进入【细度总结】分支，话题={topic!r}")
        summary, summary_time = _handle_focused_summary(ctx, full_convs, topic)
    else:
        print(f"[DEBUG][summary_handler] -> 进入【粗度总结】分支")
        summary, summary_time = _handle_broad_summary(ctx, full_convs)

    if summary is None:
        # 细度总结内容不足时已在内部发送提示，直接返回
        print(f"[DEBUG][summary_handler] 细度总结内容不足，流程提前结束")
        return

    print(f"[DEBUG][summary_handler] 总结生成完成 (耗时: {summary_time:.2f}s)")

    # Step 4: 发送并写库
    response_ts = send_answer(
        client=ctx.client,
        channel_id=ctx.channel_id,
        user_id=ctx.user_id,
        answer=summary,
    )["ts"]

    ctx.memory.write_into_memory(
        table_name=ctx.channel_name,
        utterance_info={
            "speaker":         "CoSearchAgent",
            "utterance":       summary,
            "convs":           ctx.convs,
            "query":           ctx.query,
            "rewrite_query":   ctx.query,
            "rewrite_thought": "",
            "clarify":         "",
            "clarify_thought": "",
            "clarify_cnt":     0,
            "search_results":  "",
            "infer_time":      str({
                "intent":      ctx.intent_time,
                "granularity": granularity_time,
                "summary":     summary_time,
            }),
            "reply_timestamp": ctx.ts,
            "reply_user":      ctx.user_name,
            "timestamp":       response_ts,
        },
    )

    print(f"[DEBUG][summary_handler] -- handle_summary_intent 完成 --")


# ── 粗度总结 ──────────────────────────────────────────────────────────────────

def _handle_broad_summary(ctx: SummaryContext, full_convs: str):
    """
    粗度总结：
      - 检测已确认选题，若存在则注入到 convs 前缀，帮助 LLM 锚定主题，减少幻觉
      - 调用 agent.summarize_convs()，使用 summary.txt prompt
    返回 (summary_text, elapsed)
    """
    print(f"[DEBUG][summary_handler] -- 粗度总结流程开始 --")

    latest_topic = extract_latest_topic(full_convs)
    print(f"[DEBUG][summary_handler] extract_latest_topic 结果: {latest_topic!r}")

    if latest_topic:
        topic_prefix = f"【本次讨论的确认选题】\n{latest_topic}\n\n"
        convs_for_summary = topic_prefix + full_convs
        print(f"[DEBUG][summary_handler] 已将确认选题注入到粗度总结上下文")
    else:
        convs_for_summary = full_convs
        print(f"[DEBUG][summary_handler] 未检测到确认选题，使用原始对话记录")

    t0 = time.time()
    summary, _ = ctx.agent.summarize_convs(query=ctx.query, convs=convs_for_summary)
    elapsed = time.time() - t0
    print(f"[DEBUG][summary_handler] 粗度总结生成完成 (耗时: {elapsed:.2f}s)")
    return summary, elapsed


# ── 细度总结 ──────────────────────────────────────────────────────────────────

def _handle_focused_summary(ctx: SummaryContext, full_convs: str, topic: str):
    """
    细度总结：
      - 从 full_convs 中筛选与 topic 相关的行（所有人含 bot 的发言均参与筛选）
      - 若相关内容不足 2 行，直接向用户发送提示，返回 (None, 0)
      - 否则调用 agent.summarize_focused()，使用 summary_focused.txt prompt
    返回 (summary_text, elapsed) 或 (None, 0)
    """
    from utils import send_answer

    print(f"[DEBUG][summary_handler] -- 细度总结流程开始，话题={topic!r} --")

    focused_lines = _filter_convs_by_topic(full_convs, topic)
    print(f"[DEBUG][summary_handler] 话题筛选结果: "
          f"原始 {len(full_convs.splitlines())} 行 -> 相关 {len(focused_lines)} 行")

    if len(focused_lines) < 2:
        print(f"[DEBUG][summary_handler] ⚠ 相关内容不足 2 行，发送提示并返回")
        send_answer(
            client=ctx.client,
            channel_id=ctx.channel_id,
            user_id=ctx.user_id,
            answer=(
                f"📋 在对话记录中，关于「{topic}」的讨论内容较少，暂时无法生成专项总结。\n\n"
                f"如果需要整体总结，可以直接说「帮我总结今天的讨论」😊"
            ),
        )
        return None, 0

    focused_convs = "\n".join(focused_lines)
    if _summary_debug_verbose():
        print(f"[DEBUG][summary_handler] 细度总结输入预览:\n{focused_convs[:300]}")

    t0 = time.time()
    summary, _ = ctx.agent.summarize_focused(
        query=ctx.query,
        topic=topic,
        focused_convs=focused_convs,
    )
    elapsed = time.time() - t0
    print(f"[DEBUG][summary_handler] 细度总结生成完成 (耗时: {elapsed:.2f}s)")
    return summary, elapsed


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _load_full_convs(ctx: SummaryContext) -> str:
    """
    从数据库加载该频道的完整对话记录（所有历史，不受 Slack API 窗口限制）。
    过滤掉 Bot 自身的总结/分工等回复，只保留用户发言和有实质内容的 Bot 回复。
    所有人（包括 bot）的发言均保留，供细度筛选使用。
    """
    try:
        rows = ctx.memory.load_all_utterances(table_name=ctx.channel_name)
        print(f"[DEBUG][summary_handler] DB {ctx.channel_name} 共读取 {len(rows)} 条记录")
        if _summary_debug_verbose():
            print(f"[DEBUG][summary_handler] DB 中所有 speaker: {set(r.get('speaker','') for r in rows)}")
            print(f"[DEBUG][summary_handler] DB 完整历史 conv 内容:")
            for idx, row in enumerate(rows, 1):
                print(f"[DEBUG][summary_handler]   [{idx:03d}] speaker={row.get('speaker','')!r:20s} utterance={row.get('utterance','')[:80]!r}")
    except Exception as e:
        print(f"[DEBUG][summary_handler] load_all_utterances 失败，回退到 convs: {e}")
        return ctx.convs

    lines = []
    for row in rows:
        speaker = row.get("speaker", "")
        utterance = row.get("utterance", "").strip()
        if not utterance:
            continue
        # 跳过 Bot 发出的总结本身（避免总结嵌套总结）
        if speaker == "CoSearchAgent" and utterance.startswith("\U0001f5d3"):
            continue
        lines.append(f"{speaker}: {utterance}")

    result = "\n".join(lines) if lines else ctx.convs
    print(f"[DEBUG][summary_handler] _load_full_convs 完成，返回 {len(result.splitlines())} 行")
    if _summary_debug_verbose():
        print(f"[DEBUG][summary_handler] full_convs 前30行原始内容:")
        for idx, line in enumerate(result.splitlines()[:30], 1):
            print(f"[DEBUG][summary_handler]   [{idx:02d}] {line}")
    return result


def _filter_convs_by_topic(full_convs: str, topic: str) -> list:
    """
    从完整对话中筛选与 topic 相关的行（大小写不敏感，所有人含 bot 均参与匹配）。

    上下文窗口策略：
      - 向前 2 行：保留提问背景
      - 向后 5 行：覆盖 bot 多行长回复（bot 回复往往在关键词命中行之后连续展开）
    多个命中区间取并集，不会重复收录。
    """
    if not topic:
        return full_convs.splitlines()

    # 将 topic 拆分为关键词（按空格/标点切分，过滤单字）
    keywords = [kw for kw in re.split(r'[\s，,、/]+', topic) if len(kw) >= 2]
    if not keywords:
        keywords = [topic]

    print(f"[DEBUG][summary_handler] 细度筛选关键词: {keywords}")

    all_lines = full_convs.splitlines()
    hit_indices = set()

    for i, line in enumerate(all_lines):
        line_lower = line.lower()
        if any(kw.lower() in line_lower for kw in keywords):
            # 前 2 行（问题背景）+ 命中行 + 后 5 行（覆盖 bot 长回复）
            for offset in range(-2, 6):
                idx = i + offset
                if 0 <= idx < len(all_lines):
                    hit_indices.add(idx)

    focused = [all_lines[i] for i in sorted(hit_indices)]
    print(f"[DEBUG][summary_handler] 关键词命中行数（含上下文窗口）: {len(focused)}")
    if _summary_debug_verbose():
        print(f"[DEBUG][summary_handler] 筛选出的完整内容如下:")
        for idx, line in enumerate(focused, 1):
            print(f"[DEBUG][summary_handler]   [{idx:02d}] {line}")
    return focused