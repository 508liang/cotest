import json
import re

from trigger_rules import clean_query_text, is_low_information_judgment_query


def format_recent_summaries(summaries: list[dict], limit: int = 3) -> str:
    if not summaries:
        return "暂无频道摘要"
    chunks = []
    for i, s in enumerate(summaries[-limit:], start=1):
        topics = ", ".join(s.get("topics") or []) or "无"
        tags = ", ".join(s.get("annotation_tags") or []) or "无"
        chunks.append(
            f"摘要{i}: {s.get('summary_text', '')}\n"
            f"主题: {topics}\n"
            f"标注: {tags}"
        )
    return "\n\n".join(chunks)


def _extract_recent_bot_answers(convs_text: str, max_items: int = 4) -> str:
    lines = [line.strip() for line in convs_text.split("\n") if line.strip()]
    hits = []
    for line in lines:
        low = line.lower()
        if low.startswith("bot:") or "cosearchagent:" in low or low.startswith("cosearchagent:"):
            hits.append(line)
    if not hits:
        return "暂无"
    return "\n".join(hits[-max_items:])


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _plan_once(agent, current_query: str, convs_text: str, summary_text: str) -> dict:
    prompt = (
        "你是多轮协作讨论的判断路由器。请基于输入判断：是否应该发起文献检索。\n"
        "目标：避免重复已回答主题、避免模糊检索词、优先给出可检索的明确争议问题。\n\n"
        f"当前触发句:\n{current_query}\n\n"
        f"近期对话:\n{convs_text}\n\n"
        f"频道摘要(历史记忆):\n{summary_text}\n\n"
        f"近期Bot回答(用于去重):\n{_extract_recent_bot_answers(convs_text)}\n\n"
        "请输出 JSON，字段如下：\n"
        "- action: retrieve | skip | need_more_context\n"
        "- search_query: 字符串，action=retrieve 时必须给出\n"
        "- reason: 简短原因\n"
        "- assistant_reply: action=skip 时给用户的简短回复（可为空）\n"
        "约束：\n"
        "1) 若当前信息不足以形成明确检索词，输出 need_more_context。\n"
        "2) 若问题与近期 Bot 已回答主题重复且无新约束，输出 skip。\n"
        "3) 若可检索，search_query 必须是完整具体问题，禁止空泛短句。\n"
        "只输出 JSON，不要附加解释。"
    )

    raw = agent.generate_openai_response(prompt)
    data = _parse_json(raw)
    if not isinstance(data, dict):
        data = {}

    action = str(data.get("action", "")).strip().lower()
    if action not in {"retrieve", "skip", "need_more_context"}:
        action = "skip"

    search_query = clean_query_text(str(data.get("search_query", "")))
    reason = str(data.get("reason", "")).strip()
    assistant_reply = str(data.get("assistant_reply", "")).strip()

    if action == "retrieve" and (not search_query or is_low_information_judgment_query(search_query)):
        action = "need_more_context"

    return {
        "action": action,
        "search_query": search_query,
        "reason": reason,
        "assistant_reply": assistant_reply,
    }


def resolve_judgment_plan(
    agent,
    current_query: str,
    recent_convs_text: str,
    recent_summaries: list[dict] | None = None,
    expanded_convs_text: str = "",
) -> dict:
    summaries_text = format_recent_summaries(recent_summaries or [], limit=3)
    first = _plan_once(agent, current_query, recent_convs_text, summaries_text)

    if first["action"] != "need_more_context":
        return first

    if not expanded_convs_text or expanded_convs_text.strip() == recent_convs_text.strip():
        return {
            "action": "skip",
            "search_query": "",
            "reason": first.get("reason") or "信息不足，且无法扩展上下文",
            "assistant_reply": first.get("assistant_reply") or "当前信息还不足以形成有效检索问题，请补充要比较的方案、约束和目标。",
        }

    second = _plan_once(agent, current_query, expanded_convs_text, summaries_text)
    if second["action"] == "need_more_context":
        return {
            "action": "skip",
            "search_query": "",
            "reason": second.get("reason") or "扩展上下文后仍不足",
            "assistant_reply": second.get("assistant_reply") or "我还缺少关键约束信息，先别检索。请补充候选方案、数据规模、部署限制。",
        }
    return second
