import time
import ast
import re
import threading
import os
import json
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from handlers.summary_handler import handle_summary_intent, SummaryContext
from agents.cosearch_agent import CoSearchAgent
from agents.search_engine import SearchEngine
from memory.cosearch_agent_memory import Memory
from memory.rag_results_memory import SearchMemory
from memory.click_memory import ClickMemory
from utils import (
    get_conversation_history,
    get_user_only_conversation_history,
    replace_utterance_ids,
    get_user_info,
    get_channel_info,
    send_rag_answer,
    send_link_only_rag_answer,
    send_clarify_question,
    update_rag_answer,
    send_answer,
    resolve_user_name,
    is_new_channel,
    send_status_message,
    delete_status_message,
    SERPAPI_KEY,
    OPENAI_KEY,
    SQL_PASSWORD,
    register_channel_display_name,
)
from memory.user_profile_memory import UserProfileMemory
from handlers.topic_handler import handle_topic_intent, _execute_topic, TopicContext
from handlers.division_handler import handle_division_intent, _execute_division, DivisionContext
from handlers.profile_confirm import (
    handle_profile_confirm, handle_profile_edit, handle_profile_modal_submit
)
from handlers.profile_utils import (
    draft_profiles_from_convs,
    notify_profile_update_if_changed,
)
from handlers.profile_watcher import watch_profile_in_background
from memory.pending_intent_memory import PendingIntentMemory
from judgment_planner import resolve_judgment_plan
from config import settings, validate_required_settings
from trigger_rules import (
    clean_query_text,
    has_confusion_cue,
    extract_candidate_terms,
    is_decision_like_message,
    is_conflict_like_message,
)

# Slack API tokens
validate_required_settings()

SLACK_BOT_TOKEN = settings.slack_bot_token
SLACK_APP_TOKEN = settings.slack_app_token
BOT_ID = settings.slack_bot_id

app = App(token=SLACK_BOT_TOKEN)

search_engine = SearchEngine(api_key=SERPAPI_KEY)
agent = CoSearchAgent(
    search_engine=search_engine,
    api_key=OPENAI_KEY,
    model_name=settings.llm_model_name,
    fallback_model_name=settings.llm_fallback_model_name,
    prompt_dir="prompts/ch",
)

memory = Memory(sql_password=SQL_PASSWORD)
search_memory = SearchMemory(sql_password=SQL_PASSWORD)
click_memory = ClickMemory(sql_password=SQL_PASSWORD)
profile_memory = UserProfileMemory(sql_password=SQL_PASSWORD)
profile_memory.create_table_if_not_exists()
pending_memory = PendingIntentMemory(sql_password=SQL_PASSWORD)
pending_memory.create_table_if_not_exists()

channel_id2names = get_channel_info(table_name="channel_info")
user_id2names = get_user_info(table_name="user_info")
# channel_id2names = {} 
# user_id2names = {}
user_id2names[BOT_ID] = "CoSearchAgent"


# 全局对象字典，供画像确认后恢复意图执行时使用
_GLOBAL_OBJECTS = {
    "agent": agent,
    "search_engine": search_engine,
    "memory": memory,
    "search_memory": search_memory,
}

# 本次运行已见过的频道（内存缓存，重启后由 DB 补充）
_seen_channels: set = set()

_LAST_PROACTIVE_TRIGGER: dict = {}
_PERIODIC_ANALYSIS_STATE: dict = {}
_CHANNEL_INFO_SYNCED: set = set()
_CHANNEL_INTRO_SENT: set = set()
_AUTO_PROMPT_STATE: dict = {}
_AUTO_ROUND_COUNTER: dict = {}
_AUTO_FOLLOWUP_STATE: dict = {}

# 用户已解释话题字典：按频道隔离，防止不同频道的历史解释互相污染。
# key = f"{channel_id}:{user_id}", value = [query1, query2, ...]
_user_explained_topics: dict[str, list[str]] = {}


def _send_channel_intro_once(client, channel_id: str, inviter: str | None = None, reason: str = "") -> bool:
    """在当前进程内按频道只发送一次自我介绍，避免重复打扰。"""
    if channel_id in _CHANNEL_INTRO_SENT:
        return False

    prefix = f"<@{inviter}> " if inviter else ""
    try:
        client.chat_postMessage(
            channel=channel_id,
            text=(
                f"{prefix}大家好，我已加入这个频道。\n"
                "我会持续监听本频道内的讨论，并按频道分别维护上下文与用户画像。\n"
                "直接 @我 并描述需求即可，例如：帮我推荐选题、规划分工、总结讨论。"
            ),
        )
        _CHANNEL_INTRO_SENT.add(channel_id)
        print(f"[DEBUG][intro] 已发送频道自我介绍: channel={channel_id!r} reason={reason!r}")
        return True
    except Exception as e:
        print(f"[DEBUG][intro] 发送频道自我介绍失败: channel={channel_id!r} reason={reason!r} err={e}")
        return False


def _looks_like_profile_intro(text: str) -> bool:
    """识别用户在普通发言中透露画像信息的常见表达。"""
    content = (text or "").strip()
    if not content:
        return False
    cues = (
        "我是", "我学", "我专业", "我的专业", "研究方向", "研究兴趣", "主修", "博士", "硕士", "本科",
        "方向是", "专业是", "从事", "擅长", "感兴趣", "关注", "想研究", "想做",
    )
    return any(cue in content for cue in cues)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        print(f"[DEBUG][proactive] 环境变量 {name} 非法: {raw!r}，回退默认值 {default}")
        return default


TERM_COOLDOWN_SECONDS = _int_env("TERM_COOLDOWN_SECONDS", 180)
JUDGMENT_COOLDOWN_SECONDS = _int_env("JUDGMENT_COOLDOWN_SECONDS", 180)
PERIODIC_ANALYSIS_MESSAGE_WINDOW = _int_env("PERIODIC_ANALYSIS_MESSAGE_WINDOW", 10)
PERIODIC_ANALYSIS_SECONDS_WINDOW = _int_env("PERIODIC_ANALYSIS_SECONDS_WINDOW", 30)
INTENT_CONTEXT_MESSAGE_LIMIT = _int_env("INTENT_CONTEXT_MESSAGE_LIMIT", 10)
AUTO_RECOGNIZE_EVERY_ROUNDS = _int_env("AUTO_RECOGNIZE_EVERY_ROUNDS", 3)
AUTO_CONFIRM_EXPIRE_SECONDS = _int_env("AUTO_CONFIRM_EXPIRE_SECONDS", 180)
FOLLOWUP_ROUNDS = _int_env("FOLLOWUP_ROUNDS", 3)
FOLLOWUP_POLL_SECONDS = _int_env("FOLLOWUP_POLL_SECONDS", 2)
FOLLOWUP_MAX_CYCLES = _int_env("FOLLOWUP_MAX_CYCLES", 4)
FOLLOWUP_MAX_SECONDS = _int_env("FOLLOWUP_MAX_SECONDS", 300)

# 全局处理锁：防止同一用户在同一频道并发触发多次处理
# key: (channel_id, user_id)，value: threading.Lock()
_processing_locks: dict = {}
_locks_mutex = threading.Lock()

def _get_user_lock(channel_id: str, user_id: str) -> threading.Lock:
    key = (channel_id, user_id)
    with _locks_mutex:
        if key not in _processing_locks:
            _processing_locks[key] = threading.Lock()
        return _processing_locks[key]

def _get_active_user_ids_in_channel(client, channel_id: str, bot_id: str,
                                     user_id2names: dict, memory: Memory,
                                     channel_name: str) -> list[str]:
    """
    获取当前频道中参与过对话的真实用户 ID 列表（排除 Bot）。
    改用从数据库读取发言记录的方式，避免需要额外的 Slack API 权限。
    """
    try:
        # 从数据库读取该频道的所有发言者
        all_rows = memory.load_all_utterances(table_name=channel_name)
        speakers = set()
        for row in all_rows:
            speaker = row.get("speaker", "")
            if speaker and speaker != "CoSearchAgent":
                speakers.add(speaker)
        
        # 反查 user_id：user_id2names 是 {uid: uname}
        active = []
        for uid, uname in user_id2names.items():
            if uid != bot_id and uid != "bot_id" and uname in speakers:
                active.append(uid)
        
        print(f"[DEBUG][app] 频道 {channel_name} 参与用户（从DB）: {active} speakers={speakers}")
        return active
    except Exception as e:
        print(f"[DEBUG][app] ⚠ 获取频道参与用户失败: {e}")
        return []


# 意图 → 中文标签映射（用于状态消息）
INTENT_LABEL_MAP = {
    "【选题】": "📚 选题建议",
    "【分工】": "📋 研究分工",
    "【总结】": "🗓️ 对话总结",
    "【专业解释】": "🧠 专业解释",
    "【知识解答】": "📖 知识解答",
    "【判断】": "⚖️ 判断分析",
    "【其他】": "💬 闲聊问答",
}


def _format_profile_text_for_explain(channel_id: str, user_id: str, user_name: str) -> str:
    profile = profile_memory.load(user_id, channel_id)
    if not profile:
        return f"用户：{user_name}\n专业：未知\n研究兴趣：暂无\n方法偏好：暂无"

    interests = "、".join(profile.get("research_interests") or []) or "暂无"
    methods = "、".join(profile.get("methodology") or []) or "暂无"
    major = profile.get("major") or "未知"
    return (
        f"用户：{profile.get('user_name') or user_name}\n"
        f"专业：{major}\n"
        f"研究兴趣：{interests}\n"
        f"方法偏好：{methods}"
    )


def _now_ts() -> float:
    return time.time()


def _normalize_term(term: str) -> str:
    clean = re.sub(r"[^\w\u4e00-\u9fff\-]+", "", (term or "").strip().lower())
    return clean


def _parse_yes_no(text: str) -> str:
    content = (text or "").strip().lower()
    if not content:
        return "unknown"

    yes_cues = ("需要", "要", "可以", "好", "行", "是", "yes", "y")
    no_cues = ("不需要", "不用", "不要", "不用了", "不", "否", "no", "n", "算了")

    if any(cue in content for cue in no_cues):
        return "no"
    if any(cue in content for cue in yes_cues):
        return "yes"
    return "unknown"


def _has_understood_cue(text: str) -> bool:
    understood_cues = ("懂了", "明白了", "清楚了", "知道了", "ok", "收到", "了解了")
    content = (text or "").lower()
    return any(c in content for c in understood_cues)


def _build_auto_key(channel_id: str, user_id: str) -> str:
    return f"{channel_id}:{user_id}"


def _term_already_known(channel_id: str, user_id: str, term: str) -> bool:
    clean_term = _normalize_term(term)
    if not clean_term:
        return True
    known = profile_memory.get_known_terms(user_id=user_id, channel_id=channel_id)
    return clean_term in known


def _mark_term_known(channel_id: str, user_id: str, user_name: str, term: str):
    clean_term = _normalize_term(term)
    if not clean_term:
        return
    added = profile_memory.add_known_term(user_id=user_id, channel_id=channel_id, term=clean_term)
    if added:
        print(f"[DEBUG][known_term] 已记录已知术语 user={user_name!r} term={clean_term!r}")


def _load_new_user_messages(channel_name: str, user_name: str, last_ts: float, max_items: int = 50) -> list[dict]:
    rows = memory.load_all_utterances(table_name=channel_name)
    out = []
    for row in rows:
        speaker = row.get("speaker") or ""
        utterance = (row.get("utterance") or "").strip()
        ts_raw = row.get("timestamp") or "0"
        if speaker != user_name or not utterance:
            continue
        try:
            ts_val = float(ts_raw)
        except Exception:
            continue
        if ts_val <= last_ts:
            continue
        out.append({"timestamp": ts_val, "utterance": utterance})
    out.sort(key=lambda x: x["timestamp"])
    return out[-max_items:]


def _run_special_judgment_choice(
    client,
    channel_id: str,
    user_id: str,
    user_name: str,
    channel_name: str,
):
    try:
        convs = get_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=str(_now_ts()),
            limit=60,
        )
        prompt = (
            "你是争议裁决助手。请基于以下对话给出明确选择，禁止模糊中立。\n"
            "输出格式必须为JSON："
            "{\"choice\":\"...\",\"reason\":\"...\",\"next_step\":\"...\"}\n"
            "要求：\n"
            "1) choice 必须给出明确站位或方案。\n"
            "2) reason 用2句以内，给出核心依据。\n"
            "3) next_step 给出可执行下一步。\n\n"
            f"用户：{user_name}\n"
            f"近期对话：\n{convs}\n"
        )
        raw = agent.generate_openai_response(prompt)
        data = _safe_load_json(raw)
        choice = str(data.get("choice") or "方案A").strip()
        reason = str(data.get("reason") or "基于当前约束与可执行性，优先该方案。").strip()
        next_step = str(data.get("next_step") or "先按该方案执行一个最小可验证版本，再复盘。").strip()
        answer = (
            f"<@{user_id}> 在你们持续争论的点上，我给出明确判断：*{choice}*。\n"
            f"依据：{reason}\n"
            f"建议下一步：{next_step}"
        )
        response = send_answer(client=client, channel_id=channel_id, user_id=user_id, answer=answer)
        memory.write_into_memory(
            table_name=channel_name,
            utterance_info={
                "speaker": "CoSearchAgent",
                "utterance": answer,
                "convs": convs,
                "query": "special_judgment_choice",
                "rewrite_query": "",
                "rewrite_thought": "",
                "clarify": "",
                "clarify_thought": "",
                "clarify_cnt": 0,
                "search_results": "",
                "infer_time": str({"workflow": "special_judgment_choice"}),
                "reply_timestamp": str(_now_ts()),
                "reply_user": user_name,
                "timestamp": response["ts"],
            },
        )
    except Exception as e:
        print(f"[DEBUG][followup_judgment] 特殊判断失败: {e}")


def _start_explain_followup_worker(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    term: str,
    trigger_ts: float,
):
    follow_key = f"explain:{channel_id}:{user_id}:{_normalize_term(term)}"
    _AUTO_FOLLOWUP_STATE[follow_key] = {"active": True}

    def _worker():
        clean_term = _normalize_term(term)
        last_ts = float(trigger_ts)
        rounds_left = FOLLOWUP_ROUNDS
        cycles = 0
        started_at = _now_ts()
        mention_seen = False
        confusion_seen = False
        understood_seen = False
        print(f"[DEBUG][followup_explain] 启动 follow_key={follow_key!r}")

        while _AUTO_FOLLOWUP_STATE.get(follow_key, {}).get("active"):
            if _now_ts() - started_at > FOLLOWUP_MAX_SECONDS:
                break
            time.sleep(max(1, FOLLOWUP_POLL_SECONDS))

            try:
                new_msgs = _load_new_user_messages(channel_name, user_name, last_ts)
            except Exception as e:
                print(f"[DEBUG][followup_explain] 读取消息失败: {e}")
                continue

            if not new_msgs:
                continue

            for msg in new_msgs:
                last_ts = msg["timestamp"]
                text = msg["utterance"]
                rounds_left -= 1
                if clean_term and clean_term in text.lower():
                    mention_seen = True
                if has_confusion_cue(text):
                    confusion_seen = True
                if _has_understood_cue(text):
                    understood_seen = True

                if rounds_left > 0:
                    continue

                if mention_seen and confusion_seen and cycles < FOLLOWUP_MAX_CYCLES:
                    convs = get_conversation_history(
                        client=client,
                        channel_id=channel_id,
                        bot_id=BOT_ID,
                        user_id2names=user_id2names,
                        ts=str(last_ts),
                        limit=30,
                    )
                    query = f"请再次解释术语：{term}，重点化解用户当前困惑。"
                    _run_retrieval_intent(
                        client=client,
                        channel_id=channel_id,
                        channel_name=channel_name,
                        user_id=user_id,
                        user_name=user_name,
                        ts=str(last_ts),
                        query=query,
                        search_query=query,
                        convs=convs,
                        intent_label="【专业解释】",
                        intent_time=0.0,
                    )
                    cycles += 1
                    rounds_left = FOLLOWUP_ROUNDS
                    mention_seen = False
                    confusion_seen = False
                    understood_seen = False
                    continue

                if (not mention_seen) or understood_seen:
                    _mark_term_known(channel_id=channel_id, user_id=user_id, user_name=user_name, term=term)
                    _AUTO_FOLLOWUP_STATE[follow_key] = {"active": False}
                    return

                cycles += 1
                rounds_left = FOLLOWUP_ROUNDS
                mention_seen = False
                confusion_seen = False
                understood_seen = False
                if cycles >= FOLLOWUP_MAX_CYCLES:
                    _AUTO_FOLLOWUP_STATE[follow_key] = {"active": False}
                    return

        _AUTO_FOLLOWUP_STATE[follow_key] = {"active": False}

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _start_judgment_followup_worker(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    trigger_ts: float,
):
    follow_key = f"judgment:{channel_id}:{user_id}:{int(trigger_ts)}"
    _AUTO_FOLLOWUP_STATE[follow_key] = {"active": True}

    def _worker():
        last_ts = float(trigger_ts)
        rounds_left = FOLLOWUP_ROUNDS
        started_at = _now_ts()
        conflict_seen = False
        print(f"[DEBUG][followup_judgment] 启动 follow_key={follow_key!r}")

        while _AUTO_FOLLOWUP_STATE.get(follow_key, {}).get("active"):
            if _now_ts() - started_at > FOLLOWUP_MAX_SECONDS:
                break
            time.sleep(max(1, FOLLOWUP_POLL_SECONDS))

            try:
                new_msgs = _load_new_user_messages(channel_name, user_name, last_ts)
            except Exception as e:
                print(f"[DEBUG][followup_judgment] 读取消息失败: {e}")
                continue

            if not new_msgs:
                continue

            for msg in new_msgs:
                last_ts = msg["timestamp"]
                text = msg["utterance"]
                rounds_left -= 1
                if is_conflict_like_message(text) or is_decision_like_message(text):
                    conflict_seen = True
                if rounds_left > 0:
                    continue

                if conflict_seen:
                    _run_special_judgment_choice(
                        client=client,
                        channel_id=channel_id,
                        user_id=user_id,
                        user_name=user_name,
                        channel_name=channel_name,
                    )
                _AUTO_FOLLOWUP_STATE[follow_key] = {"active": False}
                return

        _AUTO_FOLLOWUP_STATE[follow_key] = {"active": False}

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _auto_triage(channel_id: str, user_id: str, user_name: str, message_text: str, convs: str) -> dict:
    profile = profile_memory.load(user_id=user_id, channel_id=channel_id) or {}
    major = str(profile.get("major") or "未知")
    interests = "、".join(profile.get("research_interests") or []) or "暂无"
    known_terms = profile_memory.get_known_terms(user_id=user_id, channel_id=channel_id)

    prompt = (
        "你是多轮对话自动识别器。目标：先识别是否有需要专业解释的术语，再识别是否存在争论需要判断。\n"
        "仅输出 JSON："
        "{\"kind\":\"explain|judgment|none\",\"term\":\"...\",\"query\":\"...\",\"reason\":\"...\"}\n"
        "规则：\n"
        "1) explain 优先级高于 judgment。\n"
        "2) explain 仅在出现专业术语且该术语与用户专业可能存在认知差时触发。\n"
        "3) term 必须是术语本体（例如 rag）。\n"
        "4) judgment 仅在出现明确争论/选择冲突时触发。\n"
        "5) 无需介入时输出 kind=none。\n\n"
        f"用户：{user_name}\n"
        f"专业：{major}\n"
        f"研究兴趣：{interests}\n"
        f"已知术语（不可重复触发）：{known_terms}\n"
        f"当前消息：{message_text}\n"
        f"最近对话：\n{convs}\n"
    )

    try:
        raw = agent.generate_openai_response(prompt)
        data = _safe_load_json(raw)
    except Exception as e:
        print(f"[DEBUG][auto_triage] LLM失败，降级规则: {e}")
        data = {}

    kind = str(data.get("kind") or "none").strip().lower()
    if kind not in ("explain", "judgment", "none"):
        kind = "none"

    term = _normalize_term(str(data.get("term") or ""))
    query = clean_query_text(str(data.get("query") or ""))
    reason = str(data.get("reason") or "").strip()

    if kind == "explain" and (not term):
        terms = extract_candidate_terms(message_text)
        term = _normalize_term(terms[0] if terms else "")

    if kind == "explain" and _term_already_known(channel_id=channel_id, user_id=user_id, term=term):
        return {"kind": "none", "term": "", "query": "", "reason": "term_known"}

    if kind == "judgment" and not query:
        query = clean_query_text(message_text)

    if kind == "explain" and not query:
        query = clean_query_text(message_text)

    return {"kind": kind, "term": term, "query": query, "reason": reason}


def _should_run_auto_recognition(channel_id: str, user_id: str) -> bool:
    key = _build_auto_key(channel_id, user_id)
    cnt = int(_AUTO_ROUND_COUNTER.get(key, 0)) + 1
    _AUTO_ROUND_COUNTER[key] = cnt
    if cnt < AUTO_RECOGNIZE_EVERY_ROUNDS:
        return False
    _AUTO_ROUND_COUNTER[key] = 0
    return True


def _set_auto_prompt(channel_id: str, user_id: str, data: dict):
    key = _build_auto_key(channel_id, user_id)
    _AUTO_PROMPT_STATE[key] = data


def _clear_auto_prompt(channel_id: str, user_id: str):
    key = _build_auto_key(channel_id, user_id)
    _AUTO_PROMPT_STATE.pop(key, None)


def _get_auto_prompt(channel_id: str, user_id: str) -> dict | None:
    key = _build_auto_key(channel_id, user_id)
    data = _AUTO_PROMPT_STATE.get(key)
    if not data:
        return None
    created_at = float(data.get("created_at") or 0.0)
    if _now_ts() - created_at > AUTO_CONFIRM_EXPIRE_SECONDS:
        _AUTO_PROMPT_STATE.pop(key, None)
        return None
    return data


def _send_auto_confirm_prompt(client, channel_id: str, user_id: str, kind: str, term: str):
    if kind == "explain":
        text = f"检测到你们在讨论术语“{term or '该术语'}”，需要我现在给一个专业解释吗？回复“需要”或“不需要”。"
    else:
        text = "检测到你们可能在争论同一问题，需要我发起一次判断分析吗？回复“需要”或“不需要”。"
    send_status_message(client=client, channel_id=channel_id, user_id=user_id, text=text)


def _try_handle_auto_prompt_reply(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    ts: str,
    text: str,
) -> bool:
    pending = _get_auto_prompt(channel_id, user_id)
    if not pending:
        return False

    decision = _parse_yes_no(text)
    if decision == "unknown":
        return False

    kind = pending.get("kind")
    term = str(pending.get("term") or "")
    query = clean_query_text(str(pending.get("query") or text))
    convs = get_conversation_history(
        client=client,
        channel_id=channel_id,
        bot_id=BOT_ID,
        user_id2names=user_id2names,
        ts=ts,
        limit=30,
    )

    if decision == "no":
        if kind == "explain":
            _mark_term_known(channel_id=channel_id, user_id=user_id, user_name=user_name, term=term)
        send_status_message(
            client=client,
            channel_id=channel_id,
            user_id=user_id,
            text="收到，这次我先不介入。",
        )
        _clear_auto_prompt(channel_id, user_id)
        return True

    if kind == "explain":
        _run_retrieval_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            search_query=query,
            convs=convs,
            intent_label="【专业解释】",
            intent_time=0.0,
        )
        _start_explain_followup_worker(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            term=term or query,
            trigger_ts=float(ts),
        )
    elif kind == "judgment":
        plan = resolve_judgment_plan(
            agent=agent,
            current_query=query,
            recent_convs_text=convs,
            recent_summaries=[],
            expanded_convs_text=convs,
        )
        use_query = plan.get("search_query") if plan.get("action") == "retrieve" else query
        _run_retrieval_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            search_query=use_query,
            convs=convs,
            intent_label="【判断】",
            intent_time=0.0,
        )
        _start_judgment_followup_worker(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            trigger_ts=float(ts),
        )

    _clear_auto_prompt(channel_id, user_id)
    return True


def _maybe_run_auto_recognition(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    ts: str,
    text: str,
):
    if _get_auto_prompt(channel_id, user_id):
        return
    if not _should_run_auto_recognition(channel_id, user_id):
        return

    convs = get_conversation_history(
        client=client,
        channel_id=channel_id,
        bot_id=BOT_ID,
        user_id2names=user_id2names,
        ts=ts,
        limit=40,
    )
    triage = _auto_triage(
        channel_id=channel_id,
        user_id=user_id,
        user_name=user_name,
        message_text=text,
        convs=convs,
    )
    kind = triage.get("kind")
    if kind not in ("explain", "judgment"):
        return

    if kind == "explain" and _term_already_known(channel_id, user_id, triage.get("term") or ""):
        return

    _set_auto_prompt(
        channel_id,
        user_id,
        {
            "kind": kind,
            "term": triage.get("term") or "",
            "query": triage.get("query") or clean_query_text(text),
            "reason": triage.get("reason") or "",
            "created_at": _now_ts(),
        },
    )
    _send_auto_confirm_prompt(
        client=client,
        channel_id=channel_id,
        user_id=user_id,
        kind=kind,
        term=triage.get("term") or "",
    )


def _parse_rewrite_output(raw_output: str, fallback_query: str) -> tuple[str, str]:
    thought = ""
    rewritten_query = ""

    for line in (raw_output or "").splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        if clean_line.startswith("重写思路：") or clean_line.startswith("判断思路："):
            thought = clean_line.split("：", 1)[-1].strip()
        elif clean_line.startswith("重写查询："):
            rewritten_query = clean_line.split("：", 1)[-1].strip()

    rewritten_query = clean_query_text(rewritten_query or fallback_query)
    return thought, rewritten_query


def _build_rewrite_context(channel_id: str, user_id: str, user_name: str, intent_context: str) -> str:
    """
    构建用于 rewrite_query 的上下文。
    对已解释过话题的用户，从对话上下文中过滤掉 CoSearchAgent 针对那些话题的
    回答块，防止历史解释内容影响新查询的改写（例如：询问"法律"时被"人工智能"
    的解释上下文带偏）。
    逻辑：逐行扫描，若某行是用户发言且内容与已解释话题匹配，则跳过紧随其后的
    那条 CoSearchAgent 回答行。
    """
    explained_key = f"{channel_id}:{user_id}"
    explained = _user_explained_topics.get(explained_key, [])
    if not explained:
        return intent_context

    lines = intent_context.splitlines()
    filtered: list[str] = []
    skip_next_agent = False
    user_prefix = user_name + ":"

    for line in lines:
        if skip_next_agent:
            skip_next_agent = False
            if line.startswith("CoSearchAgent:"):
                print(f"[DEBUG][rewrite_ctx] 过滤已解释话题的bot回答: {line[:60]!r}…")
                continue  # 跳过这条与已解释话题对应的 bot 回答
        # 检测当前行是否是用户询问了某个已解释话题
        if line.startswith(user_prefix):
            content = re.sub(r"@\S+\s*", "", line[len(user_prefix):]).strip()
            for topic in explained:
                clean_topic = re.sub(r"@\S+\s*", "", topic).strip()
                if clean_topic and (clean_topic in content or content in clean_topic):
                    skip_next_agent = True
                    break
        filtered.append(line)

    return "\n".join(filtered)


def _resolve_contextual_search_query(user_name: str, query: str, convs: str) -> tuple[str, float, str]:
    try:
        rewrite_output, rewrite_time = agent.rewrite_query(
            user=user_name,
            query=query,
            convs=convs,
        )
        rewrite_thought, rewrite_query = _parse_rewrite_output(rewrite_output, query)
        return rewrite_query, rewrite_time, rewrite_thought
    except Exception as e:
        print(f"[DEBUG][rewrite_query] 检索词确认失败，回退原始query: {e}")
        return clean_query_text(query), 0.0, ""


def _run_retrieval_intent(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    ts: str,
    query: str,
    search_query: str,
    convs: str,
    intent_label: str,
    intent_time: float,
    rewrite_time: float = 0.0,
    rewrite_thought: str = "",
):
    workflow = {
        "【知识解答】": "knowledge_answer",
        "【专业解释】": "professional_explain",
        "【判断】": "judgment_analysis",
    }.get(intent_label, "knowledge_answer")

    user_profile = ""
    audience_model = ""
    escalation_context = ""
    if intent_label == "【专业解释】":
        user_profile = _format_profile_text_for_explain(channel_id=channel_id, user_id=user_id, user_name=user_name)
        audience_model = (
            "请使用该用户已知领域做类比；尽量先给直觉解释，再给一句简化定义；"
            "控制术语密度，优先保证可理解性。"
        )
    elif intent_label == "【判断】":
        escalation_context = "当前请求来自统一 app 主流程，请输出可比较的优缺点；若争议明显可附执行建议。"

    answer, search_results, source, workflow_time = agent.run_retrieval_workflow(
        workflow_type=workflow,
        query=search_query,
        serpapi_key=SERPAPI_KEY,
        user_profile=user_profile,
        audience_model=audience_model,
        escalation_context=escalation_context,
    )

    if intent_label == "【专业解释】":
        # 专业解释只展示正文，不附参考链接
        response_ts = send_link_only_rag_answer(
            client=client,
            channel_id=channel_id,
            user_id=user_id,
            answer=answer,
            references=[],
        )["ts"]
    else:
        response_ts = send_rag_answer(
            client=client,
            channel_id=channel_id,
            query=search_query,
            user_id=user_id,
            answer=answer,
            references=search_results,
        )["ts"]

    memory.write_into_memory(
        table_name=channel_name,
        utterance_info={
            "speaker": "CoSearchAgent",
            "utterance": answer,
            "convs": convs,
            "query": query,
            "rewrite_query": search_query,
            "rewrite_thought": rewrite_thought,
            "clarify": "",
            "clarify_thought": "",
            "clarify_cnt": 0,
            "search_results": str(search_results),
            "infer_time": str({
                "intent": intent_time,
                "rewrite": rewrite_time,
                "workflow": workflow_time,
                "source": source,
                "workflow_type": workflow,
            }),
            "reply_timestamp": ts,
            "reply_user": user_name,
            "timestamp": response_ts,
        },
    )

    if intent_label != "【专业解释】":
        search_memory.create_table_if_not_exists(table_name=f"{channel_name}_search")
        search_memory.write_into_memory(
            table_name=f"{channel_name}_search",
            search_info={
                "user_name": user_name,
                "query": search_query,
                "answer": answer,
                "search_results": str(search_results),
                "start": 0,
                "end": 2,
                "click_time": time.time(),
                "timestamp": response_ts,
            },
        )


def _should_trigger_proactive(
    channel_name: str,
    target_user_id: str,
    term_key: str,
    now_ts: float,
    cooldown_seconds: int,
) -> bool:
    key = f"{channel_name}:{target_user_id}:{term_key}"
    last = float(_LAST_PROACTIVE_TRIGGER.get(key, 0.0))
    if now_ts - last < cooldown_seconds:
        return False
    _LAST_PROACTIVE_TRIGGER[key] = now_ts
    return True


def _safe_load_json(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _build_profiles_context(channel_id: str, max_users: int = 6) -> str:
    try:
        profiles = profile_memory.load_all(channel_id)
    except Exception:
        profiles = []

    if not profiles:
        return "暂无用户画像"

    chunks = []
    for p in profiles[:max_users]:
        interests = ", ".join(p.get("research_interests") or []) or "暂无"
        methods = ", ".join(p.get("methodology") or []) or "暂无"
        chunks.append(
            f"user_id={p.get('user_id')} user_name={p.get('user_name')}\n"
            f"major={p.get('major') or '未知'}\n"
            f"interests={interests}\n"
            f"methodology={methods}"
        )
    return "\n\n".join(chunks)


def _periodic_triage(channel_id: str, query: str, convs: str) -> dict:
    prompt = (
        "你是协同讨论巡检器。根据最近对话判断是否需要 Bot 主动介入。\n"
        "仅输出 JSON："
        "{\"intent\":\"【专业解释】|【判断】|【其他】\",\"query\":\"...\",\"reason\":\"...\"}\n"
        "规则：\n"
        "1) 若不需要介入，intent 输出【其他】且 query 置空。\n"
        "2) 若需要【判断】，query 必须是可检索的完整争议问题。\n"
        "3) 若需要【专业解释】，query 必须包含待解释术语。\n\n"
        f"当前消息:\n{query}\n\n"
        f"最近对话:\n{convs}\n\n"
        f"用户画像:\n{_build_profiles_context(channel_id)}\n"
    )

    try:
        raw = agent.generate_openai_response(prompt)
        data = _safe_load_json(raw)
        intent = str(data.get("intent", "【其他】")).strip()
        normalized_intent = "【其他】"
        if intent in ("【专业解释】", "【判断】", "【其他】"):
            normalized_intent = intent
        return {
            "intent": normalized_intent,
            "query": clean_query_text(str(data.get("query", ""))),
            "reason": str(data.get("reason", "")).strip(),
        }
    except Exception as e:
        print(f"[DEBUG][periodic] triage失败，降级跳过: {e}")
        return {"intent": "【其他】", "query": "", "reason": f"triage_error:{type(e).__name__}"}


def _should_run_periodic_analysis(channel_name: str, now_ts: float) -> bool:
    state = _PERIODIC_ANALYSIS_STATE.get(channel_name, {"count": 0, "last_ts": 0.0})
    state["count"] = int(state.get("count", 0)) + 1
    elapsed = now_ts - float(state.get("last_ts", 0.0))
    should = (
        state["count"] >= PERIODIC_ANALYSIS_MESSAGE_WINDOW
        or elapsed >= PERIODIC_ANALYSIS_SECONDS_WINDOW
    )
    if should:
        state["count"] = 0
        state["last_ts"] = now_ts
    _PERIODIC_ANALYSIS_STATE[channel_name] = state
    return should


def _handle_proactive_and_periodic(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    ts: str,
    message_text: str,
):
    try:
        now_ts = time.time()
        convs = get_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=20,
        )
        expanded_convs = get_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=60,
        )

        # 主动触发：术语困惑场景
        if has_confusion_cue(message_text):
            terms = extract_candidate_terms(message_text)
            term_key = terms[0] if terms else "explain"
            if _should_trigger_proactive(
                channel_name=channel_name,
                target_user_id=user_id,
                term_key=term_key,
                now_ts=now_ts,
                cooldown_seconds=TERM_COOLDOWN_SECONDS,
            ):
                print(f"[DEBUG][proactive] 触发专业解释: user={user_name} query={message_text!r}")
                _run_retrieval_intent(
                    client=client,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=user_id,
                    user_name=user_name,
                    ts=ts,
                    query=clean_query_text(message_text),
                    search_query=clean_query_text(message_text),
                    convs=convs,
                    intent_label="【专业解释】",
                    intent_time=0.0,
                )
                return

        # 主动触发：判断/争议场景
        if is_decision_like_message(message_text) or is_conflict_like_message(message_text):
            plan = resolve_judgment_plan(
                agent=agent,
                current_query=message_text,
                recent_convs_text=convs,
                recent_summaries=[],
                expanded_convs_text=expanded_convs,
            )
            if plan.get("action") == "retrieve":
                if _should_trigger_proactive(
                    channel_name=channel_name,
                    target_user_id=user_id,
                    term_key="judgment",
                    now_ts=now_ts,
                    cooldown_seconds=JUDGMENT_COOLDOWN_SECONDS,
                ):
                    print(f"[DEBUG][proactive] 触发判断分析: user={user_name} query={plan.get('search_query')!r}")
                    _run_retrieval_intent(
                        client=client,
                        channel_id=channel_id,
                        channel_name=channel_name,
                        user_id=user_id,
                        user_name=user_name,
                        ts=ts,
                        query=clean_query_text(message_text),
                        search_query=plan.get("search_query") or clean_query_text(message_text),
                        convs=convs,
                        intent_label="【判断】",
                        intent_time=0.0,
                    )
                    return

        # 周期巡检：每N条或每T秒触发一次
        if _should_run_periodic_analysis(channel_name, now_ts):
            triage = _periodic_triage(channel_id=channel_id, query=message_text, convs=expanded_convs)
            if triage.get("intent") == "【专业解释】" and triage.get("query"):
                if _should_trigger_proactive(
                    channel_name=channel_name,
                    target_user_id=user_id,
                    term_key="periodic_explain",
                    now_ts=now_ts,
                    cooldown_seconds=TERM_COOLDOWN_SECONDS,
                ):
                    print(f"[DEBUG][periodic] 触发专业解释: {triage}")
                    _run_retrieval_intent(
                        client=client,
                        channel_id=channel_id,
                        channel_name=channel_name,
                        user_id=user_id,
                        user_name=user_name,
                        ts=ts,
                        query=triage["query"],
                        search_query=triage["query"],
                        convs=convs,
                        intent_label="【专业解释】",
                        intent_time=0.0,
                    )
                    return

            if triage.get("intent") == "【判断】" and triage.get("query"):
                plan = resolve_judgment_plan(
                    agent=agent,
                    current_query=triage["query"],
                    recent_convs_text=convs,
                    recent_summaries=[],
                    expanded_convs_text=expanded_convs,
                )
                if plan.get("action") == "retrieve" and _should_trigger_proactive(
                    channel_name=channel_name,
                    target_user_id=user_id,
                    term_key="periodic_judgment",
                    now_ts=now_ts,
                    cooldown_seconds=JUDGMENT_COOLDOWN_SECONDS,
                ):
                    print(f"[DEBUG][periodic] 触发判断分析: {triage}")
                    _run_retrieval_intent(
                        client=client,
                        channel_id=channel_id,
                        channel_name=channel_name,
                        user_id=user_id,
                        user_name=user_name,
                        ts=ts,
                        query=triage["query"],
                        search_query=plan.get("search_query") or triage["query"],
                        convs=convs,
                        intent_label="【判断】",
                        intent_time=0.0,
                    )
    except Exception as e:
        print(f"[DEBUG][proactive] 主动/巡检流程异常，已降级跳过: {e}")

@app.action("profile_confirm")
def on_profile_confirm(ack, body, client):
    handle_profile_confirm(
        ack=ack, body=body, client=client,
        profile_memory=profile_memory,
        pending_memory=pending_memory,
        user_id2names=user_id2names,
        topic_handler_fn=_execute_topic,
        division_handler_fn=_execute_division,
        TopicContext=TopicContext,
        DivisionContext=DivisionContext,
        global_objects=_GLOBAL_OBJECTS,
    )


@app.action("profile_edit")
def on_profile_edit(ack, body, client):
    handle_profile_edit(ack=ack, body=body, client=client)


@app.view("profile_edit_modal")
def on_profile_modal_submit(ack, body, client):
    handle_profile_modal_submit(
        ack=ack, body=body, client=client,
        profile_memory=profile_memory,
        pending_memory=pending_memory,
        user_id2names=user_id2names,
        topic_handler_fn=_execute_topic,
        division_handler_fn=_execute_division,
        TopicContext=TopicContext,
        DivisionContext=DivisionContext,
        global_objects=_GLOBAL_OBJECTS,
    )


@app.action("click")
def click_link(ack, body):
    channel_name, user_name = body["channel"]["id"], body["user"]["username"]
    event_ts = body["container"]["message_ts"]

    search_info = search_memory.load_search_results_from_timestamp(table_name=f"{channel_name}_search",
                                                                   timestamp=str(event_ts))

    action = body["actions"][0]
    link, ts = action["value"], action["action_ts"]

    click_memory.create_table_if_not_exists(f"{channel_name}_click")
    click_memory.write_into_memory(table_name=f"{channel_name}_click",
                                   click_info={
                                       "user_name": user_name,
                                       "query": search_info["query"],
                                       "link": link,
                                       "timestamp": ts
                                   })

    ack()


@app.action("next")
def return_next_page(ack, body, client):
    channel_id, user_id = body['channel']['id'], body['user']['id']
    channel_name, user_name = channel_id, user_id2names.get(user_id, user_id)
    event_ts = body["container"]["message_ts"]

    search_info = search_memory.load_search_results_from_timestamp(
        table_name=f"{channel_name}_search", timestamp=str(event_ts)
    )
    if not search_info:
        print(f"[DEBUG][next] ⚠ 未找到 search_info，忽略翻页")
        ack()
        return

    search_results = ast.literal_eval(search_info["search_results"])

    start, end = search_info["start"], search_info["end"]
    if start + 2 < len(search_results):
        start = start + 2
        end = end + 2

    ts = update_rag_answer(client=client, query=search_info["query"], channel_id=channel_id, user_id=user_id,
                           answer=search_info["answer"], references=search_results, start=start, end=end,
                           ts=event_ts)["ts"]

    search_memory.write_into_memory(table_name=f"{channel_name}_search", search_info={
        "user_name": user_name,
        "query": search_info["query"],
        "answer": search_info["answer"],
        "search_results": str(search_results),
        "start": start,
        "end": end,
        "click_time": time.time(),
        "timestamp": ts
    })

    ack()


@app.action("previous")
def return_previous_page(ack, body, client):
    channel_id, user_id = body['channel']['id'], body['user']['id']
    channel_name, user_name = channel_id, user_id2names.get(user_id, user_id)
    event_ts = body["container"]["message_ts"]

    search_info = search_memory.load_search_results_from_timestamp(
        table_name=f"{channel_name}_search", timestamp=str(event_ts)
    )
    if not search_info:
        print(f"[DEBUG][previous] ⚠ 未找到 search_info，忽略翻页")
        ack()
        return

    search_results = ast.literal_eval(search_info["search_results"])

    start, end = search_info["start"], search_info["end"]
    if start - 2 >= 0:
        start = start - 2
        end = end - 2

    ts = update_rag_answer(client=client, query=search_info["query"], channel_id=channel_id, user_id=user_id,
                           answer=search_info["answer"], references=search_results, start=start, end=end,
                           ts=event_ts)["ts"]

    search_memory.write_into_memory(table_name=f"{channel_name}_search", search_info={
        "user_name": user_name,
        "query": search_info["query"],
        "answer": search_info["answer"],
        "search_results": str(search_results),
        "start": start,
        "end": end,
        "click_time": time.time(),
        "timestamp": ts
    })

    ack()


@app.event("member_joined_channel")
def handle_member_joined_channel(ack, event, client):
    ack()

    joined_user = event.get("user")
    channel_id = event.get("channel")
    if not channel_id:
        return

    if channel_id not in _CHANNEL_INFO_SYNCED:
        register_channel_display_name(client, channel_id, SQL_PASSWORD)
        _CHANNEL_INFO_SYNCED.add(channel_id)

    # 仅当 Bot 自己入频道时做频道初始化与自我介绍。
    if joined_user != BOT_ID:
        return

    # 预初始化：Bot 刚入频道就落库 seen_channels，后续无需依赖首次@触发。
    try:
        new_channel = is_new_channel(
            channel_id=channel_id,
            channel_name=channel_id,
            seen_channels=_seen_channels,
            sql_password=SQL_PASSWORD,
        )
        print(f"[DEBUG][join] Bot入频道预初始化完成 channel={channel_id!r} new_channel={new_channel}")
    except Exception as e:
        print(f"[DEBUG][join] Bot入频道预初始化失败 channel={channel_id!r}: {e}")

    inviter = event.get("inviter")
    _send_channel_intro_once(client, channel_id, inviter=inviter, reason="member_joined_channel")

@app.event("app_mention")
@app.event("message")
def handle_message_event(ack, event, client):
    ack()

    channel_id, ts = event.get("channel"), event.get("ts")
    user_id = event.get("user")
    subtype = (event.get("subtype") or "").strip()
    if not channel_id or not ts or not user_id:
        return

    # 忽略系统事件类消息，避免“加入频道”等系统文本污染对话/画像。
    if subtype in {
        "channel_join", "group_join", "channel_leave", "group_leave",
        "message_changed", "message_deleted", "bot_message",
        "channel_topic", "channel_purpose",
    }:
        print(f"[DEBUG][handle_message] 跳过系统消息 subtype={subtype!r} ts={ts}")
        return

    channel_name = channel_id

    if user_id == BOT_ID:
        return

    if channel_id not in _CHANNEL_INFO_SYNCED:
        register_channel_display_name(client, channel_id, SQL_PASSWORD)
        _CHANNEL_INFO_SYNCED.add(channel_id)

    user_lock = _get_user_lock(channel_id, user_id)
    if not user_lock.acquire(blocking=False):
        print(f"[DEBUG][handle_message] ⚠ 并发跳过 ts={ts}")
        return

    try:
        user_name = resolve_user_name(client, user_id, user_id2names, SQL_PASSWORD)
        raw_text = (event.get("text") or "").strip("\n").strip()
        user_utterance = replace_utterance_ids(raw_text, id2names=user_id2names)
        print(f"\n{'='*60}")
        print(f"[DEBUG][handle_message] user={user_name!r} text={user_utterance!r}")

        memory.create_table_if_not_exists(table_name=channel_name)
        memory.write_into_memory(
            table_name=channel_name,
            utterance_info={"speaker": user_name, "utterance": user_utterance, "convs": "",
                            "query": "", "rewrite_query": "", "rewrite_thought": "",
                            "clarify": "", "clarify_thought": "", "clarify_cnt": 0,
                            "search_results": "", "infer_time": "", "reply_timestamp": "",
                            "reply_user": "", "timestamp": ts}
        )

        # 支持 Slack 原生 @Bot（<@BOT_ID>）和历史别名前缀。
        mention_token = f"<@{BOT_ID}>"
        legacy_prefix = settings.slack_legacy_mention_prefix
        mentioned_bot = mention_token in raw_text
        query = ""
        if mentioned_bot:
            query = raw_text.replace(mention_token, "").strip()
        elif user_utterance.startswith(legacy_prefix):
            query = user_utterance[len(legacy_prefix):].strip()

        new_channel = is_new_channel(
            channel_id=channel_id, channel_name=channel_name,
            seen_channels=_seen_channels, sql_password=SQL_PASSWORD,
        )

        # ── 新频道：首条消息触发频道级欢迎（无论是否 @Bot）────────────────────
        if new_channel:
            _send_channel_intro_once(client, channel_id, reason="new_channel_first_touch")

        # 兜底：若因事件订阅/BOT_ID 不一致漏掉入频通知，则在首次@Bot时补发一次。
        if mentioned_bot or event.get("type") == "app_mention":
            _send_channel_intro_once(client, channel_id, reason="mention_fallback")

        if _try_handle_auto_prompt_reply(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            text=user_utterance,
        ):
            return

        if not mentioned_bot and not user_utterance.startswith(legacy_prefix) and event.get("type") != "app_mention":
            _maybe_run_auto_recognition(
                client=client,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                ts=ts,
                text=user_utterance,
            )
            if _looks_like_profile_intro(user_utterance):
                send_status_message(
                    client, channel_id, user_id,
                    "已收到你的背景信息，我会更新你的频道画像，并在提炼后发确认卡片。"
                )
            # 被动画像监听（无冷却机制）
            watch_profile_in_background(
                client=client, channel_id=channel_id, channel_name=channel_name,
                user_id=user_id, user_name=user_name,
                agent=agent, memory=memory, profile_memory=profile_memory,
            )
            return

        print(f"[DEBUG][handle_message] query={query!r}")

        if not query:
            # 新频道刚发送过欢迎时，避免紧接着再发一条引导语造成重复打扰。
            if new_channel:
                print(f"[DEBUG][handle_message] query为空，但新频道欢迎已发送，跳过重复引导")
                return

            print(f"[DEBUG][handle_message] query为空，发送引导语并返回")
            send_status_message(
                client, channel_id, user_id,
                "👋 你好！请告诉我你需要什么帮助，例如：\n"
                "• `@我 帮我推荐选题`\n"
                "• `@我 帮我规划分工`\n"
                "• `@我 总结一下我们的讨论`"
            )
            return

        # 先看当前@消息是否显式包含画像线索（如“我是XX专业”）。
        # 这类消息容易被归到【其他】，若仅依赖意图分支会错过画像提炼时机。
        profile_watch_started = False
        if _looks_like_profile_intro(query):
            print("[DEBUG][handle_message] 检测到@消息中的画像线索，提前触发画像监控")
            watch_profile_in_background(
                client=client,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                agent=agent,
                memory=memory,
                profile_memory=profile_memory,
            )
            profile_watch_started = True

        # ── ① 先结合近5轮对话分类意图 ────────────────────────────────────────
        intent_context = get_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=INTENT_CONTEXT_MESSAGE_LIMIT,
        )
        intent_label, intent_time = agent.classify_intent(query=query, convs=intent_context)
        intent_display = INTENT_LABEL_MAP.get(intent_label, f"🔍 {intent_label}")
        print(f"[DEBUG][handle_message] 意图={intent_label!r} 耗时={intent_time:.2f}s")

        # ── ② 立即发"已识别需求类型"状态消息 ────────────────────────────────
        intent_status_ts = send_status_message(
            client, channel_id, user_id,
            f"已识别需求类型：{intent_display}"
        )

        # ── ③ 获取当前频道参与用户 ID（用于限定画像范围）─────────────────────
        active_user_ids = _get_active_user_ids_in_channel(
            client, channel_id, BOT_ID, user_id2names, memory, channel_name
        )
        # 确保触发者在列表中
        if user_id not in active_user_ids:
            active_user_ids.append(user_id)

        # ── ④ 按意图按需拉取 convs ───────────────────────────────────────────
        convs = get_conversation_history(
            client=client, channel_id=channel_id, bot_id=BOT_ID,
            user_id2names=user_id2names, ts=ts,
            limit=5 if intent_label == "【其他】" else 50
        )
        if intent_label in ("【选题】", "【分工】"):
            user_only_convs = get_user_only_conversation_history(
                client=client, channel_id=channel_id, bot_id=BOT_ID,
                user_id2names=user_id2names, ts=ts, limit=50
            )
        else:
            user_only_convs = ""
        print(f"[DEBUG][handle_message] convs行数={len(convs.splitlines())}")

        # ── 按意图触发后台画像监控 ───────────────────────────────────────────
        # 仅在知识/解释/判断场景启用，避免总结或闲聊命令引发无关画像更新
        if (not profile_watch_started) and intent_label in ("【知识解答】", "【专业解释】", "【判断】"):
            watch_profile_in_background(
                client=client, channel_id=channel_id, channel_name=channel_name,
                user_id=user_id, user_name=user_name,
                agent=agent, memory=memory, profile_memory=profile_memory,
            )

        # ── ⑤ 各意图分支 ─────────────────────────────────────────────────────
        if intent_label == "【选题】":
            delete_status_message(client, channel_id, intent_status_ts)
            handle_topic_intent(TopicContext(
                client=client, channel_id=channel_id, channel_name=channel_name,
                user_id=user_id, user_name=user_name, ts=ts,
                query=query, convs=convs, intent_time=intent_time,
                agent=agent, search_engine=search_engine,
                memory=memory, search_memory=search_memory,
                profile_memory=profile_memory,
                user_id2names=user_id2names, bot_id=BOT_ID,
                user_only_convs=user_only_convs,
                is_new_channel=new_channel,
                active_user_ids=active_user_ids,
            ))
            return

        if intent_label == "【分工】":
            delete_status_message(client, channel_id, intent_status_ts)
            handle_division_intent(DivisionContext(
                client=client, channel_id=channel_id, channel_name=channel_name,
                user_id=user_id, user_name=user_name, ts=ts,
                query=query, convs=convs, intent_time=intent_time,
                agent=agent,
                memory=memory, profile_memory=profile_memory,
                user_id2names=user_id2names, bot_id=BOT_ID,
                user_only_convs=user_only_convs,
                is_new_channel=new_channel,
                active_user_ids=active_user_ids,
            ))
            return

        if intent_label == "【总结】":
            progress_ts = send_status_message(
                client, channel_id, user_id,
                "📖 正在读取对话记录，生成总结中，请稍候…"
            )
            delete_status_message(client, channel_id, intent_status_ts)
            handle_summary_intent(SummaryContext(
                client=client, channel_id=channel_id, channel_name=channel_name,
                user_id=user_id, user_name=user_name, ts=ts,
                query=query, convs=convs, intent_time=intent_time,
                agent=agent, memory=memory,
            ))
            delete_status_message(client, channel_id, progress_ts)
            return

        if intent_label in ("【知识解答】", "【专业解释】", "【判断】"):
            delete_status_message(client, channel_id, intent_status_ts)
            resolved_query = query
            rewrite_time = 0.0
            rewrite_thought = ""
            if intent_label == "【判断】":
                plan = resolve_judgment_plan(
                    agent=agent,
                    current_query=query,
                    recent_convs_text=intent_context,
                    recent_summaries=[],
                    expanded_convs_text=convs,
                )
                if plan.get("action") == "retrieve":
                    resolved_query = plan.get("search_query") or query
                else:
                    resolved_query, rewrite_time, rewrite_thought = _resolve_contextual_search_query(
                        user_name=user_name,
                        query=query,
                        convs=intent_context,
                    )
            else:
                rewrite_context = _build_rewrite_context(channel_id, user_id, user_name, intent_context)
                resolved_query, rewrite_time, rewrite_thought = _resolve_contextual_search_query(
                    user_name=user_name,
                    query=query,
                    convs=rewrite_context,
                )

            progress_ts = send_status_message(
                client, channel_id, user_id,
                f"🔍 已结合近5轮对话确认检索内容：{resolved_query}\n正在检索学术资料并生成回答，请稍候…"
            )
            _run_retrieval_intent(
                client=client,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                ts=ts,
                query=query,
                search_query=resolved_query,
                convs=convs,
                intent_label=intent_label,
                intent_time=intent_time,
                rewrite_time=rewrite_time,
                rewrite_thought=rewrite_thought,
            )
            # 记录已解释话题，防止后续新话题查询被当前解释内容污染
            if intent_label in ("【专业解释】", "【知识解答】"):
                explained_key = f"{channel_id}:{user_id}"
                _user_explained_topics.setdefault(explained_key, []).append(query)
                print(f"[DEBUG][explain_dict] 已记录话题 user={user_name!r} topic={query!r}, "
                      f"历史话题数={len(_user_explained_topics[explained_key])}")
            delete_status_message(client, channel_id, progress_ts)
            return

        if intent_label == "【其他】":
            reply, _ = agent.chitchat(query=query, convs=convs)
            delete_status_message(client, channel_id, intent_status_ts)
            response_ts = send_answer(
                client=client, channel_id=channel_id,
                user_id=user_id, answer=reply,
            )["ts"]
            memory.write_into_memory(
                table_name=channel_name,
                utterance_info={
                    "speaker": "CoSearchAgent", "utterance": reply, "convs": convs,
                    "query": query, "rewrite_query": "", "rewrite_thought": "",
                    "clarify": "", "clarify_thought": "", "clarify_cnt": 0,
                    "search_results": "",
                    "infer_time": str({"intent": intent_time}),
                    "reply_timestamp": ts, "reply_user": user_name, "timestamp": response_ts,
                }
            )
            return

        # 普通搜索/问答
        delete_status_message(client, channel_id, intent_status_ts)
        progress_ts = send_status_message(
            client, channel_id, user_id, "🔍 正在搜索资料，生成回复中，请稍候…"
        )

        rewrite_output, rewrite_time = agent.rewrite_query(user=user_name, query=query, convs=convs)
        rewrite_thought, rewrite_query = _parse_rewrite_output(rewrite_output, query)

        clarify_cnt = memory.get_clarify_cnt_for_speaker(table_name=channel_name, reply_user=user_name)

        if clarify_cnt > 0:
            clarify_thought, clarify_question, clarify_time = "", "", 0
        else:
            clarify_output, clarify_time = agent.ask_clarify_query(
                user=user_name, query=rewrite_query, convs=convs)
            clarify_output   = clarify_output.split("\n")
            clarify_thought  = clarify_output[0].lstrip("判断思路：").strip()
            clarify_question = clarify_output[1].lstrip("澄清性问题：").strip()
            if "不需要提出澄清性问题" not in clarify_question:
                delete_status_message(client, channel_id, progress_ts)
                response_ts = send_clarify_question(
                    client=client, channel_id=channel_id, user_id=user_id,
                    clarify_question=clarify_question)["ts"]
                memory.write_into_memory(
                    table_name=channel_name,
                    utterance_info={
                        "speaker": "CoSearchAgent", "utterance": clarify_question, "convs": convs,
                        "query": query, "rewrite_query": rewrite_query,
                        "rewrite_thought": rewrite_thought,
                        "clarify": clarify_question, "clarify_thought": clarify_thought,
                        "clarify_cnt": clarify_cnt + 1, "search_results": "",
                        "infer_time": str({"rewrite": rewrite_time, "clarify": clarify_time}),
                        "reply_user": user_name, "reply_timestamp": ts, "timestamp": response_ts,
                    }
                )
                return

        answer, search_results, extract_time, answer_time = agent.generate_answer(rewrite_query)
        delete_status_message(client, channel_id, progress_ts)

        response_ts = send_rag_answer(
            client=client, channel_id=channel_id, query=rewrite_query,
            user_id=user_id, answer=answer, references=search_results)["ts"]

        memory.write_into_memory(
            table_name=channel_name,
            utterance_info={
                "speaker": "CoSearchAgent", "utterance": answer, "convs": convs,
                "query": query, "rewrite_query": rewrite_query,
                "rewrite_thought": rewrite_thought,
                "clarify": clarify_question, "clarify_thought": clarify_thought,
                "clarify_cnt": 0, "search_results": str(search_results),
                "infer_time": str({"rewrite": rewrite_time, "clarify": clarify_time,
                                   "extract": extract_time, "answer": answer_time}),
                "reply_timestamp": ts, "reply_user": user_name, "timestamp": response_ts,
            }
        )
        search_memory.create_table_if_not_exists(table_name=f"{channel_name}_search")
        search_memory.write_into_memory(
            table_name=f"{channel_name}_search",
            search_info={
                "user_name": user_name, "query": rewrite_query, "answer": answer,
                "search_results": str(search_results), "start": 0, "end": 2,
                "click_time": time.time(), "timestamp": response_ts,
            }
        )
        print(f"[DEBUG][handle_message] 处理完成\n{'='*60}\n")

    finally:
        user_lock.release()


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()