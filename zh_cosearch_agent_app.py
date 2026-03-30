import time
import ast
import re
import threading
import queue
import os
import json
from dataclasses import dataclass
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
    slack_chat_update,
    SERPAPI_KEY,
    OPENAI_KEY,
    SQL_PASSWORD,
    register_channel_display_name,
)
from memory.imm_profile_store import ImmProfileStore
from memory.mental_model_memory import MentalModelMemory
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
triage_agent = CoSearchAgent(
    search_engine=search_engine,
    api_key=OPENAI_KEY,
    model_name=settings.llm_fallback_model_name,
    fallback_model_name=settings.llm_fallback_model_name,
    prompt_dir="prompts/ch",
)

memory = Memory(sql_password=SQL_PASSWORD)
search_memory = SearchMemory(sql_password=SQL_PASSWORD)
click_memory = ClickMemory(sql_password=SQL_PASSWORD)
mental_model_memory = MentalModelMemory(jl_dir="jl")
profile_memory = ImmProfileStore(mental_model_memory=mental_model_memory, sql_password=SQL_PASSWORD)
pending_memory = PendingIntentMemory(sql_password=SQL_PASSWORD)
pending_memory.create_table_if_not_exists()

channel_id2names = get_channel_info(table_name="channel_info")
user_id2names = get_user_info(table_name="user_info")
# channel_id2names = {}
# user_id2names = {}
user_id2names[BOT_ID] = "CoSearchAgent"


def _resolve_name_for_imm_bootstrap(uid: str) -> str:
    return str(user_id2names.get(uid, uid))


mental_model_memory.bootstrap_imm_from_jl(user_name_resolver=_resolve_name_for_imm_bootstrap)


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
_EXPLAIN_TRIGGER_STATE: dict = {}
_MM_ACTIVE_CHANNEL_LAST_TS: dict[str, float] = {}
_MM_ACTIVE_CHANNEL_USERS: dict[str, dict[str, float]] = {}
_MM_ARCHIVED_CHANNELS: set[str] = set()

MM_UPDATE_RECENT_ROUNDS = 5


def _has_active_judgment_followup(channel_id: str, user_id: str) -> bool:
    prefix = f"judgment:{channel_id}:{user_id}:"
    for key, state in _AUTO_FOLLOWUP_STATE.items():
        if key.startswith(prefix) and bool((state or {}).get("active")):
            return True
    return False


# 用户已解释话题字典：按频道隔离，防止不同频道的历史解释互相污染。
# key = f"{channel_id}:{user_id}", value = [query1, query2, ...]
_user_explained_topics: dict[str, list[str]] = {}


def _send_channel_intro_once(client, channel_id: str, inviter: str | None = None, reason: str = "") -> bool:
    """已停用频道打招呼逻辑。"""
    return False


def _looks_like_profile_intro(text: str) -> bool:
    """识别用户是否在当前消息中明确提供了画像线索。"""
    content = (text or "").strip()
    if not content:
        return False

    task_request_cues = (
        "总结", "归纳", "复盘", "回顾", "选题", "分工", "解释", "判断",
        "帮我", "请帮", "能不能", "可不可以", "怎么", "如何", "请问",
    )
    if any(k in content for k in task_request_cues):
        return False

    strong_patterns = (
        r"我(是|来自|学|读)(.{0,20})(专业|方向|学院)",
        r"我(是|来自|学|读)(.{0,20})(本科生|硕士生|博士生|研究生|本科|硕士|博士|phd|PhD)",
        r"(我的|我)(研究方向|研究兴趣|专业)是",
        r"我(主修|从事|主要做)",
        r"我对.{1,30}(感兴趣|有兴趣|更关注|关注|想研究|想做|计划研究)",
    )
    return any(re.search(p, content) for p in strong_patterns)


def _is_guidance_like_query(text: str) -> bool:
    """判断是否为引导用户继续补充信息的话术，而非可检索问题。"""
    q = clean_query_text(text or "")
    if not q:
        return True

    cues = (
        "请继续", "继续介绍", "继续补充", "请补充", "先介绍", "具体介绍",
        "说说你的", "说明你的", "描述你的", "进一步", "详细一点",
        "你的研究方向", "您的研究方向", "项目兴趣", "先说下", "先聊聊",
    )
    if any(c in q for c in cues):
        return True

    # 纯引导类短句通常不适合直接检索。
    if len(q) <= 20 and ("请" in q or "你" in q or "您" in q):
        return True
    return False


def _is_searchworthy_auto_query(response_type: str, query: str, user_utterance: str) -> bool:
    """非@自动回复路径的检索门槛，避免把引导语直接送入检索。"""
    q = clean_query_text(query or "")
    if not q:
        return False

    # 判断类如果是引导补充信息，先不检索。
    if response_type == "judgment" and _is_guidance_like_query(q):
        return False

    # 用户在做画像自我介绍时，优先走画像更新，不直接检索。
    if _looks_like_profile_intro(user_utterance):
        return False
    return True


def _is_smalltalk_message(text: str) -> bool:
    """识别非任务型寒暄/客套，避免触发检索。"""
    q = clean_query_text(text or "")
    if not q:
        return True

    # 明确停滞表达不应被当作寒暄拦截。
    if _is_topic_stall_signal(q):
        return False

    direct_set = {
        "你好", "您好", "哈喽", "嗨", "hi", "hello", "在吗", "在不在",
        "早上好", "中午好", "下午好", "晚上好",
        "谢谢", "感谢", "辛苦了", "收到", "好的", "ok", "okk", "嗯", "嗯嗯",
    }
    if q.lower() in direct_set:
        return True

    # 短寒暄+标点，如“你好呀”“hello~”
    short = re.sub(r"[\s~～!！.。]+", "", q).lower()
    if short in {"你好呀", "你好啊", "hello呀", "hi呀", "在吗呀", "谢谢你", "谢谢啦"}:
        return True

    # 含明显任务意图则不视为寒暄
    task_cues = (
        "总结", "选题", "分工", "解释", "判断", "怎么", "如何", "为什么", "请", "帮", "?", "？",
        "卡住", "没思路", "没有思路", "不知道", "想不出", "没进展", "没有方向", "没有想法", "没想法",
    )
    if any(c in q for c in task_cues):
        return False

    return len(q) <= 6


def _is_topic_stall_message(text: str) -> bool:
    """选题讨论中“无进展/无思路”信号。"""
    q = clean_query_text(text or "")
    if not q:
        return False
    topic_cues = ("选题", "题目", "方向", "做什么", "研究什么")
    stall_cues = ("没思路", "不知道", "想不出", "卡住", "没进展", "没有方向", "不会", "没有想法")
    return any(c in q for c in topic_cues) and any(c in q for c in stall_cues)


def _is_topic_stall_signal(text: str) -> bool:
    """宽松判定：用于 mm_decision.reason/query 的停滞信号识别。"""
    q = clean_query_text(text or "")
    if not q:
        return False
    # 兼容模型输出中意外空格/换行导致的关键词断裂。
    q_compact = re.sub(r"\s+", "", q)
    stall_cues = (
        "没思路", "没有思路", "不知道", "想不出", "卡住", "没进展", "没有方向", "无思路", "缺乏选题思路", "没有想法", "没想法", "无想法",
    )
    return any(c in q for c in stall_cues) or any(c in q_compact for c in stall_cues)


def _is_division_stall_signal(text: str) -> bool:
    """分工阶段停滞信号：仅在“分工相关 + 推进困难”时触发。"""
    q = clean_query_text(text or "")
    if not q:
        return False
    q_compact = re.sub(r"\s+", "", q)
    division_cues = (
        "分工", "任务分配", "谁负责", "谁来做", "怎么分配", "怎么分工",
    )
    stall_cues = (
        "没思路", "没有思路", "不知道", "想不出", "卡住", "没进展", "推进不动", "不清楚", "不会",
        "没有想法", "没想法", "分不清", "拿不准",
    )
    return (
        (any(c in q for c in division_cues) or any(c in q_compact for c in division_cues))
        and (any(c in q for c in stall_cues) or any(c in q_compact for c in stall_cues))
    )


def _build_imm_smm_context(channel_id: str, user_id: str, user_name: str, convs: str, query: str) -> str:
    """统一生成：频道全部非Bot IMM + SMM + 近五轮对话 + query。"""
    channel_name = channel_id2names.get(channel_id, channel_id)
    active_user_ids = _get_active_user_ids_in_channel(
        client=app.client,
        channel_id=channel_id,
        bot_id=BOT_ID,
        user_id2names=user_id2names,
        memory=memory,
        channel_name=channel_name,
    )
    imm_bundle: dict[str, dict] = {}
    seen: set[str] = set()
    for uid in active_user_ids:
        clean_uid = str(uid or "").strip()
        if not clean_uid or clean_uid == BOT_ID or clean_uid in seen:
            continue
        seen.add(clean_uid)
        uname = str(user_id2names.get(clean_uid) or clean_uid)
        imm_bundle[clean_uid] = mental_model_memory.get_imm(user_id=clean_uid, user_name=uname)

    # 兜底：至少包含当前用户。
    if not imm_bundle:
        imm_bundle[user_id] = mental_model_memory.get_imm(user_id=user_id, user_name=user_name)

    smm = mental_model_memory.get_smm(channel_id=channel_id)
    conv_lines = [line for line in str(convs or "").splitlines() if str(line).strip()]
    recent_convs = "\n".join(conv_lines[-INTENT_CONTEXT_MESSAGE_LIMIT:])
    return (
        "【IMM(频道非Bot用户合集)】\n"
        f"{json.dumps(imm_bundle or {}, ensure_ascii=False)}\n\n"
        "【SMM】\n"
        f"{json.dumps(smm or {}, ensure_ascii=False)}\n\n"
        "【近五轮对话】\n"
        f"{recent_convs}\n\n"
        "【当前问题】\n"
        f"{query}"
    )


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
AUTO_RECOGNIZE_EVERY_ROUNDS = _int_env("AUTO_RECOGNIZE_EVERY_ROUNDS", 1)
LOW_INFO_TERM_WINDOW_SECONDS = _int_env("LOW_INFO_TERM_WINDOW_SECONDS", 100)
EXPLAIN_REPEAT_COOLDOWN_SECONDS = _int_env("EXPLAIN_REPEAT_COOLDOWN_SECONDS", 180)
AUTO_CONFIRM_EXPIRE_SECONDS = _int_env("AUTO_CONFIRM_EXPIRE_SECONDS", 180)
FOLLOWUP_ROUNDS = _int_env("FOLLOWUP_ROUNDS", 3)
FOLLOWUP_POLL_SECONDS = _int_env("FOLLOWUP_POLL_SECONDS", 2)
FOLLOWUP_MAX_CYCLES = _int_env("FOLLOWUP_MAX_CYCLES", 4)
FOLLOWUP_MAX_SECONDS = _int_env("FOLLOWUP_MAX_SECONDS", 300)
MM_TIMER_POLL_SECONDS = _int_env("MM_TIMER_POLL_SECONDS", 5)
MM_ACTIVE_CHANNEL_WINDOW_SECONDS = _int_env("MM_ACTIVE_CHANNEL_WINDOW_SECONDS", 300)
MM_TERM_SOLVING_REVIEW_SECONDS = _int_env("MM_TERM_SOLVING_REVIEW_SECONDS", 300)
MESSAGE_LOCK_WAIT_SECONDS = _int_env("MESSAGE_LOCK_WAIT_SECONDS", 20)
EVENT_DEDUP_WINDOW_SECONDS = _int_env("EVENT_DEDUP_WINDOW_SECONDS", 300)

# 异步处理队列：按 (channel_id, user_id) 分区串行，避免同一用户消息乱序处理。
@dataclass
class _MessageTask:
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    ts: str
    event_type: str
    raw_text: str
    user_utterance: str
    query: str
    mentioned_bot: bool


_message_task_queues: dict[tuple[str, str], queue.Queue] = {}
_message_worker_threads: dict[tuple[str, str], threading.Thread] = {}
_worker_mutex = threading.Lock()
_mm_timer_worker_thread: threading.Thread | None = None

# 事件去重：避免同一条消息被 app_mention + message 双订阅重复处理。
# key: (channel_id, user_id, ts) -> first_seen_epoch
_recent_message_event_keys: dict[tuple[str, str, str], float] = {}
_event_dedupe_lock = threading.Lock()


def _is_duplicate_message_event(channel_id: str, user_id: str, ts: str, event_type: str = "") -> bool:
    now = time.time()
    key = (channel_id, user_id, ts)
    with _event_dedupe_lock:
        first_seen = _recent_message_event_keys.get(key)
        if first_seen and (now - first_seen) <= EVENT_DEDUP_WINDOW_SECONDS:
            print(
                f"[DEBUG][handle_message] 去重命中，跳过重复事件 "
                f"type={event_type!r} channel={channel_id!r} user={user_id!r} ts={ts!r}"
            )
            return True

        _recent_message_event_keys[key] = now

        # 仅在键数较多时清理过期项，控制常驻内存。
        if len(_recent_message_event_keys) > 2000:
            expire_before = now - EVENT_DEDUP_WINDOW_SECONDS
            expired_keys = [k for k, seen_at in _recent_message_event_keys.items() if seen_at < expire_before]
            for k in expired_keys:
                _recent_message_event_keys.pop(k, None)

    return False

def _get_message_task_queue(channel_id: str, user_id: str) -> queue.Queue:
    key = (channel_id, user_id)
    with _worker_mutex:
        q = _message_task_queues.get(key)
        if q is None:
            q = queue.Queue()
            _message_task_queues[key] = q
        return q


def _ensure_message_worker(client, channel_id: str, user_id: str) -> None:
    key = (channel_id, user_id)
    with _worker_mutex:
        thread = _message_worker_threads.get(key)
        if thread and thread.is_alive():
            return

        q = _message_task_queues.get(key)
        if q is None:
            q = queue.Queue()
            _message_task_queues[key] = q

        def _worker():
            while True:
                task = q.get()
                try:
                    _process_message_task(client=client, task=task)
                except Exception as e:
                    print(
                        f"[DEBUG][handle_message] 后台处理异常 channel={task.channel_id!r} "
                        f"user={task.user_id!r} ts={task.ts!r}: {e}"
                    )
                finally:
                    q.task_done()

        thread = threading.Thread(target=_worker, daemon=True)
        _message_worker_threads[key] = thread
        thread.start()


def _enqueue_message_task(client, task: _MessageTask) -> None:
    _ensure_message_worker(client=client, channel_id=task.channel_id, user_id=task.user_id)
    q = _get_message_task_queue(task.channel_id, task.user_id)
    q.put(task)

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
            speaker = str(row.get("speaker", "")).strip()
            if speaker and speaker != "CoSearchAgent":
                speakers.add(speaker)

        # 反查 user_id：user_id2names 是 {uid: uname}
        # 兼容两种 speaker 落库格式：真实用户名 或 Slack UID（例如 U0A2JNV4WTT）
        speakers_lower = {s.lower() for s in speakers}
        active = []
        for uid, uname in user_id2names.items():
            if uid == bot_id or uid == "bot_id":
                continue
            uname_str = str(uname or "").strip()
            if uid in speakers or (uname_str and uname_str.lower() in speakers_lower):
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
    imm = mental_model_memory.get_imm(user_id=user_id, user_name=user_name)
    imm_profile = (imm or {}).get("个人画像") if isinstance((imm or {}).get("个人画像"), dict) else {}
    imm_kb = (imm or {}).get("个人领域知识库") if isinstance((imm or {}).get("个人领域知识库"), dict) else {}
    imm_major = str(imm_profile.get("专业领域") or (imm or {}).get("professional_background") or "").strip()
    top_terms = list(imm_kb.get("提取术语") or (imm or {}).get("familiar_terms") or [])[:12]

    if not profile:
        return (
            f"用户：{user_name}\n"
            f"专业：{imm_major or '未知'}\n"
            "研究兴趣：暂无\n"
            "方法偏好：暂无\n"
            f"已掌握术语：{'、'.join(top_terms) if top_terms else '暂无'}"
        )

    interests = "、".join(profile.get("research_interests") or []) or "暂无"
    methods = "、".join(profile.get("methodology") or []) or "暂无"
    major = imm_major or profile.get("major") or "未知"
    return (
        f"用户：{profile.get('user_name') or user_name}\n"
        f"专业：{major}\n"
        f"研究兴趣：{interests}\n"
        f"方法偏好：{methods}\n"
        f"已掌握术语：{'、'.join(top_terms) if top_terms else '暂无'}"
    )


def _infer_mention_response_type(query: str, imm: dict) -> str:
    """在@场景下，用查询文本 + IMM 快速推断回复类型（不再额外调用LLM）。"""
    text = clean_query_text(query or "")
    if not text:
        return "knowledge"
    if is_decision_like_message(text) or is_conflict_like_message(text):
        return "judgment"
    if has_confusion_cue(text):
        return "professional_explain"

    imm_kb = (imm or {}).get("个人领域知识库") if isinstance((imm or {}).get("个人领域知识库"), dict) else {}
    terms = list(imm_kb.get("提取术语") or (imm or {}).get("familiar_terms") or []) + list((imm or {}).get("known_terms") or [])
    lowered = text.lower()
    for term in terms:
        clean_term = str(term or "").strip().lower()
        if clean_term and clean_term in lowered:
            return "professional_explain"
    return "knowledge"


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


def _is_low_info_followup(text: str) -> bool:
    content = clean_query_text(text or "").lower()
    if not content:
        return False
    low_info_phrases = (
        "这是啥", "这是什么", "啥", "是什么", "什么意思", "难吗", "没学过", "没学过啊", "不懂", "听不懂",
    )
    if content in low_info_phrases:
        return True
    return len(content) <= 6 and has_confusion_cue(content)


def _extract_recent_term_from_convs(convs: str, lookback_lines: int = 12) -> str:
    if not convs:
        return ""
    lines = [ln.strip() for ln in str(convs).splitlines() if ln.strip()]
    for line in reversed(lines[-lookback_lines:]):
        terms = extract_candidate_terms(line)
        if terms:
            return _normalize_term(terms[0])
    return ""


def _strip_speaker_prefix(line: str) -> str:
    text = (line or "").strip()
    if not text:
        return ""
    if ":" in text:
        return text.split(":", 1)[1].strip()
    if "：" in text:
        return text.split("：", 1)[1].strip()
    return text


def _extract_recent_user_texts(convs: str, lookback_lines: int = 12) -> list[str]:
    if not convs:
        return []
    lines = [ln.strip() for ln in str(convs).splitlines() if ln.strip()]
    result: list[str] = []
    for line in lines[-lookback_lines:]:
        if line.startswith("CoSearchAgent:"):
            continue
        text = _strip_speaker_prefix(line)
        if text:
            result.append(text)
    return result


def _extract_recent_decision_query(convs: str, fallback: str) -> str:
    recent_texts = _extract_recent_user_texts(convs, lookback_lines=12)
    for text in reversed(recent_texts):
        if is_decision_like_message(text):
            return clean_query_text(text)
    return clean_query_text(fallback)


def _has_conflict_escalation(convs: str, current_text: str, min_signals: int = 3) -> bool:
    recent_texts = _extract_recent_user_texts(convs, lookback_lines=10)
    signals = 0
    for text in recent_texts + [clean_query_text(current_text)]:
        if is_conflict_like_message(text) or is_decision_like_message(text):
            signals += 1
    return signals >= min_signals


def _extract_recent_term_from_channel_window(
    channel_name: str,
    now_ts: float,
    window_seconds: int = 100,
    max_scan_rows: int = 120,
) -> str:
    if not channel_name:
        return ""
    cutoff_ts = float(now_ts) - float(window_seconds)
    try:
        rows = memory.load_all_utterances(table_name=channel_name)
    except Exception:
        return ""

    scanned = 0
    for row in reversed(rows):
        if scanned >= max_scan_rows:
            break
        scanned += 1

        speaker = str(row.get("speaker") or "")
        if speaker == "CoSearchAgent":
            continue

        utterance = str(row.get("utterance") or "").strip()
        if not utterance:
            continue

        ts_raw = row.get("timestamp") or "0"
        try:
            ts_val = float(ts_raw)
        except Exception:
            continue

        if ts_val < cutoff_ts:
            break

        terms = extract_candidate_terms(utterance)
        if terms:
            return _normalize_term(terms[0])
    return ""


def _build_auto_key(channel_id: str, user_id: str) -> str:
    return f"{channel_id}:{user_id}"


def _build_explain_term_key(channel_id: str, user_id: str, term: str) -> str:
    return f"{channel_id}:{user_id}:{_normalize_term(term)}"


def _get_explain_trigger_meta(channel_id: str, user_id: str, term: str) -> dict:
    key = _build_explain_term_key(channel_id, user_id, term)
    return _EXPLAIN_TRIGGER_STATE.get(key) or {"count": 0, "last_ts": 0.0}


def _record_explain_trigger(channel_id: str, user_id: str, term: str):
    clean_term = _normalize_term(term)
    if not clean_term:
        return
    key = _build_explain_term_key(channel_id, user_id, clean_term)
    meta = _EXPLAIN_TRIGGER_STATE.get(key) or {"count": 0, "last_ts": 0.0}
    meta["count"] = int(meta.get("count", 0)) + 1
    meta["last_ts"] = _now_ts()
    _EXPLAIN_TRIGGER_STATE[key] = meta


def _build_auto_confirm_text(kind: str, term: str, repeat_explain: bool = False) -> str:
    if kind == "explain":
        if repeat_explain:
            return f"检测到你对术语“{term or '该术语'}”可能还有疑问，需要我进一步解释吗？"
        return f"检测到你们在讨论术语“{term or '该术语'}”，需要我现在给一个专业解释吗？"
    return "检测到你们可能在争论同一问题，需要我发起一次判断分析吗？"


def _build_auto_confirm_blocks(kind: str, term: str, repeat_explain: bool = False) -> list[dict]:
    text = _build_auto_confirm_text(kind=kind, term=term, repeat_explain=repeat_explain)
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{text}\n\n点击按钮确认，确认后我再开始检索。",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "需要"},
                    "style": "primary",
                    "action_id": "auto_prompt_accept",
                    "value": "yes",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "不需要"},
                    "action_id": "auto_prompt_decline",
                    "value": "no",
                },
            ],
        },
    ]


def _update_auto_prompt_card(client, channel_id: str, pending: dict, text: str):
    prompt_ts = str(pending.get("prompt_ts") or "").strip()
    if not prompt_ts:
        return
    try:
        slack_chat_update(
            client=client,
            channel=channel_id,
            ts=prompt_ts,
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": text,
                    },
                }
            ],
        )
    except Exception as e:
        print(f"[DEBUG][auto_prompt] 更新确认卡片失败: {e}")


def _parse_explain_search_output(raw_output: str, fallback_query: str) -> tuple[str, str]:
    thought = ""
    rewritten_query = ""

    for line in (raw_output or "").splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        if clean_line.startswith("检索思路："):
            thought = clean_line.split("：", 1)[-1].strip()
        elif clean_line.startswith("检索词："):
            rewritten_query = clean_line.split("：", 1)[-1].strip()

    rewritten_query = clean_query_text(rewritten_query or fallback_query)
    return thought, rewritten_query


def _fallback_explain_search_query(query: str, term: str = "") -> str:
    normalized_term = _normalize_term(term)
    if normalized_term:
        if re.fullmatch(r"[a-z]{2,12}", normalized_term):
            return normalized_term.upper()
        return normalized_term
    return clean_query_text(query)


def _resolve_professional_explain_search_query(query: str, convs: str, term: str = "") -> tuple[str, float, str]:
    fallback_query = _fallback_explain_search_query(query=query, term=term)
    try:
        rewrite_output, rewrite_time = agent.rewrite_professional_explain_query(
            query=query,
            convs=convs,
            term=term,
        )
        rewrite_thought, rewrite_query = _parse_explain_search_output(rewrite_output, fallback_query)
        return rewrite_query, rewrite_time, rewrite_thought
    except Exception as e:
        print(f"[DEBUG][explain_rewrite] 检索词改写失败，回退默认术语检索: {e}")
        return fallback_query, 0.0, ""


def _queue_auto_prompt(
    client,
    channel_name: str,
    channel_id: str,
    user_id: str,
    user_name: str,
    kind: str,
    term: str,
    query: str,
    reason: str,
    trigger_ts: str,
):
    repeat_explain = False
    if kind == "explain":
        clean_term = _normalize_term(term)
        meta = _get_explain_trigger_meta(channel_id=channel_id, user_id=user_id, term=clean_term)
        prev_count = int(meta.get("count", 0))
        last_ts = float(meta.get("last_ts", 0.0))
        now_ts = _now_ts()
        # 避免短时间内同术语重复触发太多次；第2次允许并改成“进一步解释”文案。
        if prev_count >= 2 and (now_ts - last_ts) < EXPLAIN_REPEAT_COOLDOWN_SECONDS:
            print(
                f"[DEBUG][auto_prompt] 跳过重复解释 term={clean_term!r} "
                f"count={prev_count} cooldown={EXPLAIN_REPEAT_COOLDOWN_SECONDS}s"
            )
            return
        repeat_explain = prev_count >= 1

    # 取消二次确认：自动识别到可介入场景后直接执行。
    key = _build_auto_key(channel_id, user_id)
    _AUTO_PROMPT_STATE[key] = {
        "kind": kind,
        "term": term or "",
        "query": query or "",
        "reason": reason or "",
        "repeat_explain": repeat_explain,
        "trigger_ts": str(trigger_ts),
        "created_at": _now_ts(),
    }
    _execute_auto_prompt_decision(
        client=client,
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        user_name=user_name,
        action_ts=str(trigger_ts),
        decision="yes",
    )
    if kind == "explain":
        _record_explain_trigger(channel_id=channel_id, user_id=user_id, term=term)


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
                # 到达观察上限且未出现新的困惑跟进，视为该术语暂时已掌握。
                mental_model_memory.update_unknown_term_status(
                    user_id=user_id,
                    user_name=user_name,
                    term=term,
                    status="已解决",
                    note="观察窗口内未继续追问，自动结案",
                    reset_timer=False,
                )
                _mark_term_known(channel_id=channel_id, user_id=user_id, user_name=user_name, term=term)
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
                    mental_model_memory.update_unknown_term_status(
                        user_id=user_id,
                        user_name=user_name,
                        term=term,
                        status="解决中",
                        note="继续追问，触发再次解释",
                        reset_timer=True,
                    )
                    cycles += 1
                    rounds_left = FOLLOWUP_ROUNDS
                    mention_seen = False
                    confusion_seen = False
                    understood_seen = False
                    continue

                if (not mention_seen) or understood_seen:
                    mental_model_memory.update_unknown_term_status(
                        user_id=user_id,
                        user_name=user_name,
                        term=term,
                        status="已解决",
                        note=("用户明确表示已理解" if understood_seen else "后续未继续讨论该术语"),
                        reset_timer=False,
                    )
                    _mark_term_known(channel_id=channel_id, user_id=user_id, user_name=user_name, term=term)
                    _AUTO_FOLLOWUP_STATE[follow_key] = {"active": False}
                    return

                cycles += 1
                rounds_left = FOLLOWUP_ROUNDS
                mention_seen = False
                confusion_seen = False
                understood_seen = False
                if cycles >= FOLLOWUP_MAX_CYCLES:
                    # 多轮解释后仍持续困惑，回退为未解决，等待后续计时器再介入。
                    mental_model_memory.update_unknown_term_status(
                        user_id=user_id,
                        user_name=user_name,
                        term=term,
                        status="未解决",
                        note="多轮解释后仍存在困惑，等待后续介入",
                        reset_timer=True,
                    )
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


def _auto_triage(
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    message_text: str,
    convs: str,
    message_ts: str,
) -> dict:
    start_time = time.time()
    print(
        f"[DEBUG][auto_triage] 开始判定 user={user_name!r} text={message_text!r} "
        f"convs_lines={len((convs or '').splitlines())}"
    )
    profile = profile_memory.load(user_id=user_id, channel_id=channel_id) or {}
    imm = mental_model_memory.get_imm(user_id=user_id, user_name=user_name)
    imm_profile = (imm or {}).get("个人画像") if isinstance((imm or {}).get("个人画像"), dict) else {}
    imm_kb = (imm or {}).get("个人领域知识库") if isinstance((imm or {}).get("个人领域知识库"), dict) else {}
    imm_stance = (imm or {}).get("个人任务认知 (Task Stance)") if isinstance((imm or {}).get("个人任务认知 (Task Stance)"), dict) else {}
    major = str(profile.get("major") or "未知")
    interests = "、".join(profile.get("research_interests") or []) or "暂无"
    known_terms = profile_memory.get_known_terms(user_id=user_id, channel_id=channel_id)
    imm_major = str(imm_profile.get("专业领域") or (imm or {}).get("professional_background") or "").strip()
    imm_terms = list(imm_kb.get("提取术语") or (imm or {}).get("familiar_terms") or [])[:20]
    focus = str(imm_stance.get("期望研究方向") or (imm or {}).get("project_understanding") or "").strip()
    imm_points = [focus] if focus else []
    merged_major = imm_major or major

    # 仅自我介绍（如“我是金融博士”）不应触发 explain/judgment。
    intro_only_pattern = re.compile(
        r"^\s*(我(是|来自|读|在读|学|学的是|专业是)|本人|目前)"
        r".{0,30}(专业|博士|硕士|本科|研究生|学生|方向).{0,20}$"
    )

    # 规则优先：当前消息明确是选择/判断问题时，直接进入 judgment。
    if is_decision_like_message(message_text):
        elapsed = time.time() - start_time
        result = {
            "kind": "judgment",
            "term": "",
            "query": clean_query_text(message_text),
            "reason": "rule_decision_like_current",
        }
        print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
        return result

    # 规则优先：争论信号可跨多轮累积，不要求单轮就出现完整选择句。
    if is_conflict_like_message(message_text) and _has_conflict_escalation(convs, message_text, min_signals=3):
        elapsed = time.time() - start_time
        result = {
            "kind": "judgment",
            "term": "",
            "query": _extract_recent_decision_query(convs, message_text),
            "reason": "rule_conflict_escalation",
        }
        print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
        return result

    # 对“这是啥/难吗/没学过”等低信息跟进短句，优先绑定最近对话中的术语，
    # 避免模型在长历史中回指更早的话题。
    if _is_low_info_followup(message_text):
        try:
            now_ts = float(message_ts)
        except Exception:
            now_ts = _now_ts()
        recent_term = _extract_recent_term_from_channel_window(
            channel_name=channel_name,
            now_ts=now_ts,
            window_seconds=LOW_INFO_TERM_WINDOW_SECONDS,
        )
        if not recent_term:
            recent_term = _extract_recent_term_from_convs(convs)
        if recent_term and not _term_already_known(channel_id=channel_id, user_id=user_id, term=recent_term):
            elapsed = time.time() - start_time
            result = {
                "kind": "explain",
                "term": recent_term,
                "query": clean_query_text(f"什么是{recent_term}"),
                "reason": "low_info_followup_bind_recent_term",
            }
            print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
            return result

    prompt = (
        "你是多轮对话自动识别器。目标：先识别是否有需要专业解释的术语，指的是某专业的专业名词出现，而用户不是该专业的，再识别是否存在争论需要判断。\n"
        "仅输出 JSON："
        "{\"kind\":\"explain|judgment|none\",\"term\":\"...\",\"query\":\"...\",\"reason\":\"...\"}\n"
        "规则：\n"
        "1) explain 优先级高于 judgment。\n"
        "2) explain 仅在出现专业术语且该术语与用户专业可能存在认知差时触发。\n"
        "3) term 必须是术语本体（例如 rag）。\n"
        "4) judgment 仅在出现明确争论/选择冲突时触发。\n"
        "5) 无需介入时输出 kind=none。\n"
        "6) 仅有身份/背景陈述（如‘我是XX专业/博士/学生’）且没有提问、比较、求解释意图时，必须输出 kind=none。\n"
        "7) 不得把用户自我身份词当作术语（如‘金融博士’、‘法学硕士’不是 explain 的 term）。\n\n"
        "8) explain 的 term 必须来自当前消息或最近2轮用户发言，不得从更早历史话题回指。\n\n"
        f"用户：{user_name}\n"
        f"专业：{merged_major}\n"
        f"研究兴趣：{interests}\n"
        f"已知术语（不可重复触发）：{known_terms}\n"
        f"IMM术语掌握：{imm_terms}\n"
        f"IMM可能触发知识点：{imm_points[:20]}\n"
        f"当前消息：{message_text}\n"
        f"最近对话：\n{convs}\n"
    )

    try:
        raw = triage_agent.generate_openai_response(prompt)
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

    # 后置兜底：自我介绍句误判为 explain 时强制降级为 none。
    if kind == "explain" and intro_only_pattern.match((message_text or "").strip()):
        elapsed = time.time() - start_time
        result = {"kind": "none", "term": "", "query": "", "reason": "intro_only_statement"}
        print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
        return result

    if kind == "explain" and _term_already_known(channel_id=channel_id, user_id=user_id, term=term):
        if is_decision_like_message(message_text) or _has_conflict_escalation(convs, message_text, min_signals=3):
            elapsed = time.time() - start_time
            result = {
                "kind": "judgment",
                "term": "",
                "query": _extract_recent_decision_query(convs, message_text),
                "reason": "fallback_from_term_known_to_judgment",
            }
            print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
            return result
        elapsed = time.time() - start_time
        result = {"kind": "none", "term": "", "query": "", "reason": "term_known"}
        print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
        return result

    if kind == "judgment" and not query:
        query = clean_query_text(message_text)

    if kind == "explain" and not query:
        query = clean_query_text(message_text)

    elapsed = time.time() - start_time
    result = {"kind": kind, "term": term, "query": query, "reason": reason}
    print(f"[DEBUG][auto_triage] 判定结束 elapsed={elapsed:.2f}s result={result}")
    return result


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


def _send_auto_confirm_prompt(
    client,
    channel_id: str,
    user_id: str,
    kind: str,
    term: str,
    repeat_explain: bool = False,
):
    blocks = _build_auto_confirm_blocks(kind=kind, term=term, repeat_explain=repeat_explain)
    response = client.chat_postMessage(
        channel=channel_id,
        text=(
            _build_auto_confirm_text(kind=kind, term=term, repeat_explain=repeat_explain)
            + " 点击按钮确认后我再开始检索。"
        ),
        blocks=blocks,
    )
    key = _build_auto_key(channel_id, user_id)
    pending = _AUTO_PROMPT_STATE.get(key) or {}
    pending["prompt_ts"] = response.get("ts")
    _AUTO_PROMPT_STATE[key] = pending


def _execute_auto_prompt_decision(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    action_ts: str,
    decision: str,
) -> bool:
    pending = _get_auto_prompt(channel_id, user_id)
    if not pending:
        return False

    kind = str(pending.get("kind") or "")
    term = str(pending.get("term") or "")
    query = clean_query_text(str(pending.get("query") or ""))
    trigger_ts = str(pending.get("trigger_ts") or action_ts)
    convs = get_conversation_history(
        client=client,
        channel_id=channel_id,
        bot_id=BOT_ID,
        user_id2names=user_id2names,
        ts=action_ts,
        limit=30,
    )

    if decision == "no":
        if kind == "explain":
            _mark_term_known(channel_id=channel_id, user_id=user_id, user_name=user_name, term=term)
        _update_auto_prompt_card(
            client=client,
            channel_id=channel_id,
            pending=pending,
            text="已忽略本次建议，如需我介入，直接 @我 或再次触发即可。",
        )
        _clear_auto_prompt(channel_id, user_id)
        return True

    rewrite_time = 0.0
    rewrite_thought = ""
    search_query = query
    if kind == "explain":
        search_query, rewrite_time, rewrite_thought = _resolve_professional_explain_search_query(
            query=query,
            convs=convs,
            term=term,
        )
    elif kind == "judgment":
        plan = resolve_judgment_plan(
            agent=agent,
            current_query=query,
            recent_convs_text=convs,
            recent_summaries=[],
            expanded_convs_text=convs,
        )
        search_query = plan.get("search_query") if plan.get("action") == "retrieve" else query

    _update_auto_prompt_card(
        client=client,
        channel_id=channel_id,
        pending=pending,
        text=f"已确认，开始处理：{search_query}",
    )

    if kind == "explain":
        # 若同一用户在该频道已有未结束的判断跟进，直接给出明确结论，
        # 避免重复进入“优缺点”流程造成二次触发噪音。
        if _has_active_judgment_followup(channel_id=channel_id, user_id=user_id):
            _run_special_judgment_choice(
                client=client,
                channel_id=channel_id,
                user_id=user_id,
                user_name=user_name,
                channel_name=channel_name,
            )
            _clear_auto_prompt(channel_id, user_id)
            return True
        _run_retrieval_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=trigger_ts,
            query=query,
            search_query=search_query,
            convs=convs,
            intent_label="【专业解释】",
            intent_time=0.0,
            rewrite_time=rewrite_time,
            rewrite_thought=rewrite_thought,
        )
        _start_explain_followup_worker(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            term=term or query,
            trigger_ts=float(action_ts),
        )
    elif kind == "judgment":
        _run_retrieval_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=trigger_ts,
            query=query,
            search_query=search_query,
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
            trigger_ts=float(action_ts),
        )

    _clear_auto_prompt(channel_id, user_id)
    return True


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
    return _execute_auto_prompt_decision(
        client=client,
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        user_name=user_name,
        action_ts=ts,
        decision=decision,
    )


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
        limit=12,
    )
    triage = _auto_triage(
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        user_name=user_name,
        message_text=text,
        convs=convs,
        message_ts=ts,
    )
    kind = triage.get("kind")
    if kind not in ("explain", "judgment"):
        return

    if kind == "explain" and _term_already_known(channel_id, user_id, triage.get("term") or ""):
        return

    _queue_auto_prompt(
        client=client,
        channel_name=channel_name,
        channel_id=channel_id,
        user_id=user_id,
        user_name=user_name,
        kind=kind,
        term=triage.get("term") or "",
        query=triage.get("query") or clean_query_text(text),
        reason=triage.get("reason") or "",
        trigger_ts=ts,
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
        user_profile = _build_imm_smm_context(
            channel_id=channel_id,
            user_id=user_id,
            user_name=user_name,
            convs=convs,
            query=query,
        )
        audience_model = (
            "请使用该用户已知领域做类比；尽量先给直觉解释，再给一句简化定义；"
            "控制术语密度，优先保证可理解性。"
        )
        if search_query == clean_query_text(query) and not rewrite_thought:
            search_query, rewrite_time, rewrite_thought = _resolve_professional_explain_search_query(
                query=query,
                convs=convs,
                term="",
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


def _classify_full_intent(
    query: str,
    convs: str,
    channel_id: str,
    user_id: str,
    user_name: str,
) -> str:
    """[DEBUG][_classify_full_intent] 七意图识别。"""
    imm = mental_model_memory.get_imm(user_id=user_id, user_name=user_name)
    smm = mental_model_memory.get_smm(channel_id=channel_id)

    prompt = (
        "你是协作研究助手的意图识别器。请根据用户的发言和当前协作状态，"
        "从以下七个意图中选择最匹配的一个，只输出意图标签，不要输出其他内容：\n"
        "【选题】：用户希望讨论或确定研究合作方向、不知道做什么\n"
        "【分工】：用户希望规划团队任务分工\n"
        "【总结】：用户希望总结当前对话或讨论进展\n"
        "【专业解释】：用户对某个术语或概念不理解，需要解释\n"
        "【知识解答】：用户希望获取事实、背景、概念信息\n"
        "【判断】：用户需要对多个方案或立场进行比较分析\n"
        "【其他】：日常闲聊或不属于以上任何类别\n\n"
        f"用户：{user_name}\n"
        f"当前消息：{query}\n"
        f"近期对话：\n{convs}\n"
        f"当前IMM（用户画像）：{json.dumps(imm or {}, ensure_ascii=False)}\n"
        f"当前SMM（协作状态）：{json.dumps(smm or {}, ensure_ascii=False)}\n"
        "请输出意图标签："
    )

    valid = {"【选题】", "【分工】", "【总结】", "【专业解释】", "【知识解答】", "【判断】", "【其他】"}
    try:
        raw = triage_agent.generate_openai_response(prompt)
        label = (raw or "").strip()
        if label in valid:
            print(f"[DEBUG][_classify_full_intent] intent={label!r} query={query!r}")
            return label
        for v in valid:
            if v in label:
                print(f"[DEBUG][_classify_full_intent] 模糊匹配 intent={v!r} raw={label!r}")
                return v
    except Exception as e:
        print(f"[DEBUG][_classify_full_intent] 意图识别失败，降级到规则: {e}")

    q = clean_query_text(query)
    if any(k in q for k in ("选题", "题目", "研究方向", "做什么")):
        return "【选题】"
    if any(k in q for k in ("分工", "谁做", "怎么分配", "任务分配")):
        return "【分工】"
    if any(k in q for k in ("总结", "归纳", "复盘", "梳理")):
        return "【总结】"
    if is_decision_like_message(q) or is_conflict_like_message(q):
        return "【判断】"
    if has_confusion_cue(q):
        return "【专业解释】"
    if _is_smalltalk_message(q):
        return "【其他】"
    return "【知识解答】"


def _dispatch_intent(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    ts: str,
    query: str,
    convs: str,
    intent_label: str,
    intent_time: float,
):
    """[DEBUG][_dispatch_intent] 七意图统一分发。"""
    print(f"[DEBUG][_dispatch_intent] intent={intent_label!r} user={user_name!r} query={query!r}")
    imm_smm_context = _build_imm_smm_context(
        channel_id=channel_id,
        user_id=user_id,
        user_name=user_name,
        convs=convs,
        query=query,
    )

    if intent_label == "【选题】":
        active_user_ids = _get_active_user_ids_in_channel(
            client, channel_id, BOT_ID, user_id2names, memory, channel_name
        )
        user_only_convs = get_user_only_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=INTENT_CONTEXT_MESSAGE_LIMIT,
        )
        ctx = TopicContext(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            convs=convs,
            intent_time=intent_time,
            agent=agent,
            search_engine=search_engine,
            memory=memory,
            search_memory=search_memory,
            profile_memory=profile_memory,
            user_id2names=user_id2names,
            bot_id=BOT_ID,
            sql_password=SQL_PASSWORD,
            user_only_convs=user_only_convs,
            active_user_ids=active_user_ids,
            imm_smm_context=imm_smm_context,
        )
        handle_topic_intent(ctx)
        return

    if intent_label == "【分工】":
        active_user_ids = _get_active_user_ids_in_channel(
            client, channel_id, BOT_ID, user_id2names, memory, channel_name
        )
        user_only_convs = get_user_only_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=INTENT_CONTEXT_MESSAGE_LIMIT,
        )
        ctx = DivisionContext(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            convs=convs,
            intent_time=intent_time,
            agent=agent,
            memory=memory,
            profile_memory=profile_memory,
            bot_id=BOT_ID,
            sql_password=SQL_PASSWORD,
            user_id2names=user_id2names,
            user_only_convs=user_only_convs,
            active_user_ids=active_user_ids,
            imm_smm_context=imm_smm_context,
        )
        handle_division_intent(ctx)
        return

    if intent_label == "【总结】":
        ctx = SummaryContext(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            convs=convs,
            intent_time=intent_time,
            agent=agent,
            memory=memory,
            imm_smm_context=imm_smm_context,
        )
        handle_summary_intent(ctx)
        return

    if intent_label in ("【专业解释】", "【知识解答】", "【判断】"):
        search_query = clean_query_text(query)
        rewrite_time = 0.0
        rewrite_thought = ""

        if intent_label == "【专业解释】":
            search_query, rewrite_time, rewrite_thought = _resolve_professional_explain_search_query(
                query=query,
                convs=convs,
                term="",
            )
        elif intent_label == "【判断】":
            if _has_active_judgment_followup(channel_id=channel_id, user_id=user_id):
                _run_special_judgment_choice(
                    client=client,
                    channel_id=channel_id,
                    user_id=user_id,
                    user_name=user_name,
                    channel_name=channel_name,
                )
                return
            plan = resolve_judgment_plan(
                agent=agent,
                current_query=query,
                recent_convs_text=convs,
                recent_summaries=[],
                expanded_convs_text=convs,
            )
            if plan.get("action") == "retrieve":
                search_query = plan.get("search_query") or search_query

        _run_retrieval_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            search_query=search_query,
            convs=convs,
            intent_label=intent_label,
            intent_time=intent_time,
            rewrite_time=rewrite_time,
            rewrite_thought=rewrite_thought,
        )

        if intent_label == "【专业解释】":
            _start_explain_followup_worker(
                client=client,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                term=search_query,
                trigger_ts=float(ts),
            )
        elif intent_label == "【判断】":
            _start_judgment_followup_worker(
                client=client,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                trigger_ts=float(ts),
            )
        return

    # 【其他】：闲聊轻回复
    imm = mental_model_memory.get_imm(user_id=user_id, user_name=user_name)
    smm = mental_model_memory.get_smm(channel_id=channel_id)
    prompt = (
        f"你是跨学科协作助手，正在帮助 {user_name} 进行学术合作。\n"
        f"用户画像：{json.dumps(imm or {}, ensure_ascii=False)}\n"
        f"协作状态：{json.dumps(smm or {}, ensure_ascii=False)}\n"
        f"近期对话：\n{convs}\n"
        f"用户说：{query}\n"
        "请给出一句简短、友好的回复。"
    )
    try:
        answer = agent.generate_openai_response(prompt)
        response = send_answer(client=client, channel_id=channel_id, user_id=user_id, answer=answer)
        memory.write_into_memory(
            table_name=channel_name,
            utterance_info={
                "speaker": "CoSearchAgent",
                "utterance": answer,
                "convs": convs,
                "query": query,
                "rewrite_query": query,
                "rewrite_thought": "",
                "clarify": "",
                "clarify_thought": "",
                "clarify_cnt": 0,
                "search_results": "",
                "infer_time": str({"workflow": "smalltalk"}),
                "reply_timestamp": ts,
                "reply_user": user_name,
                "timestamp": response.get("ts", ""),
            },
        )
    except Exception as e:
        print(f"[DEBUG][_dispatch_intent] 闲聊回复失败: {e}")


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
    start_time = time.time()
    print(
        f"[DEBUG][periodic] 开始判定 query={query!r} "
        f"convs_lines={len((convs or '').splitlines())}"
    )
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
        raw = triage_agent.generate_openai_response(prompt)
        data = _safe_load_json(raw)
        intent = str(data.get("intent", "【其他】")).strip()
        normalized_intent = "【其他】"
        if intent in ("【专业解释】", "【判断】", "【其他】"):
            normalized_intent = intent
        result = {
            "intent": normalized_intent,
            "query": clean_query_text(str(data.get("query", ""))),
            "reason": str(data.get("reason", "")).strip(),
        }
        elapsed = time.time() - start_time
        print(f"[DEBUG][periodic] 判定结束 elapsed={elapsed:.2f}s result={result}")
        return result
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
        if _is_smalltalk_message(message_text):
            print(f"[DEBUG][proactive] 跳过寒暄消息: user={user_name!r} text={message_text!r}")
            return

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
                _queue_auto_prompt(
                    client=client,
                    channel_name=channel_name,
                    channel_id=channel_id,
                    user_id=user_id,
                    user_name=user_name,
                    kind="explain",
                    term=terms[0] if terms else "",
                    query=clean_query_text(message_text),
                    reason="proactive_confusion",
                    trigger_ts=ts,
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
                    _queue_auto_prompt(
                        client=client,
                        channel_name=channel_name,
                        channel_id=channel_id,
                        user_id=user_id,
                        user_name=user_name,
                        kind="judgment",
                        term="",
                        query=plan.get("search_query") or clean_query_text(message_text),
                        reason="proactive_judgment",
                        trigger_ts=ts,
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
                    periodic_terms = extract_candidate_terms(triage["query"])
                    _queue_auto_prompt(
                        client=client,
                        channel_name=channel_name,
                        channel_id=channel_id,
                        user_id=user_id,
                        user_name=user_name,
                        kind="explain",
                        term=periodic_terms[0] if periodic_terms else "",
                        query=triage["query"],
                        reason=triage.get("reason") or "periodic_explain",
                        trigger_ts=ts,
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
                    _queue_auto_prompt(
                        client=client,
                        channel_name=channel_name,
                        channel_id=channel_id,
                        user_id=user_id,
                        user_name=user_name,
                        kind="judgment",
                        term="",
                        query=plan.get("search_query") or triage["query"],
                        reason=triage.get("reason") or "periodic_judgment",
                        trigger_ts=ts,
                    )
    except Exception as e:
        print(f"[DEBUG][proactive] 主动/巡检流程异常，已降级跳过: {e}")


def _pick_timer_target_user(active_users: list[dict]) -> tuple[str, str]:
    if not active_users:
        return "", ""
    row = active_users[0]
    uid = str(row.get("user_id") or "").strip()
    uname = str(row.get("user_name") or uid).strip()
    return uid, (uname or uid)


def _record_mm_channel_activity(channel_id: str, user_id: str, now_ts: float) -> None:
    cid = str(channel_id or "").strip()
    uid = str(user_id or "").strip()
    if not cid or not uid:
        return
    _MM_ACTIVE_CHANNEL_LAST_TS[cid] = float(now_ts)
    users = _MM_ACTIVE_CHANNEL_USERS.get(cid) or {}
    users[uid] = float(now_ts)
    _MM_ACTIVE_CHANNEL_USERS[cid] = users


def _is_archived_error(exc: Exception) -> bool:
    return "is_archived" in str(exc).lower()


def _cleanup_mm_activity(now_ts: float) -> None:
    expire_before = float(now_ts) - float(MM_ACTIVE_CHANNEL_WINDOW_SECONDS)
    stale_channels = [
        cid for cid, ts in _MM_ACTIVE_CHANNEL_LAST_TS.items()
        if float(ts) < expire_before
    ]
    for cid in stale_channels:
        _MM_ACTIVE_CHANNEL_LAST_TS.pop(cid, None)
        _MM_ACTIVE_CHANNEL_USERS.pop(cid, None)


def _review_solving_terms_for_user(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
) -> None:
    imm = mental_model_memory.get_imm(user_id=user_id, user_name=user_name)
    unknowns = list((imm or {}).get("认知盲区 (未涉及知识)") or [])
    if not unknowns:
        return

    review_terms: list[tuple[str, int]] = []
    for row in unknowns:
        if not isinstance(row, dict):
            continue
        term = str(row.get("未知术语") or "").strip()
        status = str(row.get("当前状态") or "").strip()
        try:
            duration = int(float(row.get("持续时长_秒") or 0))
        except Exception:
            duration = 0
        if term and status == "解决中" and duration >= MM_TERM_SOLVING_REVIEW_SECONDS:
            review_terms.append((term, duration))

    if not review_terms:
        return

    convs = get_conversation_history(
        client=client,
        channel_id=channel_id,
        bot_id=BOT_ID,
        user_id2names=user_id2names,
        ts=str(time.time()),
        limit=40,
    )
    recent_texts = _extract_recent_user_texts(convs, lookback_lines=14)

    for term, duration in review_terms:
        norm_term = _normalize_term(term)
        mention_any = False
        mention_confusion = False
        mention_understood = False

        for text in recent_texts:
            txt = str(text or "")
            txt_norm = _normalize_term(txt)
            has_term = bool(norm_term and (norm_term in txt_norm or norm_term in txt.lower()))
            if not has_term:
                continue
            mention_any = True
            if has_confusion_cue(txt):
                mention_confusion = True
            if _has_understood_cue(txt):
                mention_understood = True

        if mention_confusion:
            mental_model_memory.update_unknown_term_status(
                user_id=user_id,
                user_name=user_name,
                term=term,
                status="未解决",
                note=f"解决中超时({duration}s)且仍有困惑，回退未解决",
                reset_timer=True,
            )
            print(
                f"[DEBUG][mm_timer] solving_term回退未解决 channel={channel_id!r} "
                f"user={user_name!r} term={term!r} duration={duration}s"
            )
            continue

        if mention_understood or (not mention_any):
            mental_model_memory.update_unknown_term_status(
                user_id=user_id,
                user_name=user_name,
                term=term,
                status="已解决",
                note=(
                    f"解决中超时({duration}s)后用户确认已理解"
                    if mention_understood else
                    f"解决中超时({duration}s)且近期未再讨论该术语"
                ),
                reset_timer=False,
            )
            _mark_term_known(channel_id=channel_id, user_id=user_id, user_name=user_name, term=term)
            print(
                f"[DEBUG][mm_timer] solving_term收敛已解决 channel={channel_id!r} "
                f"user={user_name!r} term={term!r} duration={duration}s"
            )


def _dispatch_timer_decision(
    client,
    channel_id: str,
    channel_name: str,
    user_id: str,
    user_name: str,
    decision: dict,
) -> None:
    response_type = str((decision or {}).get("response_type") or "none").strip().lower()
    response_query = clean_query_text(str((decision or {}).get("query") or ""))
    response_reason = str((decision or {}).get("reason") or "").strip().lower()
    if response_type == "none" or not response_query:
        return

    now_ts = time.time()
    term_key = f"timer:{response_reason}:{response_type}:{response_query[:40]}"
    cooldown = TERM_COOLDOWN_SECONDS if response_type == "professional_explain" else JUDGMENT_COOLDOWN_SECONDS
    if not _should_trigger_proactive(
        channel_name=channel_name,
        target_user_id=user_id,
        term_key=term_key,
        now_ts=now_ts,
        cooldown_seconds=cooldown,
    ):
        return

    ts = str(now_ts)
    if response_type in {"topic", "division"}:
        intent_label = "【选题】" if response_type == "topic" else "【分工】"
        convs = get_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=INTENT_CONTEXT_MESSAGE_LIMIT,
        )
        _dispatch_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=response_query,
            convs=convs,
            intent_label=intent_label,
            intent_time=0.0,
        )
        print(
            f"[DEBUG][mm_timer] 已触发 {response_type} channel={channel_id!r} "
            f"user={user_name!r} reason={response_reason!r}"
        )
        return

    if response_type in {"professional_explain", "judgment"}:
        intent_label = "【专业解释】" if response_type == "professional_explain" else "【判断】"
        convs = get_conversation_history(
            client=client,
            channel_id=channel_id,
            bot_id=BOT_ID,
            user_id2names=user_id2names,
            ts=ts,
            limit=INTENT_CONTEXT_MESSAGE_LIMIT,
        )
        _run_retrieval_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=response_query,
            search_query=response_query,
            convs=convs,
            intent_label=intent_label,
            intent_time=0.0,
            rewrite_time=0.0,
            rewrite_thought=("imm_timer" if intent_label == "【专业解释】" else ""),
        )
        if response_type == "professional_explain":
            # 计时器触发解释后，将该术语置为“解决中”，并开启按术语独立 followup。
            mental_model_memory.update_unknown_term_status(
                user_id=user_id,
                user_name=user_name,
                term=response_query,
                status="解决中",
                note="计时器触发LLM解释，进入观察阶段",
                reset_timer=True,
            )
            _start_explain_followup_worker(
                client=client,
                channel_id=channel_id,
                channel_name=channel_name,
                user_id=user_id,
                user_name=user_name,
                term=response_query,
                trigger_ts=float(now_ts),
            )
        print(
            f"[DEBUG][mm_timer] 已触发 {response_type} channel={channel_id!r} "
            f"user={user_name!r} reason={response_reason!r}"
        )


def _run_mm_timer_tick(client) -> None:
    now_ts = time.time()
    _cleanup_mm_activity(now_ts)

    active_channel_ids = [
        cid for cid, last_ts in _MM_ACTIVE_CHANNEL_LAST_TS.items()
        if (now_ts - float(last_ts)) <= MM_ACTIVE_CHANNEL_WINDOW_SECONDS
    ]
    if not active_channel_ids:
        return

    known_user_map: dict[str, str] = {}
    for row in (mental_model_memory.list_known_users() or []):
        uid = str(row.get("user_id") or "").strip()
        if not uid or uid == BOT_ID:
            continue
        known_user_map[uid] = str(row.get("user_name") or uid).strip() or uid

    for channel_id in active_channel_ids:
        if channel_id in _MM_ARCHIVED_CHANNELS:
            continue
        channel_name = channel_id2names.get(channel_id, channel_id)
        try:
            active_users_map = _MM_ACTIVE_CHANNEL_USERS.get(channel_id) or {}
            active_users = [
                {"user_id": uid, "user_name": known_user_map.get(uid, uid)}
                for uid, ts in active_users_map.items()
                if (now_ts - float(ts)) <= MM_ACTIVE_CHANNEL_WINDOW_SECONDS and uid in known_user_map
            ]
            if not active_users:
                continue

            # 1) 用户级：IMM 未解决术语超时。
            for row in active_users:
                uid = str(row.get("user_id") or "").strip()
                uname = str(row.get("user_name") or uid).strip()

                # 1.1) 解决中术语超时复核：根据近期对话收敛到 已解决/未解决。
                _review_solving_terms_for_user(
                    client=client,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=uid,
                    user_name=uname,
                )

                result = mental_model_memory.evaluate_timer_proactive(
                    channel_id=channel_id,
                    user_id=uid,
                    user_name=uname,
                    include_user_term_rule=True,
                    include_channel_rules=False,
                )
                decision = (result or {}).get("decision") or {}
                if bool(decision.get("should_respond")):
                    _dispatch_timer_decision(
                        client=client,
                        channel_id=channel_id,
                        channel_name=channel_name,
                        user_id=uid,
                        user_name=uname,
                        decision=decision,
                    )

            # 2) 频道级：冲突超时、选题/分工阶段超时（频道维度一次触发）。
            target_uid, target_name = _pick_timer_target_user(active_users)
            if not target_uid:
                continue
            channel_result = mental_model_memory.evaluate_timer_proactive(
                channel_id=channel_id,
                user_id=target_uid,
                user_name=target_name,
                include_user_term_rule=False,
                include_channel_rules=True,
            )
            channel_decision = (channel_result or {}).get("decision") or {}
            if bool(channel_decision.get("should_respond")):
                _dispatch_timer_decision(
                    client=client,
                    channel_id=channel_id,
                    channel_name=channel_name,
                    user_id=target_uid,
                    user_name=target_name,
                    decision=channel_decision,
                )
        except Exception as e:
            if _is_archived_error(e):
                _MM_ARCHIVED_CHANNELS.add(channel_id)
                print(f"[DEBUG][mm_timer] 跳过归档频道 channel={channel_id!r} reason=is_archived")
                continue
            raise


def _ensure_mm_timer_worker(client) -> None:
    global _mm_timer_worker_thread
    if MM_TIMER_POLL_SECONDS <= 0:
        print("[DEBUG][mm_timer] MM_TIMER_POLL_SECONDS<=0，已禁用后台计时巡检")
        return

    if _mm_timer_worker_thread and _mm_timer_worker_thread.is_alive():
        return

    def _worker() -> None:
        print(f"[DEBUG][mm_timer] 后台计时巡检已启动 poll={MM_TIMER_POLL_SECONDS}s")
        while True:
            try:
                _run_mm_timer_tick(client=client)
            except Exception as e:
                print(f"[DEBUG][mm_timer] 巡检异常，已跳过本轮: {e}")
            time.sleep(max(1, MM_TIMER_POLL_SECONDS))

    _mm_timer_worker_thread = threading.Thread(target=_worker, daemon=True)
    _mm_timer_worker_thread.start()

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


@app.action("auto_prompt_accept")
def on_auto_prompt_accept(ack, body, client):
    ack()
    # 已取消二次确认，保留 action handler 仅为兼容历史卡片。
    return


@app.action("auto_prompt_decline")
def on_auto_prompt_decline(ack, body, client):
    ack()
    # 已取消二次确认，保留 action handler 仅为兼容历史卡片。
    return


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

    # 已关闭入频道打招呼。


def _process_message_task(client, task: _MessageTask) -> None:
    channel_id = task.channel_id
    channel_name = task.channel_name
    user_id = task.user_id
    user_name = task.user_name
    ts = task.ts
    raw_text = task.raw_text
    user_utterance = task.user_utterance
    mentioned_bot = task.mentioned_bot
    query = task.query
    event_type = task.event_type

    legacy_prefix = settings.slack_legacy_mention_prefix

    # 非@寒暄消息：静默跳过（不触发心智更新与检索）。
    if not mentioned_bot and not user_utterance.startswith(legacy_prefix) and event_type != "app_mention":
        if _is_smalltalk_message(user_utterance):
            print(f"[DEBUG][handle_message] 非@寒暄消息，静默跳过: user={user_name!r} text={user_utterance!r}")
            return

    # 每条消息至少一次 LLM：先分析并更新 IMM/SMM。
    mm_convs = get_conversation_history(
        client=client,
        channel_id=channel_id,
        bot_id=BOT_ID,
        user_id2names=user_id2names,
        ts=ts,
        limit=max(10, MM_UPDATE_RECENT_ROUNDS * 2),
    )
    mm_result = mental_model_memory.analyze_and_update(
        agent=agent,
        channel_id=channel_id,
        user_id=user_id,
        user_name=user_name,
        message_text=user_utterance,
        convs=mm_convs,
        enable_response_decision=bool(
            mentioned_bot or user_utterance.startswith(legacy_prefix) or event_type == "app_mention"
        ),
    )
    mm_decision = (mm_result or {}).get("decision") or {}
    mm_smm = (mm_result or {}).get("smm") or {}
    mm_smm_transition = (mm_result or {}).get("smm_transition") or {}
    mm_smm_life = mm_smm.get("任务生命周期") if isinstance(mm_smm.get("任务生命周期"), dict) else {}
    smm_phase = str(mm_smm_life.get("当前所处阶段") or mm_smm.get("current_phase") or "")
    smm_phase_status = str(mm_smm.get("phase_status") or "")
    if smm_phase and smm_phase_status:
        print(f"[DEBUG][mental_model] phase={smm_phase!r} status={smm_phase_status!r} decision={mm_decision}")

    if not mentioned_bot and not user_utterance.startswith(legacy_prefix) and event_type != "app_mention":
        if _is_smalltalk_message(user_utterance):
            if bool(mm_decision.get("should_respond")):
                print(
                    f"[DEBUG][handle_message] 寒暄消息拦截自动回复: "
                    f"decision={mm_decision}"
                )
            return

        if bool(mm_smm_transition.get("changed")):
            to_stage = str(mm_smm_transition.get("to_phase") or "")
            from_stage = str(mm_smm_transition.get("from_phase") or "")
            send_status_message(
                client,
                channel_id,
                user_id,
                f"项目阶段已从【{from_stage}】推进到【{to_stage}】。",
            )

        profile_related = _looks_like_profile_intro(user_utterance)

        # 非@消息仅用于更新 IMM/SMM，不直接触发自动回复；主动介入统一交由后台计时器巡检。
        if bool(mm_decision.get("should_respond")):
            print(
                "[DEBUG][handle_message] 非@即时回复已关闭，等待计时器触发: "
                f"decision={mm_decision}"
            )

        # 无需即时回应时，不再重复跑旧的自动确认逻辑。
        if profile_related:
            # 仅在当前消息包含画像线索时触发监听。
            # 是否真正有画像增量由 watcher 内部判定；无增量/冷却命中时不再提前提示。
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

        # 计时器与对话解耦：非@消息不再即时触发主动机制。
        # 主动介入统一由后台 MM_TIMER worker 负责。
        return

    # @ 消息统一两阶段：
    # 1) 已完成：mental_model_memory.analyze_and_update（LLM）
    # 2) 这里执行一次基于 @内容 + IMM 的直接回复（不再走旧意图链路）
    if mentioned_bot or user_utterance.startswith(legacy_prefix) or event_type == "app_mention":
        if bool(mm_smm_transition.get("changed")):
            to_stage = str(mm_smm_transition.get("to_phase") or "")
            from_stage = str(mm_smm_transition.get("from_phase") or "")
            send_status_message(
                client,
                channel_id,
                user_id,
                f"项目阶段已从【{from_stage}】推进到【{to_stage}】。",
            )

        if not query:
            send_status_message(
                client,
                channel_id,
                user_id,
                "请直接描述你的需求，例如：\n"
                "• `@我 帮我推荐选题`\n"
                "• `@我 帮我规划分工`\n"
                "• `@我 总结一下我们的讨论`"
            )
            return

        # @路径恢复七意图识别与分发，不再被三类型 response_type 限制。
        intent_label = _classify_full_intent(
            query=query,
            convs=mm_convs,
            channel_id=channel_id,
            user_id=user_id,
            user_name=user_name,
        )
        # @分工请求：按策略直接转为总结型回复（含后续行动建议）。
        if intent_label == "【分工】":
            print("[DEBUG][handle_message] @分工请求按策略转为【总结】")
            intent_label = "【总结】"
        _dispatch_intent(
            client=client,
            channel_id=channel_id,
            channel_name=channel_name,
            user_id=user_id,
            user_name=user_name,
            ts=ts,
            query=query,
            convs=mm_convs,
            intent_label=intent_label,
            intent_time=0.0,
        )
        return

    # 以上两个分支已覆盖 message 与 app_mention，旧意图链路已移除。
    return

@app.event("app_mention")
@app.event("message")
def handle_message_event(ack, event, client):
    ack()

    channel_id, ts = event.get("channel"), event.get("ts")
    user_id = event.get("user")
    event_type = (event.get("type") or "").strip()
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

    channel_name = channel_id2names.get(channel_id, channel_id)

    if user_id == BOT_ID:
        return

    # 同一条消息可能同时触发 app_mention 和 message 事件，这里按 ts 去重。
    if _is_duplicate_message_event(channel_id, user_id, ts, event_type=event_type):
        return

    if channel_id not in _CHANNEL_INFO_SYNCED:
        register_channel_display_name(client, channel_id, SQL_PASSWORD)
        _CHANNEL_INFO_SYNCED.add(channel_id)

    user_name = resolve_user_name(client, user_id, user_id2names, SQL_PASSWORD)
    _record_mm_channel_activity(channel_id=channel_id, user_id=user_id, now_ts=time.time())
    raw_text = (event.get("text") or "").strip("\n").strip()
    user_utterance = replace_utterance_ids(raw_text, id2names=user_id2names)

    files = event.get("files") or []
    if files:
        file_desc = []
        for f in files:
            name = str((f or {}).get("name") or "")
            mimetype = str((f or {}).get("mimetype") or "")
            file_desc.append(f"{name}({mimetype})")
        print(
            f"[DEBUG][handle_message] 检测到文件附件 subtype={subtype!r} "
            f"count={len(files)} files={file_desc}"
        )

        # 图片消息常出现 text 为空，补充视觉识别结果供后续心智分析使用。
        vision_text = agent.describe_images_from_slack_files(files=files, slack_bot_token=SLACK_BOT_TOKEN)
        if vision_text:
            if user_utterance:
                user_utterance += "\n"
            user_utterance += f"[图片识别] {vision_text}"
            print(f"[DEBUG][handle_message] 已注入图片识别文本 chars={len(vision_text)}")
        elif not user_utterance:
            # 图片无OCR结果时保底填充，避免 message_text 为空导致心智分析无效。
            user_utterance = "[用户发送了图片/附件，暂未识别到可用文本]"
            print("[DEBUG][handle_message] 图片附件未识别到文本，使用保底提示")

    print(f"\n{'='*60}")
    print(f"[DEBUG][handle_message] user={user_name!r} text={user_utterance!r}")

    memory.create_table_if_not_exists(table_name=channel_name)
    memory.write_into_memory(
        table_name=channel_name,
        utterance_info={
            "speaker": user_name,
            "utterance": user_utterance,
            "convs": "",
            "query": "",
            "rewrite_query": "",
            "rewrite_thought": "",
            "clarify": "",
            "clarify_thought": "",
            "clarify_cnt": 0,
            "search_results": "",
            "infer_time": "",
            "reply_timestamp": "",
            "reply_user": "",
            "timestamp": ts,
        },
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

    task = _MessageTask(
        channel_id=channel_id,
        channel_name=channel_name,
        user_id=user_id,
        user_name=user_name,
        ts=ts,
        event_type=event_type,
        raw_text=raw_text,
        user_utterance=user_utterance,
        query=query,
        mentioned_bot=mentioned_bot,
    )
    _enqueue_message_task(client=client, task=task)
    queue_depth = _get_message_task_queue(channel_id, user_id).qsize()
    print(
        f"[DEBUG][handle_message] 已异步入队 channel={channel_id!r} user={user_id!r} "
        f"ts={ts!r} queue_depth={queue_depth}"
    )
    return


try:
    _ensure_mm_timer_worker(app.client)
except Exception as e:
    print(f"[DEBUG][mm_timer] 启动失败，已降级跳过: {e}")


if __name__ == "__main__":
    _ensure_mm_timer_worker(app.client)
    SocketModeHandler(app, SLACK_APP_TOKEN).start()