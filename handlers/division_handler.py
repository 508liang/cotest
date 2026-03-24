"""
handlers/division_handler.py

修改记录：
  1. [画像污染] DivisionContext 新增 user_only_convs 字段；handle_division_intent
     直接使用 ctx.user_only_convs 而非再次调 get_user_only_conversation_history，
     与 app 层保持来源唯一性。
  2. [Debug]   全链路添加 [DEBUG][division_handler] 前缀日志，覆盖搜索、画像、分工
               生成各阶段的详细状态。
  3. [内容质量] 搜索词从2条扩展到5条，覆盖"研究方法论 / 实证流程 / 典型案例 /
               跨学科合作 / 关键词精准检索"五个维度；参考文献格式改为
               "标题+摘要+来源"；refs上限从5条提到8条；
               enriched_convs 中以结构化方式注入选题+参考资料。
"""

import time
from dataclasses import dataclass, field
from typing import Any

from memory.imm_profile_store import ImmProfileStore
from utils import resolve_user_name
from handlers.profile_utils import (
    extract_latest_topic,
    draft_profiles_from_convs,
    profiles_have_changed,
    merge_profile_with_existing,
    notify_profile_update_if_changed,   # ⬅ 新增
)


# ── DataClass ────────────────────────────────────────────────────────────────

@dataclass
class DivisionContext:
    client: Any
    channel_id: str
    channel_name: str
    user_id: str
    user_name: str
    ts: str
    query: str
    convs: str          # 含Bot回复的完整对话
    intent_time: float
    agent: Any
    memory: Any
    profile_memory: Any
    bot_id: str
    sql_password: str
    user_id2names: dict = field(default_factory=dict)  # ★ 补充缺失字段
    # ⬇ 新增：纯用户对话（不含Bot回复），专用于画像提炼
    user_only_convs: str = field(default="")
    is_new_channel: bool = field(default=False)   # ⬅ 新增
    # ★ 新增：当前频道实际发言的用户名列表
    channel_speakers: list = field(default_factory=list)
    active_user_ids: list = None   # ⬅ 新增


# ── 主入口 ────────────────────────────────────────────────────────────────────

def handle_division_intent(ctx: DivisionContext):
    from handlers.profile_confirm import send_profile_confirm_card
    from memory.pending_intent_memory import PendingIntentMemory

    print(f'[DEBUG][division_handler] ── handle_division_intent 开始 ──')
    print(f'[DEBUG][division_handler] user={ctx.user_name!r} channel={ctx.channel_name!r}')
    print(f'[DEBUG][division_handler] query={ctx.query!r}')

    pending_memory = PendingIntentMemory(sql_password=ctx.sql_password)
    pending_memory.create_table_if_not_exists()

    # ── Step 1: 获取纯用户对话（画像唯一来源）────────────────────────────────
    user_only = ctx.user_only_convs
    if not user_only:
        from utils import get_user_only_conversation_history
        print(f'[DEBUG][division_handler] ⚠ ctx.user_only_convs 为空，降级调用 API（不推荐）')
        user_only = get_user_only_conversation_history(
            client=ctx.client, channel_id=ctx.channel_id,
            bot_id=ctx.bot_id, user_id2names=ctx.user_id2names, ts=ctx.ts, limit=50
        )

    profile_source = f'{user_only}\n{ctx.query}'
    print(f'[DEBUG][division_handler] 画像提炼输入行数={len(profile_source.splitlines())} '
          f'前200字:\n{profile_source[:200]}')

    # ── 严格取当前频道参与的非Bot用户 ────────────────────────────────────────
    allowed_ids = [
        uid for uid in (ctx.active_user_ids or [])
        if uid and uid not in (ctx.bot_id, 'bot_id')
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
    print(f'[DEBUG][division_handler] 当前频道参与的非Bot用户: {[(u, n) for u, n in active_users]}')

    # ── Step 2: 提炼画像草稿 ──────────────────────────────────────────────────
    drafts = draft_profiles_from_convs(ctx.agent, profile_source, active_users)
    print(f'[DEBUG][division_handler] 提炼草稿 {len(drafts)} 份')

    # ── Step 3: 检测当前用户的画像是否有增量变化 ─────────────────────────────
    existing = ctx.profile_memory.load(ctx.user_id, ctx.channel_id)
    my_draft = next((p for p in drafts if p.get('user_id') == ctx.user_id), None)

    # ★ 核心变更：如果当前用户画像有增量，先挂起意图，让用户确认后再执行
    if my_draft and profiles_have_changed(existing, my_draft):
        merged_profile = merge_profile_with_existing(existing, my_draft)
        print(f'[DEBUG][division_handler] ★ 检测到当前用户画像有增量，先挂起意图等待确认')
        print(f'[DEBUG][division_handler]   existing.major={existing.get("major") if existing else None!r}')
        print(f'[DEBUG][division_handler]   merged.major={merged_profile.get("major")!r}')

        # 保存挂起意图
        pending_memory.save(
            user_id=ctx.user_id,
            channel_id=ctx.channel_id,
            intent_label='【分工】',
            payload={
                'channel_id': ctx.channel_id, 'channel_name': ctx.channel_name,
                'user_name': ctx.user_name, 'ts': ctx.ts,
                'query': ctx.query, 'convs': ctx.convs, 'intent_time': ctx.intent_time,
                'bot_id': ctx.bot_id, 'agent': '__global__', 'memory': '__global__',
                'user_only_convs': user_only,
                'active_user_ids': ctx.active_user_ids or [],
            }
        )

        # 发送画像确认卡片
        reason = '方向变更' if (
            existing and existing.get('major') and merged_profile.get('major')
            and existing.get('major') != merged_profile.get('major')
        ) else '画像更新'
        send_profile_confirm_card(ctx.client, ctx.channel_id, ctx.user_id,
                                   merged_profile, reason=reason)
        print(f'[DEBUG][division_handler] 已发送画像确认卡片，等待用户确认后继续执行分工')
        return

    # ── Step 4: 其他用户的画像增量 → 静默通知（不阻断当前用户流程）─────────────
    for d in drafts:
        uid = d.get('user_id')
        if not uid or uid == ctx.user_id:
            continue
        existing_d = ctx.profile_memory.load(uid, ctx.channel_id)
        notified = notify_profile_update_if_changed(
            client=ctx.client,
            channel_id=ctx.channel_id,
            user_id=uid,
            existing=existing_d,
            new_draft=d,
            profile_memory=ctx.profile_memory,
        )
        print(f'[DEBUG][division_handler] 其他用户画像通知 [{d.get("user_name")}]: notified={notified}')

    # ── Step 5: 组装最新画像，执行分工 ────────────────────────────────────────
    all_profiles = []
    for uid, uname in active_users:
        p = ctx.profile_memory.load(uid, ctx.channel_id)
        if not p:
            p = next((d for d in drafts if d.get('user_id') == uid), None)
        if p and (p.get('major') or p.get('methodology') or p.get('research_interests')):
            all_profiles.append(p)
            print(f'[DEBUG][division_handler] 纳入画像 [{uname}]:'
                  f' major={p.get("major")!r}'
                  f' methods={p.get("methodology")}'
                  f' interests={p.get("research_interests")}')
        else:
            print(f'[DEBUG][division_handler] 跳过画像 [{uname}]（无实质内容）')

    print(f'[DEBUG][division_handler] 执行分工，画像数量={len(all_profiles)}')
    _execute_division(ctx, profiles=all_profiles, fallback_convs=profile_source)


# ── 分工执行（RAG 核心流程）─────────────────────────────────────────────────────

def _execute_division(ctx: DivisionContext, profiles: list, fallback_convs: str = ''):
    from utils import send_answer, send_status_message, delete_status_message, slack_chat_update

    print(f'[DEBUG][division_handler] ── _execute_division 开始 ──')

    # ── 组装画像描述 ──────────────────────────────────────────────────────────
    if profiles:
        user_profiles_text = ImmProfileStore.format_for_prompt(profiles)
    elif fallback_convs:
        user_profiles_text = (
            '以下为用户的对话内容，请据此推断各用户的学科背景和研究方向：\n' + fallback_convs
        )
        print(f'[DEBUG][division_handler] 结构化画像为空，使用对话内容兜底')
    else:
        user_profiles_text = f'当前提问者：{ctx.user_name}。可结合对话内容推断学科背景。'

    print(f'[DEBUG][division_handler] user_profiles_text:\n{user_profiles_text[:300]}')

    # ── 提取最新选题：只从用户发言（非Bot）中识别明确确认的选题 ────────────────
    source_for_topic = ctx.user_only_convs if ctx.user_only_convs else ctx.convs
    latest_topic = extract_latest_topic(source_for_topic)
    print(f'[DEBUG][division_handler] extract_latest_topic 结果: {latest_topic!r}')

    if not latest_topic:
        print(f'[DEBUG][division_handler] ⚠ 未检测到已确认选题，发送提示并返回')
        send_answer(
            client=ctx.client, channel_id=ctx.channel_id, user_id=ctx.user_id,
            answer=(
                '📋 当前对话中未检测到已明确确认的研究选题。\n\n'
                '请在对话中用明确的表述确认选题，例如：\n'
                '• "我们的选题是：XXX"\n'
                '• "确定选题为 XXX"\n'
                '• "选题定为 XXX"\n\n'
                '或者先使用 `@coresearchagent_kx 帮我推荐选题` 获取选题建议，'
                '确认后再发起分工规划 😊'
            ),
        )
        return

    print(f'[DEBUG][division_handler] 检测到最新选题: {latest_topic!r}')

    # ★ 识别到选题后，立即展示给用户，同时继续后续流程（不阻断）
    try:
        ctx.client.chat_postMessage(
            channel=ctx.channel_id,
            text=(
                f'<@{ctx.user_id}> ✅ 已识别到确认选题：\n'
                f'> 📌 *{latest_topic}*\n\n'
                f'⏳ 正在基于选题与团队画像生成分工方案，请稍候…'
            ),
        )
        print(f'[DEBUG][division_handler] 已向用户展示识别到的选题: {latest_topic!r}')
    except Exception as e:
        print(f'[DEBUG][division_handler] ⚠ 展示选题消息失败: {e}')

    # ── 分工不使用 RAG：仅基于选题与团队画像生成 ─────────────────────────────
    refs_text = '（分工功能不使用外部检索资料，请仅基于选题与团队成员画像制定方案）'
    search_elapsed = 0.0
    raw_results: list = []

    # ── topic_context 只使用用户纯对话+query，不混入Bot回复 ─────────────────
    topic_context = (
        f'【当前确认选题】\n{latest_topic}\n\n'
        f'【用户对话背景（不含Bot回复）】\n'
        f'{(ctx.user_only_convs or ctx.convs)[-2000:]}'
    )

    # ── 流式生成分工方案 ────────────────────────────────────────────────────────
    division, division_time, response_ts = ctx.agent.propose_division_stream(
        query=ctx.query,
        convs=topic_context,
        user_profiles=user_profiles_text,
        refs=refs_text,
        client_slack=ctx.client,
        channel_id=ctx.channel_id,
        user_id=ctx.user_id,
        search_status_lines=None,
        search_results=None,
    )
    print(f'[DEBUG][division_handler] 分工生成完成({division_time:.2f}s)')

    # ── 写库 ──────────────────────────────────────────────────────────────────
    ctx.memory.write_into_memory(
        table_name=ctx.channel_name,
        utterance_info={
            'speaker':         'CoSearchAgent',
            'utterance':       division,
            'convs':           ctx.convs,
            'query':           ctx.query,
            'rewrite_query':   ctx.query,
            'rewrite_thought': '',
            'clarify':         '',
            'clarify_thought': '',
            'clarify_cnt':     0,
            'search_results':  str(raw_results[:8]),
            'infer_time':      str({
                'intent':   ctx.intent_time,
                'search':   search_elapsed,
                'division': division_time,
            }),
            'reply_timestamp': ctx.ts,
            'reply_user':      ctx.user_name,
            'timestamp':       response_ts,
        },
    )

    print(f'[DEBUG][division_handler] ── _execute_division 完成 ──')