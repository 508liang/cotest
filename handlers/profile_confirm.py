"""
handlers/profile_confirm.py

修改记录：
  1. [画像写库时机] 画像只在用户点击"确认"或 Modal 提交后写库，
     profile_watcher / notify_profile_update_if_changed 不再静默保存。
  2. [确认卡片统一] send_profile_update_notify 废弃，统一使用
     send_profile_confirm_card 发送带"确认/修改"按钮的卡片。
  3. [频道参与用户] _resume_pending_intent 中只使用当前频道参与用户的画像，
     通过 pending payload 里存储的 active_user_ids 过滤。
  4. [新用户初始化] handle_profile_confirm 中若用户画像不存在则初始化空记录。
"""

import json
import time
from typing import Any

from utils import resolve_user_name, slack_chat_update


# ── 发送画像确认卡片（统一入口） ──────────────────────────────────────────────

def send_profile_confirm_card(client, channel_id: str, user_id: str,
                               draft_profile: dict,
                               reason: str = "首次录入") -> str:
    """
    发送画像草稿确认卡片，返回消息 ts。
    卡片展示：专业 / 研究兴趣 / 擅长方法 / 关键词
    按钮：✅ 确认  |  ✏️ 修改

    reason 影响卡片提示文案：
      "首次录入" / "新频道" / "首次问候" / "画像更新" / "方向变更"
    """
    major     = draft_profile.get("major", "未识别")
    interests = "、".join(draft_profile.get("research_interests") or []) or "暂无"
    methods   = "、".join(draft_profile.get("methodology") or []) or "暂无"
    keywords  = "、".join(draft_profile.get("keywords") or []) or "暂无"
    uname     = draft_profile.get("user_name", "您")

    if reason == "新频道":
        header_text = (
            f"👋 <@{user_id}> 欢迎来到新频道！\n"
            f"我保存了您之前的学术背景，请确认在此研究场景下是否仍然适用："
        )
    elif reason == "首次问候":
        header_text = (
            f"👋 <@{user_id}> 欢迎使用 CoSearchAgent！\n"
            f"这是您上次保存的学术背景，请确认是否仍然准确："
        )
    elif reason == "方向变更":
        header_text = (
            f"🔄 <@{user_id}> 检测到您的研究方向可能已变更，请先确认更新后的画像：\n"
            f"_确认后将继续为您生成结果_"
        )
    elif reason == "画像更新":
        header_text = (
            f"🔔 <@{user_id}> 检测到您的学术背景有新内容，请先确认是否更新画像：\n"
            f"_确认后将继续为您生成结果_"
        )
    elif reason.startswith("第") and "次使用" in reason:
        header_text = (
            f"📋 <@{user_id}> 您已使用 {reason.replace('次使用', '')} 次，"
            f"帮您确认一下学术背景是否有更新："
        )
    else:
        header_text = (
            f"👤 <@{user_id}> 我根据对话提炼了您的学术背景，请确认是否准确："
        )

    print(f"[DEBUG][profile_confirm] 发送确认卡片 reason={reason!r} → user={uname!r}")

    # ★ 第一步：用占位 ts 发送卡片
    placeholder_profile = {
        **draft_profile,
        "_target_user_id": user_id,
        "_channel_id": channel_id,
        "_notify_ts": "__PENDING__",
    }
    placeholder_json = json.dumps(placeholder_profile, ensure_ascii=False)

    blocks = _build_confirm_card_blocks(
        header_text, major, interests, methods, keywords, placeholder_json
    )

    resp = client.chat_postMessage(
        channel=channel_id,
        text=(
            f"用户画像待确认：专业={major}；研究兴趣={interests}；"
            f"擅长方法={methods}；关键词={keywords}"
        ),
        blocks=blocks,
    )
    ts = resp["ts"]

    # ★ 第二步：把真实 ts 写入按钮 value，更新卡片
    real_profile = {
        **draft_profile,
        "_target_user_id": user_id,
        "_channel_id": channel_id,
        "_notify_ts": ts,
    }
    real_json = json.dumps(real_profile, ensure_ascii=False)
    updated_blocks = _build_confirm_card_blocks(
        header_text, major, interests, methods, keywords, real_json
    )
    try:
        slack_chat_update(
            client=client,
            channel=channel_id,
            ts=ts,
            blocks=updated_blocks,
        )
    except Exception as e:
        print(f"[DEBUG][profile_confirm] ⚠ 更新卡片value失败: {e}")

    print(f"[DEBUG][profile_confirm] 确认卡片已发送 ts={ts}")
    return ts


def _build_confirm_card_blocks(header_text: str, major: str, interests: str,
                                methods: str, keywords: str, profile_json: str) -> list:
    """构建确认卡片的 blocks 结构。"""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{header_text}\n\n"
                    f"*专业：* {major}\n"
                    f"*研究兴趣：* {interests}\n"
                    f"*擅长方法：* {methods}\n"
                    f"*研究关键词：* {keywords}"
                )
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✅ 确认"},
                    "style": "primary",
                    "action_id": "profile_confirm",
                    "value": profile_json,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "✏️ 修改"},
                    "action_id": "profile_edit",
                    "value": profile_json,
                },
            ]
        }
    ]


# ── send_profile_update_notify 统一转发到 send_profile_confirm_card ───────────

def send_profile_update_notify(client, channel_id: str, user_id: str,
                                updated_profile: dict,
                                direction_changed: bool = False) -> str:
    """
    兼容旧调用入口，统一使用带"确认/修改"按钮的卡片。
    不再发送无按钮的"通知"卡片，确保用户主动确认后才写库。
    """
    reason = "方向变更" if direction_changed else "画像更新"
    return send_profile_confirm_card(client, channel_id, user_id, updated_profile, reason=reason)


# ── 按钮处理：✅ 确认 ─────────────────────────────────────────────────────────

def handle_profile_confirm(ack, body, client,
                            profile_memory, pending_memory,
                            user_id2names,
                            topic_handler_fn, division_handler_fn,
                            TopicContext, DivisionContext,
                            global_objects=None):
    """
    用户点击"确认"：
      1. 此处才真正写库（画像唯一性：user_id 为主键，ON DUPLICATE KEY UPDATE）
      2. 更新卡片为已确认状态
      3. 继续执行挂起意图
    """
    ack()

    user_id    = body["user"]["id"]
    channel_id = body["channel"]["id"]
    message_ts = body["container"]["message_ts"]

    print(f"[DEBUG][profile_confirm] ✅ 用户确认画像 user_id={user_id!r}")

    # 1. ★ 此处才写库（全局唯一画像，user_id 为主键）
    profile = json.loads(body["actions"][0]["value"])
    target_user_id = profile.get("_target_user_id", user_id)
    if target_user_id != user_id:
        print(f"[DEBUG][profile_confirm] ⚠ 非本人确认被拒绝 actor={user_id!r} target={target_user_id!r}")
        try:
            client.chat_postEphemeral(
                channel=channel_id,
                user=user_id,
                text="这张画像确认卡只允许目标用户本人操作。",
            )
        except Exception as e:
            print(f"[DEBUG][profile_confirm] ⚠ 发送拒绝提示失败: {e}")
        return

    # 清理内部字段
    profile.pop("_target_user_id", None)
    notify_channel = profile.pop("_channel_id", None)
    notify_ts      = profile.pop("_notify_ts", None)
    resolved_name = resolve_user_name(
        client=client,
        user_id=user_id,
        user_id2names=user_id2names,
        sql_password=profile_memory.conn_params["passwd"],
    )
    profile["user_id"]   = user_id
    profile["user_name"] = resolved_name
    profile_memory.save(profile, channel_id)
    profile_memory.mark_profile_confirmed(user_id=user_id, channel_id=channel_id)
    print(f"[DEBUG][profile_confirm] ★ 画像已写库: major={profile.get('major')!r} "
          f"interests={profile.get('research_interests')} "
          f"methods={profile.get('methodology')}")

    # 2. 更新卡片为"已确认"状态
    major     = profile.get("major", "未识别")
    interests = "、".join(profile.get("research_interests") or []) or "暂无"
    methods   = "、".join(profile.get("methodology") or []) or "暂无"
    
    # 优先使用 notify_ts（卡片自身的 ts），其次用 message_ts
    update_ts = notify_ts if notify_ts and notify_ts not in ("", "__PENDING__") else message_ts
    try:
        slack_chat_update(
            client=client,
            channel=channel_id,
            ts=update_ts,
            text=(
                f"✅ 画像已确认并保存\n\n"
                f"专业：{major} 研究兴趣：{interests} 擅长：{methods}"
            ),
            blocks=[{
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"✅ 画像已确认并保存\n\n"
                        f"*专业：* {major}　*研究兴趣：* {interests}　*擅长：* {methods}"
                    )
                }
            }]
        )
        print(f'[DEBUG][profile_confirm] 卡片已更新为"已确认"状态 ts={update_ts}')
    except Exception as e:
        print(f"[DEBUG][profile_confirm] ⚠ 更新卡片失败: {e}")

    # 3. 取出挂起意图并执行
    _resume_pending_intent(
        user_id=user_id, client=client, channel_id=channel_id,
        confirmed_profile=profile,
        pending_memory=pending_memory,
        profile_memory=profile_memory,
        user_id2names=user_id2names,
        topic_handler_fn=topic_handler_fn,
        division_handler_fn=division_handler_fn,
        TopicContext=TopicContext,
        DivisionContext=DivisionContext,
        global_objects=global_objects,
    )


# ── 按钮处理：✏️ 修改 → 打开 Modal ──────────────────────────────────────────

def handle_profile_edit(ack, body, client):
    """打开画像编辑 Modal。"""
    ack()

    profile    = json.loads(body["actions"][0]["value"])
    trigger_id = body["trigger_id"]
    user_id    = body["user"]["id"]
    target_user_id = profile.get("_target_user_id", user_id)

    if target_user_id != user_id:
        print(f"[DEBUG][profile_confirm] ⚠ 非本人编辑被拒绝 actor={user_id!r} target={target_user_id!r}")
        try:
            client.chat_postEphemeral(
                channel=body["channel"]["id"],
                user=user_id,
                text="这张画像修改卡只允许目标用户本人操作。",
            )
        except Exception as e:
            print(f"[DEBUG][profile_confirm] ⚠ 发送拒绝提示失败: {e}")
        return

    print(f"[DEBUG][profile_confirm] ✏️ 打开编辑Modal user_id={user_id!r}")

    major     = profile.get("major", "")
    interests = "、".join(profile.get("research_interests") or [])
    methods   = "、".join(profile.get("methodology") or [])
    keywords  = "、".join(profile.get("keywords") or [])

    # 把 profile（含 _channel_id/_notify_ts）序列化进 private_metadata，供 submit 恢复
    client.views_open(
        trigger_id=trigger_id,
        view={
            "type": "modal",
            "callback_id": "profile_edit_modal",
            "private_metadata": json.dumps(profile, ensure_ascii=False),
            "title": {"type": "plain_text", "text": "修改我的学术背景"},
            "submit": {"type": "plain_text", "text": "保存"},
            "close":  {"type": "plain_text", "text": "取消"},
            "blocks": [
                {
                    "type": "input",
                    "block_id": "major_block",
                    "label": {"type": "plain_text", "text": "专业 / 研究领域"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "major_input",
                        "initial_value": major,
                        "placeholder": {"type": "plain_text", "text": "如：法学、计算机科学"},
                    }
                },
                {
                    "type": "input",
                    "block_id": "interests_block",
                    "label": {"type": "plain_text", "text": "研究兴趣（用顿号分隔）"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "interests_input",
                        "initial_value": interests,
                        "placeholder": {"type": "plain_text", "text": "如：公司治理、法律科技"},
                    },
                    "optional": True,
                },
                {
                    "type": "input",
                    "block_id": "methods_block",
                    "label": {"type": "plain_text", "text": "擅长研究方法（用顿号分隔）"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "methods_input",
                        "initial_value": methods,
                        "placeholder": {"type": "plain_text", "text": "如：定量分析、深度学习"},
                    },
                    "optional": True,
                },
                {
                    "type": "input",
                    "block_id": "keywords_block",
                    "label": {"type": "plain_text", "text": "研究关键词（用顿号分隔）"},
                    "element": {
                        "type": "plain_text_input",
                        "action_id": "keywords_input",
                        "initial_value": keywords,
                        "placeholder": {"type": "plain_text", "text": "如：公司法第22条、司法实践"},
                    },
                    "optional": True,
                },
            ]
        }
    )
    print(f"[DEBUG][profile_confirm] Modal已打开")


# ── Modal 提交处理 ────────────────────────────────────────────────────────────

def handle_profile_modal_submit(ack, body, client,
                                 profile_memory, pending_memory,
                                 user_id2names,
                                 topic_handler_fn, division_handler_fn,
                                 TopicContext, DivisionContext,
                                 global_objects=None):
    """
    用户在 Modal 中保存修改：
      1. ★ 此处才真正写库
      2. 更新或发送确认消息
      3. 继续执行挂起意图
    """
    ack()

    user_id = body["user"]["id"]
    values  = body["view"]["state"]["values"]
    print(f"[DEBUG][profile_confirm] Modal提交 user_id={user_id!r}")

    # 解析表单
    major         = values["major_block"]["major_input"]["value"] or ""
    interests_str = values["interests_block"]["interests_input"]["value"] or ""
    methods_str   = values["methods_block"]["methods_input"]["value"] or ""
    keywords_str  = values.get("keywords_block", {}).get("keywords_input", {}).get("value") or ""

    def split_items(s: str) -> list[str]:
        return [x.strip() for x in s.replace(",", "、").split("、") if x.strip()]

    interests = split_items(interests_str)
    methods   = split_items(methods_str)
    keywords  = split_items(keywords_str)

    # 恢复原始 profile，再用表单内容覆盖
    orig = json.loads(body["view"]["private_metadata"])
    target_user_id   = orig.pop("_target_user_id", user_id)
    if target_user_id != user_id:
        print(f"[DEBUG][profile_confirm] ⚠ 非本人Modal提交被拒绝 actor={user_id!r} target={target_user_id!r}")
        return

    notify_channel_id = orig.pop("_channel_id", None)
    notify_ts         = orig.pop("_notify_ts", None)

    resolved_name = resolve_user_name(
        client=client,
        user_id=user_id,
        user_id2names=user_id2names,
        sql_password=profile_memory.conn_params["passwd"],
    )

    channel_id = notify_channel_id or ""
    if not channel_id:
        print("[DEBUG][profile_confirm] ⚠ Modal缺少 channel_id，无法按频道保存画像")
        return

    profile = {
        **orig,
        "user_id":            user_id,
        "user_name":          resolved_name,
        "major":              major,
        "research_interests": interests,
        "methodology":        methods,
        "keywords":           keywords,
    }
    # ★ 此处才写库
    profile_memory.save(profile, channel_id)
    profile_memory.mark_profile_confirmed(user_id=user_id, channel_id=channel_id)
    print(f"[DEBUG][profile_confirm] ★ Modal保存写库: major={major!r} "
          f"interests={interests} methods={methods} keywords={keywords}")

    pending = pending_memory.load(user_id, channel_id)

    if channel_id:
        interests_disp = "、".join(interests) or "暂无"
        methods_disp   = "、".join(methods) or "暂无"
        # 更新原确认卡片或新发消息
        if notify_ts and notify_ts not in ("", "__PENDING__"):
            try:
                slack_chat_update(
                    client=client,
                    channel=channel_id,
                    ts=notify_ts,
                    text=(
                        f"✅ <@{user_id}> 画像已修改并保存\n\n"
                        f"专业：{major or '未填写'} 研究兴趣：{interests_disp} 擅长：{methods_disp}"
                    ),
                    blocks=[{
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"✅ <@{user_id}> 画像已修改并保存\n\n"
                                f"*专业：* {major or '未填写'}　"
                                f"*研究兴趣：* {interests_disp}　"
                                f"*擅长：* {methods_disp}"
                            )
                        }
                    }]
                )
                print(f"[DEBUG][profile_confirm] 确认卡片已更新为'已保存' ts={notify_ts}")
            except Exception as e:
                print(f"[DEBUG][profile_confirm] ⚠ chat_update失败: {e}，改发新消息")
                client.chat_postMessage(
                    channel=channel_id,
                    text=f"✅ <@{user_id}> 画像已更新：专业={major or '未填写'}"
                )
        else:
            client.chat_postMessage(
                channel=channel_id,
                text=f"✅ <@{user_id}> 画像已更新：专业={major or '未填写'}"
            )

    # 继续执行挂起意图
    _resume_pending_intent(
        user_id=user_id, client=client, channel_id=channel_id,
        confirmed_profile=profile,
        pending_memory=pending_memory,
        profile_memory=profile_memory,
        user_id2names=user_id2names,
        topic_handler_fn=topic_handler_fn,
        division_handler_fn=division_handler_fn,
        TopicContext=TopicContext,
        DivisionContext=DivisionContext,
        global_objects=global_objects,
    )


# ── 恢复执行挂起意图（confirm / modal submit 共用）────────────────────────────

def _resume_pending_intent(user_id, client, channel_id, confirmed_profile,
                            pending_memory, profile_memory, user_id2names,
                            topic_handler_fn, division_handler_fn,
                            TopicContext, DivisionContext,
                            global_objects=None):
    """
    从 DB 取出挂起意图并重新执行。

    ★ 频道参与用户限制：只使用 pending payload 中 active_user_ids 列表里的用户画像，
      而不是数据库中所有用户的画像。
    ★ 画像来源：当前确认用户用 confirmed_profile，其余从 DB 读。
    """
    print(f"[DEBUG][profile_confirm] _resume_pending_intent user_id={user_id!r}")

    pending = pending_memory.load(user_id, channel_id)
    if not pending:
        print(f"[DEBUG][profile_confirm] ⚠ 无挂起意图，直接返回")
        return

    intent_label = pending["intent_label"]
    p            = pending["payload"]
    pending_memory.delete(user_id, channel_id)
    print(f"[DEBUG][profile_confirm] 恢复意图: {intent_label!r} query={p.get('query')!r}")

    g = global_objects or {}

    # ★ 严格取当前频道非Bot参与用户（以 pending 里的 active_user_ids 为准）
    bot_id = p.get("bot_id", "")
    allowed_ids = [
        uid for uid in (p.get("active_user_ids") or [])
        if uid and uid not in (bot_id, "bot_id")
    ]
    if not allowed_ids:
        allowed_ids = [user_id]

    all_profiles = []
    for uid in allowed_ids:
        uname = user_id2names.get(uid, uid)
        if uid == user_id:
            all_profiles.append(confirmed_profile)
            print(f"[DEBUG][profile_confirm] 画像来源[当前用户确认]: {uname!r}")
        else:
            pr = profile_memory.load(uid, channel_id)
            if pr and (pr.get("major") or pr.get("research_interests") or pr.get("methodology")):
                all_profiles.append(pr)
                print(f"[DEBUG][profile_confirm] 画像来源[DB-频道参与用户]: {uname!r} major={pr.get('major')!r}")
            else:
                print(f"[DEBUG][profile_confirm] 跳过[DB无内容]: {uname!r}")

    print(f"[DEBUG][profile_confirm] 共纳入画像 {len(all_profiles)} 份（当前频道非Bot用户）")

    user_only_convs = p.get("user_only_convs", "") or ""
    if user_only_convs:
        print(f"[DEBUG][profile_confirm] 恢复挂起意图：复用pending内纯用户对话，行数={len(user_only_convs.splitlines())}")
    else:
        try:
            from utils import get_user_only_conversation_history
            user_only_convs = get_user_only_conversation_history(
                client=client,
                channel_id=channel_id,
                bot_id=p.get("bot_id", ""),
                user_id2names=user_id2names,
                ts=p.get("ts", ""),
                limit=50,
            )
            print(f"[DEBUG][profile_confirm] 恢复挂起意图：已拉取纯用户对话，行数={len(user_only_convs.splitlines())}")
        except Exception as e:
            print(f"[DEBUG][profile_confirm] ⚠ 拉取纯用户对话失败，降级为空: {e}")

    if intent_label == "【选题】":
        from memory.user_profile_memory import UserProfileMemory
        from utils import send_status_message, delete_status_message

        user_profiles_text = (
            UserProfileMemory.format_for_prompt(all_profiles) if all_profiles
            else f"当前提问者：{user_id2names.get(user_id, user_id)}。"
        )
        print(f"[DEBUG][profile_confirm] 恢复选题，user_profiles_text:\n{user_profiles_text[:200]}")

        # ★ 发送"开始搜索"提示
        progress_ts = send_status_message(
            client, channel_id, user_id,
            "✅ 画像已确认！🔍 正在搜索学术资料，生成选题建议中，请稍候…"
        )

        ctx = TopicContext(
            client=client,
            channel_id=p["channel_id"],
            channel_name=p["channel_name"],
            user_id=user_id,
            user_name=p["user_name"],
            ts=p["ts"],
            query=p["query"],
            convs=p["convs"],
            intent_time=p["intent_time"],
            agent=g.get("agent"),
            search_engine=g.get("search_engine"),
            memory=g.get("memory"),
            search_memory=g.get("search_memory"),
            profile_memory=profile_memory,
            user_id2names=user_id2names,
            bot_id=p["bot_id"],
            user_only_convs=user_only_convs,
            active_user_ids=allowed_ids,
        )
        delete_status_message(client, channel_id, progress_ts)
        topic_handler_fn(ctx, user_profiles_text=user_profiles_text)

    elif intent_label == "【分工】":
        from utils import send_status_message, delete_status_message

        print(f"[DEBUG][profile_confirm] 恢复分工")

        # ★ 发送"开始搜索"提示
        progress_ts = send_status_message(
            client, channel_id, user_id,
            "✅ 画像已确认！🔍 正在搜索研究方法资料，生成分工方案中，请稍候…"
        )

        ctx = DivisionContext(
            client=client,
            channel_id=p["channel_id"],
            channel_name=p["channel_name"],
            user_id=user_id,
            user_name=p["user_name"],
            ts=p["ts"],
            query=p["query"],
            convs=p["convs"],
            intent_time=p["intent_time"],
            agent=g.get("agent"),
            memory=g.get("memory"),
            profile_memory=profile_memory,
            user_id2names=user_id2names,
            bot_id=p["bot_id"],
            user_only_convs=user_only_convs,
            active_user_ids=allowed_ids,
        )
        delete_status_message(client, channel_id, progress_ts)
        division_handler_fn(ctx, profiles=all_profiles, fallback_convs="")

    print(f"[DEBUG][profile_confirm] _resume_pending_intent 完成")