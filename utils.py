import pymysql
import nltk
import re
from rouge import Rouge
from config import settings

# API keys (loaded from centralized settings)
SERPAPI_KEY = settings.serpapi_key
OPENAI_KEY = settings.openai_api_key

# Database password (loaded from centralized settings)
SQL_PASSWORD = settings.db_password


def get_table_info(table_name):
    connection = pymysql.connect(
        host=settings.db_host,
        user=settings.db_user,
        passwd=SQL_PASSWORD,
        port=settings.db_port,
        db=settings.db_name,
    )

    query = f"SELECT * FROM {table_name}"

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            table_contents = cursor.fetchall()
        return table_contents
    except pymysql.err.ProgrammingError as e:
        # 1146: table does not exist. Startup should not fail when metadata tables are absent.
        if len(e.args) >= 1 and e.args[0] == 1146:
            print(f"[DEBUG][utils] 表 {table_name!r} 不存在，返回空结果")
            return []
        raise
    finally:
        connection.close()


def get_channel_info(table_name="channel_info"):
    contents = get_table_info(table_name=table_name)
    channel_id2names = {content[1]: content[2] for content in contents}
    return channel_id2names


def ensure_user_info_table_exists() -> None:
    """确保 user_info 元数据表存在，避免启动时因缺表产生噪声日志。"""
    try:
        connection = pymysql.connect(
            host=settings.db_host,
            user=settings.db_user,
            passwd=SQL_PASSWORD,
            port=settings.db_port,
            db=settings.db_name,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_info (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(64) NOT NULL UNIQUE,
                    user_name VARCHAR(128)
                )
                """
            )
        connection.commit()
        connection.close()
    except Exception as e:
        print(f"[DEBUG][utils] 确保 user_info 表存在失败: {e}")


def get_user_info(table_name="user_info"):
    if table_name == "user_info":
        ensure_user_info_table_exists()
    contents = get_table_info(table_name=table_name)
    user_id2names = {content[1]: content[2] for content in contents}
    return user_id2names


def send_rag_answer(client, channel_id, user_id, query, answer, references, start=0, end=2):
    formatted_answer = format_response(response=answer, user_id=user_id)
    blocks = send_answer_block(query, formatted_answer, references, start, end)
    response = client.chat_postMessage(
        channel=channel_id,
        text=build_blocks_fallback_text(blocks),
        blocks=blocks
    )
    return response


def send_link_only_rag_answer(client, channel_id, user_id, answer, references, max_links=3):
    formatted_answer = format_response(
        response=strip_reference_markers(answer),
        user_id=user_id,
    )
    blocks = send_link_only_answer_block(formatted_answer, references, max_links=max_links)
    response = client.chat_postMessage(
        channel=channel_id,
        text=build_blocks_fallback_text(blocks),
        blocks=blocks
    )
    return response


def send_rag_references(client, channel_id: str, query: str, user_id: str,
                         references: list[dict], start: int = 0, end: int = 2) -> dict:
    """
    单独发送参考文献来源卡片（不含正文），供流式输出场景追加在正文消息之后。
    直接构建来源 blocks，避免空正文被 Slack 拒绝。
    """
    if not references:
        print(f"[DEBUG][send_rag_references] 无来源，跳过")
        return {}

    end = min(end, len(references))
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":mag: *{query}* ({len(references)} 条参考来源)"
            }
        },
        {"type": "divider"},
    ]

    for i, result in enumerate(references[start:end]):
        link = (result.get("link") or "").strip()
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*[{start + i + 1}] {title}*\n{snippet}"
            }
        }
        if link:
            block["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "Click", "emoji": True},
                "value": link,
                "url": link,
                "action_id": "click"
            }
        blocks.append(block)

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "emoji": True, "text": "Previous"},
                "style": "primary",
                "value": "click_me_123",
                "action_id": "previous"
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "emoji": True, "text": "Next"},
                "style": "primary",
                "value": "click_me_123",
                "action_id": "next"
            },
        ]
    })

    try:
        resp = client.chat_postMessage(
            channel=channel_id,
            text=build_blocks_fallback_text(blocks),
            blocks=blocks,
        )
        print(f"[DEBUG][send_rag_references] 来源卡片发送成功 ts={resp['ts']} 条数={len(references[start:end])}")
        return resp
    except Exception as e:
        print(f"[DEBUG][send_rag_references] ⚠ 发送失败: {e}")
        return {}


def send_utterance(client, channel_id, utterance):
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": utterance
            }
        }
    ]
    response = client.chat_postMessage(
        channel=channel_id,
        text=utterance,
        blocks=blocks
    )
    return response


def delete_utterance(client, channel_id, ts):
    client.chat_delete(
        channel=channel_id,
        ts=ts
    )


def update_rag_answer(client, channel_id, user_id, query, answer, references, ts, start=0, end=2):
    formatted_answer = format_response(response=answer, user_id=user_id)
    blocks = send_answer_block(query, formatted_answer, references, start, end)
    response = slack_chat_update(
        client=client,
        channel=channel_id,
        ts=ts,
        blocks=blocks,
    )
    return response


def send_clarify_question(client, channel_id, user_id, clarify_question):
    clarify_question = format_response(response=clarify_question, user_id=user_id)
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": clarify_question
            }
        }
    ]
    response = client.chat_postMessage(
        channel=channel_id,
        text=clarify_question,
        blocks=blocks
    )
    return response


def send_answer(client, channel_id, user_id, answer):
    answer = format_response(response=answer, user_id=user_id)
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": answer
            }
        }
    ]
    response = client.chat_postMessage(
        channel=channel_id,
        text=answer,
        blocks=blocks
    )
    return response


def get_conversation_history(client, channel_id, bot_id, user_id2names, ts, limit=20):
    messages = client.conversations_history(channel=channel_id, latest=ts, limit=limit)["messages"]
    convs = []
    for message in reversed(messages):
        if "已加入此频道" in message["text"]:
            continue
        if message["user"] == bot_id:
            # blocks 不一定存在，fallback 到 text
            bot_text = (
                message.get("blocks", [{}])[0].get("text", {}).get("text")
                or message.get("text", "")
            )
            convs.append(
                f"{user_id2names.get(message['user'], message['user'])}: {replace_utterance_ids(bot_text, user_id2names)}")
        else:
            convs.append(f"{user_id2names.get(message['user'], message['user'])}: {replace_utterance_ids(message['text'], user_id2names)}")
    return "\n".join(convs)

def get_user_only_conversation_history(client, channel_id, bot_id, user_id2names, ts, limit=20):
    """
    只返回人类用户的发言，在API层面整条跳过Bot消息。
    用于画像提炼，彻底避免Bot回复内容（包括多行）污染用户画像。
    limit 默认50，确保捕获足够的用户发言（Bot消息多时真实用户消息可能被稀释）。
    """
    # ── DEBUG ──────────────────────────────────────────────────────────────
    print(f"[DEBUG][get_user_only_history] channel={channel_id} ts={ts} limit={limit}")
    # ───────────────────────────────────────────────────────────────────────
    messages = client.conversations_history(channel=channel_id, latest=ts, limit=limit)["messages"]
    print(f"[DEBUG][get_user_only_history] 原始消息总数={len(messages)}")
    convs = []
    skipped_bot = 0
    skipped_join = 0
    for message in reversed(messages):
        text = message.get("text", "")
        if "已加入此频道" in text:
            skipped_join += 1
            continue
        if message.get("user") == bot_id:
            skipped_bot += 1
            continue  # 整条Bot消息直接跳过
        speaker = user_id2names.get(message.get("user", ""), message.get("user", "未知"))
        cleaned = replace_utterance_ids(text, user_id2names)
        convs.append(f"{speaker}: {cleaned}")
    print(
        f"[DEBUG][get_user_only_history] 保留用户消息={len(convs)} "
        f"跳过Bot={skipped_bot} 跳过入频={skipped_join}"
    )
    result = "\n".join(convs)
    print(f"[DEBUG][get_user_only_history] 最终文本预览（前300字）:\n{result[:300]}")
    return result

def add_brackets_to_numbers(answer, threshold):
    def process_match(match):
        number = int(match.group(1))
        if number > threshold:
            return ''
        return f'[{match.group(1)}]'

    answer = re.sub(r'\[(\d+)\]', process_match, answer)
    answer = re.sub(r'(\[\d+\]\s*)+', lambda m: f' *{m.group()}* ', answer)
    return answer


def strip_reference_markers(answer):
    cleaned = re.sub(r'\[(\d+)\]', '', answer or '')
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\s+([，。！？；：,.!?;:])', r'\1', cleaned)
    return cleaned.strip()


def build_blocks_fallback_text(blocks):
    parts = []

    for block in blocks or []:
        block_type = block.get("type")
        if block_type == "section":
            text_obj = block.get("text") or {}
            text = (text_obj.get("text") or "").strip()
            if text:
                parts.append(text)
        elif block_type == "context":
            for element in block.get("elements") or []:
                text = (element.get("text") or "").strip()
                if text:
                    parts.append(text)
        elif block_type == "actions":
            labels = []
            for element in block.get("elements") or []:
                text_obj = element.get("text") or {}
                label = (text_obj.get("text") or "").strip()
                if label:
                    labels.append(label)
            if labels:
                parts.append("操作: " + " / ".join(labels))

    return "\n".join(part for part in parts if part).strip() or "Slack message"


def slack_chat_update(client, channel, ts, blocks=None, text=""):
    payload = {
        "channel": channel,
        "ts": ts,
        "text": (text or "").strip() or build_blocks_fallback_text(blocks),
    }
    if blocks is not None:
        payload["blocks"] = blocks
    return client.chat_update(**payload)


def send_link_only_answer_block(answer, results, max_links=3):
    blocks = [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": answer
        }
    }]

    links = []
    seen = set()
    for result in results or []:
        link = (result.get("link") or "").strip()
        if not link or link in seen:
            continue
        seen.add(link)
        links.append(link)
        if len(links) >= max_links:
            break

    if links:
        link_text = "\n".join(
            f"• <{link}|链接{i + 1}>"
            for i, link in enumerate(links)
        )
        blocks.extend([
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*参考链接*\n{link_text}"
                }
            }
        ])

    return blocks


def send_answer_block(query, answer, results, start, end):
    blocks = [{
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": add_brackets_to_numbers(answer, len(results))
        }
    }, {
        "type": "divider"
    }, {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f":mag: *{query}* ({len(results)} results)"
        }
    },
        {
            "type": "divider"
        }]
    for i, result in enumerate(results[start:end]):
        link = (result.get("link") or "").strip()
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*[{start + i + 1}] {result.get('title', '')}*\n{result.get('snippet', '')}"
            }
        }
        if link:
            block["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "Click", "emoji": True},
                "value": link,
                "url": link,
                "action_id": "click"
            }
        blocks.append(block)
    blocks.append({
        "type": "divider"
    })
    blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "emoji": True,
                    "text": "Previous"
                },
                "style": "primary",
                "value": "click_me_123",
                "action_id": "previous"
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "emoji": True,
                    "text": "Next"
                },
                "style": "primary",
                "value": "click_me_123",
                "action_id": "next"
            },
        ]
    })
    return blocks


def replace_utterance_ids(utterance, id2names):
    for user_id, user_name in id2names.items():
        id_pattern = f"<@{user_id}>"
        if id_pattern in utterance:
            utterance = utterance.replace(id_pattern, f"@{user_name}")
    return utterance


def format_response(response, user_id):
    return f"<@{user_id}> {response}"


def get_search_blocks(query, results, user_id, start, end):
    search_blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"<@{user_id}> :mag: *{query}* ({len(results)} results)"
            }
        },
        {
            "type": "divider"
        }
    ]
    for i, result in enumerate(results[start:end]):
        link = (result.get("link") or "").strip()
        block = {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*[{start + i + 1}] {result.get('title', '')}*\n{result.get('snippet', '')}"
            }
        }
        if link:
            block["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "Click", "emoji": True},
                "value": link,
                "url": link,
                "action_id": "click"
            }
        search_blocks.append(block)
    search_blocks.append({
        "type": "divider"
    })
    search_blocks.append({
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "emoji": True,
                    "text": "Previous"
                },
                "style": "primary",
                "value": "click_me_123",
                "action_id": "previous"
            },
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "emoji": True,
                    "text": "Next"
                },
                "style": "primary",
                "value": "click_me_123",
                "action_id": "next"
            },
        ]
    })

    return search_blocks


def send_search_results(client, query, search_results, channel_id, user_id, start=0, end=3):
    blocks = get_search_blocks(query=query,
                               results=search_results,
                               user_id=user_id,
                               start=start,
                               end=end)
    response = client.chat_postMessage(
        channel=channel_id,
        text=build_blocks_fallback_text(blocks),
        blocks=blocks
    )
    return response


def update_search_results(client, query, search_results, channel_id, user_id, start, end, ts):
    blocks = get_search_blocks(query=query,
                               results=search_results,
                               user_id=user_id,
                               start=start,
                               end=end)
    response = slack_chat_update(
        client=client,
        channel=channel_id,
        ts=ts,
        blocks=blocks,
    )
    return response


def chinese_sentence_tokenizer(text):
    pattern = r'[^？。！]*[？。！]'

    sentences = nltk.regexp_tokenize(text, pattern)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def rouge1_similarity(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    rouge_1_score = scores[0]['rouge-1']['f']
    return rouge_1_score


def get_top10_similar_results(answer, search_results):
    similarity_scores = [(index, rouge1_similarity(" ".join(answer), " ".join(result["snippet"])))
                         for index, result in enumerate(search_results) if result["snippet"]]
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    top5_results = [search_results[index] for index, _ in similarity_scores[:10]]
    return top5_results


def merge_citation_marks(answer, search_results, check_search_results):
    search_results = [item for sublist in [search_results, check_search_results] for item in sublist]
    search_results = list({tuple(sorted(d.items())): d for d in search_results}.values())
    search_results = get_top10_similar_results(answer, search_results)

    answer_segs = chinese_sentence_tokenizer(answer)
    for answer_seg in answer_segs:
        scores = {i: rouge1_similarity(answer_seg, result["snippet"]) for i, result in enumerate(search_results)}
        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:2]

        marks = [item[0] for item in scores if item[1] >= 0.3]
        if not marks:
            continue

        for mark in marks:
            answer_seg += f" *<{search_results[mark]['link']}|[{mark + 1}]>*"
    # for item, result in enumerate(search_results):
    #     scores = {i: rouge1_similarity(answer_seg, result["snippet"]) for i, answer_seg in enumerate(answer_segs)}
    #     scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    #     print(scores)
    #     marks = {k: v for k, v in scores.items() if v > 0.6}
    #     if not marks:
    #         max_key, max_value = max(scores.items(), key=lambda item: item[1])
    #         marks = {max_key: max_value}
    #
    #     for k in marks.keys():
    #         answer_segs[k] += f" *<{result['link']}|[{item + 1}]>*"
    #
    # answer = " ".join(answer_segs)
    # print(answer)
    return answer, search_results


def resolve_user_name(client, user_id: str, user_id2names: dict, sql_password: str) -> str:
    """
    动态解析未知用户的真实姓名：
    1. 优先从内存字典取（已知用户直接返回）
    2. 未知时调 Slack API 获取 display_name / real_name
    3. 写入内存字典和数据库 user_info 表，供后续复用
    返回用户名字符串（失败时返回 user_id 本身）
    """
    def _looks_like_user_id(value: str) -> bool:
        return bool(re.fullmatch(r"[UW][A-Z0-9]{8,15}", (value or "").strip()))

    cached_name = (user_id2names.get(user_id) or "").strip()
    if cached_name and not _looks_like_user_id(cached_name):
        return cached_name

    db_name = ""
    try:
        connection = pymysql.connect(
            host=settings.db_host,
            user=settings.db_user,
            passwd=sql_password,
            port=settings.db_port,
            db=settings.db_name,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT user_name FROM user_info WHERE user_id = %s LIMIT 1",
                (user_id,),
            )
            row = cursor.fetchone()
            if row:
                db_name = (row[0] if isinstance(row, tuple) else row.get("user_name") or "").strip()
        connection.close()
    except Exception as e:
        print(f"DEBUG: 读取 user_info 失败: {e}")

    try:
        resp = client.users_info(user=user_id)
        profile = resp["user"]["profile"]
        name = (profile.get("display_name") or profile.get("real_name") or "").strip()
    except Exception as e:
        print(f"DEBUG: 无法获取用户 {user_id} 信息: {e}")
        name = ""

    # 优先用 Slack 名称，其次用 DB 已有昵称；都没有时再回退 user_id
    if not name:
        if db_name and not _looks_like_user_id(db_name):
            name = db_name
        elif cached_name and not _looks_like_user_id(cached_name):
            name = cached_name
        else:
            name = user_id

    # 写入内存
    user_id2names[user_id] = name

    # 写入数据库 user_info 表，重启后仍可复用
    try:
        ensure_user_info_table_exists()
        connection = pymysql.connect(
            host=settings.db_host,
            user=settings.db_user,
            passwd=sql_password,
            port=settings.db_port,
            db=settings.db_name,
        )
        with connection.cursor() as cursor:
            if not _looks_like_user_id(name):
                cursor.execute(
                    "INSERT INTO user_info (user_id, user_name) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE user_name = VALUES(user_name)",
                    (user_id, name)
                )
            else:
                print(f"DEBUG: 跳过覆盖 user_info，避免把昵称写回ID [{user_id}]")
        connection.commit()
        connection.close()
        print(f"DEBUG: 新用户已注册 [{user_id}] → [{name}]")
    except Exception as e:
        print(f"DEBUG: 写入 user_info 失败: {e}")

    return name


def register_channel_display_name(client, channel_id: str, sql_password: str) -> str:
    """
    解析频道可读名称并写入 channel_info。
    注意：仅用于桌面浏览器展示，不应作为对话表名。
    """
    display_name = channel_id
    try:
        resp = client.conversations_info(channel=channel_id)
        ch = resp.get("channel", {})
        display_name = (
            ch.get("name")
            or ch.get("name_normalized")
            or channel_id
        ).strip()
    except Exception as e:
        print(f"[DEBUG][channel_info] 无法解析频道 {channel_id} 名称: {e}")

    try:
        connection = pymysql.connect(
            host=settings.db_host,
            user=settings.db_user,
            passwd=sql_password,
            port=settings.db_port,
            db=settings.db_name,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_info (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    channel_id VARCHAR(64) NOT NULL UNIQUE,
                    channel_name VARCHAR(128)
                )
                """
            )
            cursor.execute(
                "INSERT INTO channel_info (channel_id, channel_name) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE channel_name = VALUES(channel_name)",
                (channel_id, display_name),
            )
        connection.commit()
        connection.close()
        print(f"[DEBUG][channel_info] 已更新 [{channel_id}] -> [{display_name}]")
    except Exception as e:
        print(f"[DEBUG][channel_info] 写入 channel_info 失败: {e}")

    return display_name


def is_new_channel(channel_id: str, channel_name: str,
                   seen_channels: set, sql_password: str) -> bool:
    """
    判断是否为 Bot 首次接触的频道：
    1. 先查内存 Set（本次运行已见过 → False）
    2. 再查数据库 seen_channels 表（历史运行已见过 → False）
    3. 两处都没有 → 写库、加入内存 Set → 返回 True
    """
    if channel_id in seen_channels:
        return False

    try:
        connection = pymysql.connect(
            host="localhost", user="root", passwd=sql_password, port=3306, db="mysql"
        )
        with connection.cursor() as cursor:
            # 建表（首次运行时自动创建）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seen_channels (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    channel_id   VARCHAR(64) NOT NULL UNIQUE,
                    channel_name VARCHAR(128),
                    first_seen   VARCHAR(32)
                )
            """)
            # 查是否已记录
            cursor.execute(
                "SELECT COUNT(*) FROM seen_channels WHERE channel_id = %s",
                (channel_id,)
            )
            exists = cursor.fetchone()[0] > 0

            if not exists:
                from datetime import datetime
                cursor.execute(
                    "INSERT INTO seen_channels (channel_id, channel_name, first_seen) "
                    "VALUES (%s, %s, %s)",
                    (channel_id, channel_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                connection.commit()
                print(f"[DEBUG][is_new_channel] 新频道已注册: {channel_id!r} ({channel_name!r})")

        connection.close()
    except Exception as e:
        print(f"[DEBUG][is_new_channel] DB操作失败: {e}")
        exists = False

    # 无论如何加入内存 Set，避免同次运行重复触发
    seen_channels.add(channel_id)

    return not exists


def send_status_message(client, channel_id: str, user_id: str, text: str) -> str:
    """
    发送一条临时状态提示消息（如"正在搜索资料..."），返回 ts 供后续删除或更新。
    """
    resp = client.chat_postMessage(
        channel=channel_id,
        text=f"<@{user_id}> {text}",
        blocks=[{
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"<@{user_id}> {text}"}
        }]
    )
    return resp["ts"]


def delete_status_message(client, channel_id: str, ts: str):
    """删除状态提示消息。"""
    try:
        client.chat_delete(channel=channel_id, ts=ts)
    except Exception as e:
        print(f"[DEBUG][delete_status_message] 删除失败: {e}")


def get_mention_count(user_id: str, sql_password: str) -> int:
    """获取用户累计@Bot次数，表不存在时返回0。"""
    try:
        conn = pymysql.connect(host="localhost", user="root", passwd=sql_password, port=3306, db="mysql")
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_mention_count (
                    user_id       VARCHAR(64) NOT NULL PRIMARY KEY,
                    mention_count INT DEFAULT 0
                )
            """)
            cur.execute(
                "SELECT mention_count FROM user_mention_count WHERE user_id = %s",
                (user_id,)
            )
            row = cur.fetchone()
        conn.commit()
        conn.close()
        return row[0] if row else 0
    except Exception as e:
        print(f"[DEBUG][get_mention_count] 失败: {e}")
        return 0


def increment_mention_count(user_id: str, sql_password: str) -> int:
    """
    累加用户@Bot次数并返回更新后的值。
    首次调用时自动创建记录。
    """
    try:
        conn = pymysql.connect(host="localhost", user="root", passwd=sql_password, port=3306, db="mysql")
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_mention_count (
                    user_id       VARCHAR(64) NOT NULL PRIMARY KEY,
                    mention_count INT DEFAULT 0
                )
            """)
            cur.execute("""
                INSERT INTO user_mention_count (user_id, mention_count)
                VALUES (%s, 1)
                ON DUPLICATE KEY UPDATE mention_count = mention_count + 1
            """, (user_id,))
            cur.execute(
                "SELECT mention_count FROM user_mention_count WHERE user_id = %s",
                (user_id,)
            )
            count = cur.fetchone()[0]
        conn.commit()
        conn.close()
        print(f"[DEBUG][increment_mention_count] user={user_id} count={count}")
        return count
    except Exception as e:
        print(f"[DEBUG][increment_mention_count] 失败: {e}")
        return 0