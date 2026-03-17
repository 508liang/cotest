"""
db_server.py — CoSearch Memory Viewer 后端
==========================================
启动方式：
    pip install flask flask-cors pymysql
    python db_server.py

默认监听 http://localhost:7788
前端（db_viewer.jsx）中的 API 常量已指向此地址。

数据库连接配置见下方 DB_CONFIG，与项目 utils.py 保持一致。
"""

import json
import sys
from pathlib import Path

import pymysql
from flask import Flask, jsonify
from flask_cors import CORS

try:
    from config import settings
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from config import settings

app = Flask(__name__)
CORS(app)  # 允许本地前端跨域访问

# ── 数据库配置（与 utils.py 一致）────────────────────────────────────────────
DB_CONFIG = dict(
    host=settings.db_host,
    user=settings.db_user,
    passwd=settings.db_password,
    port=settings.db_port,
    db=settings.db_name,
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def load_channel_map():
    """读取 channel_info，返回大小写兼容的 {channel_id: channel_name} 映射。"""
    mapping = {}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE 'channel_info'")
            has_table = bool(cur.fetchone())
            if not has_table:
                return mapping
            cur.execute("SELECT channel_id, channel_name FROM channel_info")
            for row in cur.fetchall():
                cid = (row.get("channel_id") or "").strip()
                cname = (row.get("channel_name") or "").strip()
                if cid and cname:
                    # 同时存原值与小写值，兼容数据库里大写ID、表名小写的情况
                    mapping[cid] = cname
                    mapping[cid.lower()] = cname
    except Exception:
        return mapping
    finally:
        conn.close()
    return mapping


def is_conversation_table(conn, table_name: str) -> bool:
    """仅保留结构符合对话表的表。"""
    required = {"speaker", "utterance", "timestamp"}
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM `{table_name}`")
            cols = {r.get("Field") for r in cur.fetchall()}
        return required.issubset(cols)
    except Exception:
        return False


# ── 工具：列出所有用户建的表 ──────────────────────────────────────────────────
def list_tables():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        rows = cur.fetchall()
    conn.close()
    # DictCursor 返回 {"Tables_in_mysql": "table_name"}
    return [list(r.values())[0] for r in rows]


# ── /profiles ────────────────────────────────────────────────────────────────
@app.get("/profiles")
def get_profiles():
    """读取 user_profile 表全量数据。"""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, user_name, major, "
                "research_interests, methodology, keywords, updated_at "
                "FROM user_profile ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # JSON 字段反序列化
    profiles = []
    for r in rows:
        for field in ("research_interests", "methodology", "keywords"):
            val = r.get(field)
            if val:
                try:
                    r[field] = json.loads(val)
                except Exception:
                    r[field] = []
            else:
                r[field] = []
        profiles.append(r)

    return jsonify({"profiles": profiles})


# ── /channels ─────────────────────────────────────────────────────────────────
@app.get("/channels")
def get_channels():
    """
    列出所有对话频道表（排除系统表和 _search / _click 后缀表）。
    返回 [{name, count}]。
    """
    all_tables = list_tables()
    system_tables = {
        "user_profile", "user_info", "channel_info",
        "pending_intent", "rag_results",
    }
    candidate_tables = [
        t for t in all_tables
        if t not in system_tables
        and not t.endswith("_search")
        and not t.endswith("_click")
        and not t.startswith("help_")
    ]

    channel_map = load_channel_map()

    channels = []
    conn = get_conn()
    for t in candidate_tables:
        try:
            if not is_conversation_table(conn, t):
                continue
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) as cnt FROM `{t}`")
                cnt = cur.fetchone()["cnt"]
            display_name = channel_map.get(t) or channel_map.get(t.lower()) or t
            channels.append({
                "name": t,
                "display_name": display_name,
                "count": cnt,
            })
        except Exception:
            pass
    conn.close()

    channels.sort(key=lambda x: x["count"], reverse=True)
    return jsonify({"channels": channels})


# ── /convs/<channel> ──────────────────────────────────────────────────────────
@app.get("/convs/<channel>")
def get_convs(channel):
    """读取指定频道的完整对话记录，按时间升序。"""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT speaker, utterance, query, rewrite_query, "
                f"clarify, search_results, infer_time, timestamp "
                f"FROM `{channel}` ORDER BY timestamp ASC"
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"rows": rows})


# ── /users ────────────────────────────────────────────────────────────────────
@app.get("/users")
def get_users():
    """读取 user_info 表（user_id ↔ user_name 映射）。"""
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, user_name FROM user_info ORDER BY user_name")
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"users": rows})


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("CoSearch Memory Viewer 后端启动中...")
    print(f"监听地址：http://localhost:7788")
    print(f"数据库：{DB_CONFIG['host']}:{DB_CONFIG['port']} / {DB_CONFIG['db']}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=7788, debug=True)
