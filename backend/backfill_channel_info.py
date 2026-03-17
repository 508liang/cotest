"""
Backfill channel_info from existing conversation tables.

Usage:
  set SLACK_BOT_TOKEN=xoxb-...
  python backend/backfill_channel_info.py

Optional:
  python backend/backfill_channel_info.py --dry-run
  python backend/backfill_channel_info.py --host localhost --user root --password root123456 --db mysql
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import pymysql
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

try:
    from config import settings
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from config import settings


SYSTEM_TABLES = {
    "user_profile",
    "user_info",
    "channel_info",
    "pending_intent",
    "rag_results",
    "seen_channels",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill channel_info table from channel IDs")
    parser.add_argument("--host", default=settings.db_host)
    parser.add_argument("--port", type=int, default=settings.db_port)
    parser.add_argument("--user", default=settings.db_user)
    parser.add_argument("--password", default=settings.db_password)
    parser.add_argument("--db", default=settings.db_name)
    parser.add_argument("--dry-run", action="store_true", help="Only print results, do not write DB")
    return parser.parse_args()


def get_conn(args: argparse.Namespace):
    return pymysql.connect(
        host=args.host,
        user=args.user,
        passwd=args.password,
        port=args.port,
        db=args.db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def list_tables(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        rows = cur.fetchall()
    return [list(r.values())[0] for r in rows]


def is_conversation_table(conn, table_name: str) -> bool:
    required = {"speaker", "utterance", "timestamp"}
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW COLUMNS FROM `{table_name}`")
            cols = {r.get("Field") for r in cur.fetchall()}
        return required.issubset(cols)
    except Exception:
        return False


def normalize_channel_id(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        return ""
    # Slack channel IDs are uppercase alnum (e.g., C0ALPNJR4GG)
    return val.upper()


def is_slack_channel_id(value: str) -> bool:
    """Accept common Slack conversation IDs: C..., G..., D..."""
    return bool(re.fullmatch(r"[CGD][A-Z0-9]{8,15}", value or ""))


def collect_channel_ids(conn) -> Set[str]:
    all_tables = list_tables(conn)
    channel_ids: Set[str] = set()

    for t in all_tables:
        if t in SYSTEM_TABLES:
            continue
        if t.endswith("_search") or t.endswith("_click") or t.startswith("help_"):
            continue
        if not is_conversation_table(conn, t):
            continue
        cid = normalize_channel_id(t)
        if is_slack_channel_id(cid):
            channel_ids.add(cid)

    # Also keep existing IDs in channel_info.
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'channel_info'")
        if cur.fetchone():
            cur.execute("SELECT channel_id FROM channel_info")
            for row in cur.fetchall():
                cid = normalize_channel_id(row.get("channel_id", ""))
                if is_slack_channel_id(cid):
                    channel_ids.add(cid)

    return channel_ids


def fetch_channel_name(client: WebClient, channel_id: str) -> str:
    try:
        resp = client.conversations_info(channel=channel_id)
        ch = resp.get("channel", {})
        name = (ch.get("name") or "").strip()
        if name:
            return name
    except SlackApiError as e:
        data = e.response.data if getattr(e, "response", None) else {}
        err = data.get("error")
        if err == "missing_scope":
            needed = data.get("needed", "")
            provided = data.get("provided", "")
            print(
                "[WARN] missing_scope for conversations.info. "
                f"needed={needed} provided={provided}. "
                "Please add Bot Token Scopes and reinstall app."
            )
        else:
            print(f"[WARN] conversations_info failed for {channel_id}: {repr(e)}")
    except Exception as e:
        print(f"[WARN] conversations_info failed for {channel_id}: {repr(e)}")
    return channel_id


def ensure_channel_info_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS channel_info (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(64) NOT NULL UNIQUE,
                channel_name VARCHAR(128)
            )
            """
        )
    conn.commit()


def upsert_channel_info(conn, channel_map: Dict[str, str], dry_run: bool = False):
    if dry_run:
        print("[DRY-RUN] would upsert channel_info:")
        for cid, cname in channel_map.items():
            print(f"  {cid} -> {cname}")
        return

    with conn.cursor() as cur:
        for cid, cname in channel_map.items():
            cur.execute(
                """
                INSERT INTO channel_info (channel_id, channel_name)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE channel_name = VALUES(channel_name)
                """,
                (cid, cname),
            )
    conn.commit()


def main():
    args = parse_args()
    token = os.getenv("SLACK_BOT_TOKEN", settings.slack_bot_token).strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is required in environment")
    try:
        token.encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(
            "SLACK_BOT_TOKEN contains non-ASCII characters. "
            "Please set a real bot token like xoxb-... (do not use placeholder text)."
        )
    if not token.startswith("xoxb-"):
        raise RuntimeError(
            "SLACK_BOT_TOKEN format looks invalid. Expected token prefix xoxb-."
        )

    conn = get_conn(args)
    try:
        ensure_channel_info_table(conn)
        channel_ids = sorted(collect_channel_ids(conn))
        print(f"[INFO] collected {len(channel_ids)} channel ids")

        client = WebClient(token=token)
        channel_map: Dict[str, str] = {}
        for cid in channel_ids:
            cname = fetch_channel_name(client, cid)
            channel_map[cid] = cname
            print(f"[INFO] {cid} -> {cname}")

        upsert_channel_info(conn, channel_map, dry_run=args.dry_run)
        print("[DONE] channel_info backfill complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
