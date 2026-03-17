"""
Backfill user names in user_info and user_profile from Slack users.info.

Usage:
  python backend/backfill_user_names.py --dry-run
  python backend/backfill_user_names.py

Optional:
  python backend/backfill_user_names.py --host localhost --user root --password xxx --db mysql
"""

import argparse
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill user names from Slack users.info")
    parser.add_argument("--host", default=settings.db_host)
    parser.add_argument("--port", type=int, default=settings.db_port)
    parser.add_argument("--user", default=settings.db_user)
    parser.add_argument("--password", default=settings.db_password)
    parser.add_argument("--db", default=settings.db_name)
    parser.add_argument("--dry-run", action="store_true", help="Only print updates, do not write DB")
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


def ensure_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_info (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL UNIQUE,
                user_name VARCHAR(128)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(64) NOT NULL UNIQUE,
                user_name VARCHAR(128),
                major VARCHAR(128),
                research_interests TEXT,
                methodology TEXT,
                keywords TEXT,
                updated_at VARCHAR(32),
                last_confirmed_ts DOUBLE DEFAULT 0
            )
            """
        )
    conn.commit()


def collect_user_ids(conn) -> List[str]:
    ids: Set[str] = set()
    with conn.cursor() as cur:
        cur.execute("SHOW TABLES LIKE 'user_info'")
        if cur.fetchone():
            cur.execute("SELECT user_id FROM user_info")
            for row in cur.fetchall():
                uid = (row.get("user_id") or "").strip()
                if uid.startswith("U"):
                    ids.add(uid)

        cur.execute("SHOW TABLES LIKE 'user_profile'")
        if cur.fetchone():
            cur.execute("SELECT user_id FROM user_profile")
            for row in cur.fetchall():
                uid = (row.get("user_id") or "").strip()
                if uid.startswith("U"):
                    ids.add(uid)

    return sorted(ids)


def fetch_user_name(client: WebClient, user_id: str) -> str:
    try:
        resp = client.users_info(user=user_id)
        profile = resp.get("user", {}).get("profile", {})
        name = (profile.get("display_name") or profile.get("real_name") or user_id).strip()
        return name or user_id
    except SlackApiError as e:
        data = e.response.data if getattr(e, "response", None) else {}
        err = data.get("error")
        if err == "missing_scope":
            needed = data.get("needed", "")
            provided = data.get("provided", "")
            print(
                "[WARN] missing_scope for users.info. "
                f"needed={needed} provided={provided}. "
                "Please add users:read and reinstall app."
            )
        else:
            print(f"[WARN] users_info failed for {user_id}: {repr(e)}")
    except Exception as e:
        print(f"[WARN] users_info failed for {user_id}: {repr(e)}")
    return user_id


def upsert_user_names(conn, mapping: Dict[str, str], dry_run: bool = False) -> None:
    if dry_run:
        print("[DRY-RUN] would update user names:")
        for uid, uname in mapping.items():
            print(f"  {uid} -> {uname}")
        return

    with conn.cursor() as cur:
        for uid, uname in mapping.items():
            cur.execute(
                """
                INSERT INTO user_info (user_id, user_name)
                VALUES (%s, %s)
                ON DUPLICATE KEY UPDATE user_name = VALUES(user_name)
                """,
                (uid, uname),
            )
            cur.execute(
                """
                UPDATE user_profile
                SET user_name = %s
                WHERE user_id = %s
                """,
                (uname, uid),
            )
    conn.commit()


def main() -> None:
    args = parse_args()
    token = os.getenv("SLACK_BOT_TOKEN", settings.slack_bot_token).strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is required in environment")
    if not token.startswith("xoxb-"):
        raise RuntimeError("SLACK_BOT_TOKEN format looks invalid. Expected xoxb-...")

    conn = get_conn(args)
    try:
        ensure_tables(conn)
        user_ids = collect_user_ids(conn)
        print(f"[INFO] collected {len(user_ids)} user ids")

        client = WebClient(token=token)
        mapping: Dict[str, str] = {}
        for uid in user_ids:
            uname = fetch_user_name(client, uid)
            mapping[uid] = uname
            print(f"[INFO] {uid} -> {uname}")

        upsert_user_names(conn, mapping, dry_run=args.dry_run)
        print("[DONE] user names backfill complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
