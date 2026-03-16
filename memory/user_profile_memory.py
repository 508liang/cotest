"""
user_profile_memory.py

管理用户学术背景 Profile 的持久化存取与更新。

Profile 字段：
    channel_id         : Slack Channel ID（与 user_id 组成唯一键）
    user_id            : Slack User ID
    user_name          : 用户名
    major              : 所属学科/专业
    research_interests : 研究兴趣（JSON数组）
    methodology        : 擅长的研究方法（JSON数组）
    updated_at         : 最后更新时间
"""

import json
import pymysql
from datetime import date
from config import settings


class UserProfileMemory:
    def __init__(
        self,
        sql_password,
        host=settings.db_host,
        user=settings.db_user,
        port=settings.db_port,
        db=settings.db_name,
    ):
        self.conn_params = dict(host=host, user=user, passwd=sql_password, port=port, db=db)

    def _conn(self):
        return pymysql.connect(**self.conn_params)

    @staticmethod
    def _normalize_channel_id(channel_id: str | None) -> str:
        return (channel_id or "").strip()

    def create_table_if_not_exists(self, table_name="user_profile"):
        conn = self._conn()
        with conn.cursor() as cur:
            # 建表（不含新字段也没关系，下面会自动补列）
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                    id                  INT AUTO_INCREMENT PRIMARY KEY,
                    channel_id          VARCHAR(64)  NOT NULL,
                    user_id             VARCHAR(64)  NOT NULL,
                    user_name           VARCHAR(64),
                    major               VARCHAR(128),
                    research_interests  TEXT,
                    methodology         TEXT,
                    keywords            TEXT,
                    known_terms         TEXT,
                    updated_at          VARCHAR(32),
                    last_confirmed_ts   DOUBLE DEFAULT 0,
                    UNIQUE KEY uq_channel_user (channel_id, user_id)
                )
            """)
            cur.execute(f"""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = '{table_name}'
                  AND COLUMN_NAME = 'channel_id'
            """)
            if cur.fetchone()[0] == 0:
                cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN channel_id VARCHAR(64) NOT NULL DEFAULT '' FIRST")
                print(f"DEBUG: 已自动为 {table_name} 添加 channel_id 列")
            # 自动补列：检查 keywords 是否存在，旧表升级时自动 ALTER
            cur.execute(f"""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = '{table_name}'
                  AND COLUMN_NAME = 'keywords'
            """)
            if cur.fetchone()[0] == 0:
                cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN keywords TEXT")
                print(f"DEBUG: 已自动为 {table_name} 添加 keywords 列")
            cur.execute(f"""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = '{table_name}'
                  AND COLUMN_NAME = 'known_terms'
            """)
            if cur.fetchone()[0] == 0:
                cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN known_terms TEXT")
                print(f"DEBUG: 已自动为 {table_name} 添加 known_terms 列")
            cur.execute(f"""
                SELECT COUNT(*) FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = '{table_name}'
                  AND COLUMN_NAME = 'last_confirmed_ts'
            """)
            if cur.fetchone()[0] == 0:
                cur.execute(f"ALTER TABLE `{table_name}` ADD COLUMN last_confirmed_ts DOUBLE DEFAULT 0")
                print(f"DEBUG: 已自动为 {table_name} 添加 last_confirmed_ts 列")

            cur.execute(f"SHOW INDEX FROM `{table_name}`")
            indexes = cur.fetchall()
            single_user_unique_indexes = []
            composite_exists = False
            index_columns: dict[str, list[str]] = {}
            for row in indexes:
                key_name = row[2]
                non_unique = row[1]
                column_name = row[4]
                index_columns.setdefault(key_name, []).append(column_name)
                if key_name == "uq_channel_user" and non_unique == 0:
                    composite_exists = True

            for key_name, columns in index_columns.items():
                if key_name == "PRIMARY":
                    continue
                if columns == ["user_id"]:
                    single_user_unique_indexes.append(key_name)

            for key_name in single_user_unique_indexes:
                cur.execute(f"ALTER TABLE `{table_name}` DROP INDEX `{key_name}`")
                print(f"DEBUG: 已移除 {table_name} 上仅按 user_id 唯一的索引 {key_name}")

            if not composite_exists:
                cur.execute(f"ALTER TABLE `{table_name}` ADD UNIQUE KEY uq_channel_user (channel_id, user_id)")
                print(f"DEBUG: 已自动为 {table_name} 添加复合唯一索引 uq_channel_user")
        conn.commit()
        conn.close()

    def load(self, user_id, channel_id, table_name="user_profile"):
        """读取单个频道内用户 Profile，不存在时返回 None。"""
        normalized_channel_id = self._normalize_channel_id(channel_id)
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT channel_id, user_id, user_name, major, research_interests, methodology, keywords, known_terms, updated_at, last_confirmed_ts "
                f"FROM `{table_name}` WHERE channel_id = %s AND user_id = %s",
                (normalized_channel_id, user_id)
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        cid, uid, uname, major, ri, meth, kw, known_terms, updated, confirmed_ts = row
        return {
            "channel_id": cid,
            "user_id": uid,
            "user_name": uname or "",
            "major": major or "",
            "research_interests": json.loads(ri) if ri else [],
            "methodology": json.loads(meth) if meth else [],
            "keywords": json.loads(kw) if kw else [],
            "known_terms": json.loads(known_terms) if known_terms else [],
            "updated_at": updated or "",
            "last_confirmed_ts": float(confirmed_ts or 0),
        }

    def load_all(self, channel_id=None, table_name="user_profile"):
        normalized_channel_id = self._normalize_channel_id(channel_id)
        conn = self._conn()
        with conn.cursor() as cur:
            if normalized_channel_id:
                cur.execute(
                    f"SELECT user_id FROM `{table_name}` WHERE channel_id = %s",
                    (normalized_channel_id,)
                )
            else:
                cur.execute(f"SELECT channel_id, user_id FROM `{table_name}`")
            rows = cur.fetchall()
        conn.close()
        results = []
        for row in rows:
            if normalized_channel_id:
                uid = row[0]
                p = self.load(uid, normalized_channel_id, table_name)
            else:
                cid, uid = row
                p = self.load(uid, cid, table_name)
            if p:
                results.append(p)
        return results

    def save(self, profile, channel_id, table_name="user_profile"):
        """新增或更新 Profile（按 channel_id + user_id UPSERT）。"""
        normalized_channel_id = self._normalize_channel_id(channel_id)
        p = {
            "channel_id": normalized_channel_id,
            "user_name": "",
            "major": "",
            "research_interests": [],
            "methodology": [],
            "keywords": [],
            "known_terms": [],
            "updated_at": str(date.today()),
            "last_confirmed_ts": 0.0,
        }
        p.update(profile)
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO `{table_name}`
                    (channel_id, user_id, user_name, major, research_interests, methodology, keywords, known_terms, updated_at, last_confirmed_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    channel_id         = VALUES(channel_id),
                    user_name          = VALUES(user_name),
                    major              = VALUES(major),
                    research_interests = VALUES(research_interests),
                    methodology        = VALUES(methodology),
                    keywords           = VALUES(keywords),
                    known_terms        = VALUES(known_terms),
                    updated_at         = VALUES(updated_at),
                    last_confirmed_ts  = last_confirmed_ts
            """, (
                normalized_channel_id,
                p["user_id"],
                p["user_name"],
                p["major"],
                json.dumps(p["research_interests"], ensure_ascii=False),
                json.dumps(p["methodology"], ensure_ascii=False),
                json.dumps(p.get("keywords") or [], ensure_ascii=False),
                json.dumps(p.get("known_terms") or [], ensure_ascii=False),
                p["updated_at"],
                float(p.get("last_confirmed_ts") or 0.0),
            ))
        conn.commit()
        conn.close()

    def get_known_terms(self, user_id: str, channel_id: str, table_name="user_profile") -> list[str]:
        profile = self.load(user_id=user_id, channel_id=channel_id, table_name=table_name)
        if not profile:
            return []
        terms = []
        for term in profile.get("known_terms") or []:
            clean = str(term or "").strip().lower()
            if clean and clean not in terms:
                terms.append(clean)
        return terms

    def add_known_term(self, user_id: str, channel_id: str, term: str, table_name="user_profile") -> bool:
        clean_term = (term or "").strip().lower()
        if not clean_term:
            return False
        profile = self.load(user_id=user_id, channel_id=channel_id, table_name=table_name)
        if not profile:
            profile = {
                "user_id": user_id,
                "user_name": "",
                "major": "",
                "research_interests": [],
                "methodology": [],
                "keywords": [],
                "known_terms": [],
            }
        known_terms = list(profile.get("known_terms") or [])
        normalized = [str(t or "").strip().lower() for t in known_terms]
        if clean_term in normalized:
            return False
        known_terms.append(clean_term)
        profile["known_terms"] = known_terms
        self.save(profile=profile, channel_id=channel_id, table_name=table_name)
        return True

    def mark_profile_confirmed(self, user_id, channel_id, confirmed_ts=None, table_name="user_profile"):
        """在用户确认画像后，记录频道内画像的确认时间戳。"""
        import time

        ts = float(confirmed_ts or time.time())
        normalized_channel_id = self._normalize_channel_id(channel_id)
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE `{table_name}` SET last_confirmed_ts = %s WHERE channel_id = %s AND user_id = %s",
                (ts, normalized_channel_id, user_id),
            )
        conn.commit()
        conn.close()

    @staticmethod
    def format_for_prompt(profiles):
        """将多个 Profile 格式化为自然语言段落，供 LLM Prompt 使用。"""
        lines = []
        for p in profiles:
            name = p.get("user_name") or p.get("user_id", "未知")
            major = p.get("major") or "未知专业"
            interests = "、".join(p.get("research_interests") or []) or "暂无"
            methods = "、".join(p.get("methodology") or []) or "暂无"
            lines.append(
                f"用户 {name}：专业为{major}，研究兴趣包括{interests}，擅长{methods}。"
            )
        return "\n".join(lines)