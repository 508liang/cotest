"""
memory/pending_intent_memory.py

存储用户画像待确认时挂起的意图状态。
用户点击"确认画像"后，从此表取出意图参数继续执行。

表结构：
    channel_id   : 频道ID（与 user_id 组成唯一键）
    user_id      : 用户Slack ID
  intent_label : 挂起的意图，如"【选题】"或"【分工】"
  payload      : JSON序列化的意图执行参数（query/convs/channel_id等）
  created_at   : 创建时间戳
"""

import json
import time
import pymysql
from config import settings


class PendingIntentMemory:
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

    def create_table_if_not_exists(self, table_name="pending_intent"):
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                    id           INT AUTO_INCREMENT PRIMARY KEY,
                    channel_id   VARCHAR(64) NOT NULL,
                    user_id      VARCHAR(64) NOT NULL,
                    intent_label VARCHAR(32),
                    payload      TEXT,
                    created_at   DOUBLE,
                    UNIQUE KEY uq_pending_channel_user (channel_id, user_id)
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

            cur.execute(f"SHOW INDEX FROM `{table_name}`")
            indexes = cur.fetchall()
            index_columns: dict[str, list[str]] = {}
            composite_exists = False
            for row in indexes:
                key_name = row[2]
                non_unique = row[1]
                column_name = row[4]
                index_columns.setdefault(key_name, []).append(column_name)
                if key_name == "uq_pending_channel_user" and non_unique == 0:
                    composite_exists = True

            for key_name, columns in index_columns.items():
                if key_name == "PRIMARY":
                    continue
                if columns == ["user_id"]:
                    cur.execute(f"ALTER TABLE `{table_name}` DROP INDEX `{key_name}`")
                    print(f"DEBUG: 已移除 {table_name} 上仅按 user_id 唯一的索引 {key_name}")

            if not composite_exists:
                cur.execute(f"ALTER TABLE `{table_name}` ADD UNIQUE KEY uq_pending_channel_user (channel_id, user_id)")
                print(f"DEBUG: 已自动为 {table_name} 添加复合唯一索引 uq_pending_channel_user")
        conn.commit()
        conn.close()

    def save(self, user_id: str, channel_id: str, intent_label: str, payload: dict,
             table_name="pending_intent"):
        """保存或覆盖频道内用户的挂起意图。"""
        normalized_channel_id = self._normalize_channel_id(channel_id)
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO `{table_name}` (channel_id, user_id, intent_label, payload, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    intent_label = VALUES(intent_label),
                    payload      = VALUES(payload),
                    created_at   = VALUES(created_at)
            """, (normalized_channel_id, user_id, intent_label, json.dumps(payload, ensure_ascii=False), time.time()))
        conn.commit()
        conn.close()

    def load(self, user_id: str, channel_id: str, table_name="pending_intent") -> dict | None:
        """读取挂起意图，不存在时返回 None。"""
        normalized_channel_id = self._normalize_channel_id(channel_id)
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT intent_label, payload FROM `{table_name}` WHERE channel_id = %s AND user_id = %s",
                (normalized_channel_id, user_id)
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        intent_label, payload_str = row
        return {"intent_label": intent_label, "payload": json.loads(payload_str)}

    def delete(self, user_id: str, channel_id: str, table_name="pending_intent"):
        """执行完成后清除挂起状态。"""
        normalized_channel_id = self._normalize_channel_id(channel_id)
        conn = self._conn()
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM `{table_name}` WHERE channel_id = %s AND user_id = %s",
                (normalized_channel_id, user_id)
            )
        conn.commit()
        conn.close()
