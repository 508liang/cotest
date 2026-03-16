import pymysql
from config import settings


class Memory:
    def __init__(self, sql_password):
        self.sql_password = sql_password
        self.connection = self.connect()

    def connect(self):
        connection = pymysql.connect(
            host=settings.db_host,
            user=settings.db_user,
            passwd=self.sql_password,
            port=settings.db_port,
            db=settings.db_name,
        )
        return connection

    def create_table_if_not_exists(self, table_name):
        create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                speaker VARCHAR(255) NOT NULL,
                utterance TEXT NOT NULL,
                convs TEXT,
                query TEXT,
                rewrite_query TEXT,
                rewrite_thought TEXT,
                clarify TEXT,
                clarify_thought TEXT,
                clarify_cnt INT,
                search_results TEXT,
                infer_time TEXT,
                reply_timestamp VARCHAR(255),
                reply_user VARCHAR(255),
                timestamp VARCHAR(255) NOT NULL
            );
        '''

        with self.connection.cursor() as cursor:
            cursor.execute(create_table_query)

        self._migrate_legacy_columns(table_name)
        self.connection.commit()

    def _migrate_legacy_columns(self, table_name):
        """Upgrade legacy short text columns to TEXT to avoid Data too long errors."""
        alter_sql = f'''
            ALTER TABLE {table_name}
            MODIFY COLUMN query TEXT NULL,
            MODIFY COLUMN rewrite_query TEXT NULL,
            MODIFY COLUMN clarify TEXT NULL,
            MODIFY COLUMN infer_time TEXT NULL;
        '''
        with self.connection.cursor() as cursor:
            try:
                cursor.execute(alter_sql)
            except Exception as e:
                # Keep startup resilient for channels with restricted ALTER privileges.
                print(f"[DEBUG][memory] 列类型迁移跳过 table={table_name!r}: {e}")

    def write_into_memory(self, table_name, utterance_info):
        insert_query = f'''
            INSERT INTO {table_name} (speaker, utterance, convs, query, rewrite_query, rewrite_thought, 
            clarify, clarify_thought, clarify_cnt, search_results, infer_time, reply_timestamp, reply_user, timestamp) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);'''

        def _safe_text(value, max_len=60000):
            if value is None:
                return ""
            value = str(value)
            if len(value) > max_len:
                return value[:max_len]
            return value

        payload = (
            utterance_info["speaker"],
            _safe_text(utterance_info["utterance"]),
            _safe_text(utterance_info["convs"]),
            _safe_text(utterance_info["query"]),
            _safe_text(utterance_info["rewrite_query"]),
            _safe_text(utterance_info["rewrite_thought"]),
            _safe_text(utterance_info["clarify"]),
            _safe_text(utterance_info["clarify_thought"]),
            utterance_info["clarify_cnt"],
            _safe_text(utterance_info["search_results"]),
            _safe_text(utterance_info["infer_time"]),
            _safe_text(utterance_info["reply_timestamp"], max_len=255),
            _safe_text(utterance_info["reply_user"], max_len=255),
            _safe_text(utterance_info["timestamp"], max_len=255),
        )

        with self.connection.cursor() as cursor:
            cursor.execute(insert_query, payload)

        self.connection.commit()

    def get_clarify_cnt_for_speaker(self, table_name, reply_user):
        query = f"SELECT clarify_cnt FROM {table_name} WHERE reply_user = %s ORDER BY timestamp DESC LIMIT 1;"

        with self.connection.cursor() as cursor:
            cursor.execute(query, (reply_user,))
            table_contents = cursor.fetchall()

        if table_contents:
            assert len(table_contents) == 1
            return table_contents[0][0]
        else:
            return 0
        """
memory/cosearch_agent_memory.py 补丁说明
========================================
在 Memory 类中新增 load_all_utterances() 方法。
将以下方法粘贴到 Memory 类的末尾（与 write_into_memory、get_clarify_cnt_for_speaker 同级）。
"""

# ── 粘贴到 Memory 类末尾 ──────────────────────────────────────────────────────

    def load_all_utterances(self, table_name: str) -> list[dict]:
        """
        读取指定频道表的全量对话记录，按时间升序返回。
        每条记录返回 {"speaker": ..., "utterance": ..., "timestamp": ...} 字典。
        供【总结】功能使用，不受 Slack API limit 限制。
        """
        connection = pymysql.connect(
            host=settings.db_host,
            user=settings.db_user,
            passwd=self.sql_password,
            port=settings.db_port,
            db=settings.db_name,
        )
        try:
            with connection.cursor() as cursor:
                # 按 timestamp 升序，取 speaker 和 utterance 两列
                cursor.execute(
                    f"SELECT speaker, utterance, timestamp FROM `{table_name}` "
                    f"ORDER BY timestamp ASC"
                )
                rows = cursor.fetchall()
        except pymysql.err.ProgrammingError as e:
            # 1146: table does not exist. Keep profile watcher/readers resilient.
            if len(e.args) >= 1 and e.args[0] == 1146:
                print(f"[DEBUG][memory] 表 {table_name!r} 不存在，返回空记录")
                return []
            raise
        finally:
            connection.close()

        return [
            {"speaker": row[0], "utterance": row[1], "timestamp": row[2]}
            for row in rows if row[1]
        ]