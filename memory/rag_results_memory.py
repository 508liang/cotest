import pymysql
from config import settings


class SearchMemory:
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

    def read_table_contents_as_list(self, table_content):
        table_content = {
            "id": table_content[0],
            "user_name": table_content[1],
            "query": table_content[2],
            "answer": table_content[3],
            "search_results": table_content[4],
            "start": table_content[5],
            "end": table_content[6],
            "timestamp": table_content[8]
        }
        return table_content

    def create_table_if_not_exists(self, table_name):
        create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_name VARCHAR(255) NOT NULL,
                        query VARCHAR(255) NOT NULL,
                        answer TEXT NOT NULL,
                        search_results TEXT NOT NULL,
                        start INT NOT NULL,
                        end INT NOT NULL,
                        click_time VARCHAR(255) NOT NULL,
                        timestamp VARCHAR(255) NOT NULL
                    );
        '''

        with self.connection.cursor() as cursor:
            cursor.execute(create_table_query)

        self.connection.commit()

    def write_into_memory(self, table_name, search_info):
        insert_query = f'''
            INSERT INTO {table_name} (user_name, query, answer, search_results, start, end, click_time, timestamp) 
                                      VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        '''

        with self.connection.cursor() as cursor:
            cursor.execute(insert_query, (
                search_info["user_name"], search_info["query"], search_info["answer"], search_info["search_results"],
                search_info["start"], search_info["end"], search_info["click_time"], search_info["timestamp"]))

        self.connection.commit()

    def load_search_results_from_timestamp(self, table_name, timestamp):
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"SELECT * FROM `{table_name}` WHERE timestamp = %s",
                (str(timestamp),)
            )
            table_contents = cursor.fetchall()
        finally:
            cursor.close()
            conn.close()

        if not table_contents:
            # timestamp 未命中，尝试取最近一条兜底
            print(f"[DEBUG][rag_results_memory] ⚠ timestamp={timestamp!r} 未命中 {table_name}，尝试最近一条兜底")
            conn2 = self.connect()
            cursor2 = conn2.cursor()
            try:
                cursor2.execute(
                    f"SELECT * FROM `{table_name}` ORDER BY click_time DESC LIMIT 1"
                )
                fallback = cursor2.fetchall()
            finally:
                cursor2.close()
                conn2.close()
            if not fallback:
                print(f"[DEBUG][rag_results_memory] ⚠ 表 {table_name} 无任何记录，返回 None")
                return None
            table_contents = fallback

        if len(table_contents) > 1:
            # 多条命中（不应出现）：取 click_time 最大的一条
            print(f"[DEBUG][rag_results_memory] ⚠ timestamp={timestamp!r} 命中 {len(table_contents)} 条，取最新一条")
            table_contents = sorted(table_contents, key=lambda r: r[7], reverse=True)

        return self.read_table_contents_as_list(table_contents[0])

