import os
import openai
import time
import backoff
import html2text
import concurrent.futures
import requests
import re

from agents.scholar_retriever import retrieve_top_paragraphs
from config import settings

text_extractor = html2text.HTML2Text()
text_extractor.ignore_links = True
text_extractor.ignore_images = True
text_extractor.ignore_tables = True
text_extractor.ignore_emphasis = True


class CoSearchAgent:
    def __init__(
        self,
        search_engine,
        api_key,
        model_name=None,
        fallback_model_name=None,
        api_base=None,
        prompt_dir="prompts/en_complete_agent",
        temperature=0,
        n=1,
    ):
        self.api_key = api_key
        self.api_base = api_base or settings.openai_api_base
        self.initialize_openai()

        self.model_name = model_name or settings.llm_model_name
        self.fallback_model_name = fallback_model_name or settings.llm_fallback_model_name
        self.temperature = temperature
        self.n = n
        self.prompt_dir = prompt_dir
        assert os.path.exists(prompt_dir)

        self.search_engine = search_engine

    def initialize_openai(self):
        openai.api_key = self.api_key
        openai.api_base = self.api_base

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def generate_openai_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                enable_thinking=False,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                n=self.n
            )["choices"][0]["message"]["content"]
        except openai.error.InvalidRequestError as e:
            msg = str(e)
            if "does not support http call" in msg.lower() or "enable_thinking" in msg.lower():
                print(f"[DEBUG][llm] 主模型请求失败，回退到 {self.fallback_model_name}: {e}")
                response = openai.ChatCompletion.create(
                    model=self.fallback_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    n=self.n
                )["choices"][0]["message"]["content"]
            else:
                raise
        response = "\n".join(response.split("\n\n"))
        return response

    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def generate_openai_response_stream(self, prompt, chunk_callback=None):
        """
        流式生成（兼容旧版 openai 库 + 阿里云 qwen）。
        每收到一个 token 调用 chunk_callback(accumulated_text)。
        返回完整文本。
        """
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            n=1,
            stream=True,
        )
        full_text = ""
        for chunk in response:
            # 旧版 openai 库流式返回的是 OpenAIObject，需用 get
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_text += token
                if chunk_callback:
                    try:
                        chunk_callback(full_text)
                    except Exception:
                        pass  # 回调失败不中断生成
        # 保持与非流式一致的换行处理
        return "\n".join(full_text.split("\n\n")) if "\n\n" in full_text else full_text

    def generate_agent_response(self, prompt_type, placeholders, replacements):
        prompt = self.load_prompt_from_file(prompt_type, placeholders, replacements)
        #print(prompt)
        return self.generate_openai_response(prompt)

    def generate_agent_response_stream(self, prompt_type, placeholders, replacements,
                                        chunk_callback=None):
        prompt = self.load_prompt_from_file(prompt_type, placeholders, replacements)
        return self.generate_openai_response_stream(prompt, chunk_callback=chunk_callback)

    def load_prompt_from_file(self, prompt_type, placeholders, replacements):
        prompt_file = os.path.join(self.prompt_dir, f"{prompt_type}.txt")
        assert os.path.exists(prompt_file)

        with open(prompt_file, "r", encoding="utf8") as file:
            prompt = file.read()

        for placeholder, replacement in zip(placeholders, replacements):
            prompt = prompt.replace(placeholder, replacement)

        return prompt

    def classify_intent(self, query, convs):
        """意图分类，返回 (label, elapsed)，如 ("【选题】", 0.5)"""
        print(f"[DEBUG][classify_intent] query={query!r}")
        start_time = time.time()
        q = (query or "").strip()
        conv_text = (convs or "")

        # 规则优先：总结请求直接命中，避免被模型误判为【其他】
        summary_keywords = ("总结", "归纳", "梳理", "汇总", "回顾", "复盘", "今天的内容", "今天内容")
        if any(k in q for k in summary_keywords):
            elapsed = time.time() - start_time
            print(f"[DEBUG][classify_intent] 规则命中总结，耗时={elapsed:.2f}s")
            return "【总结】", elapsed

        # 规则优先：明确入门/路线类提问，且对话中没有自报专业背景时，直接走【专业解释】；
        # 若已有"我是X专业"等背景信息，交由 LLM 判断（需区分本域还是跨域）。
        # 典型场景："如何入门人工智能"、"学习路线"。
        learning_pattern = re.compile(r"(怎么|如何).{0,8}入门|入门.{0,4}(方法|建议|路径|指南)|学习.{0,8}(路线|路径|规划)")
        if learning_pattern.search(q):
            has_background = ("我是" in conv_text) or ("我学" in conv_text) or ("专业" in conv_text)
            if not has_background:
                elapsed = time.time() - start_time
                print(f"[DEBUG][classify_intent] 规则命中学习型提问（无背景），归类专业解释，耗时={elapsed:.2f}s")
                return "【专业解释】", elapsed
            # 有背景时交由 LLM 判断是跨域还是本域

        output = self.generate_agent_response(
            "intent_classification",
            ["[query]", "[convs]"],
            [query, convs]
        ).strip()

        # 仅保留标准标签，兼容模型输出带解释文本的情况
        match = re.search(r"【(选题|分工|专业解释|知识解答|判断|总结|其他)】", output)
        label = match.group(0) if match else "【其他】"

        elapsed = time.time() - start_time
        print(f"[DEBUG][classify_intent] 原始={output!r} 标签={label!r} 耗时={elapsed:.2f}s")
        return label, elapsed

    def extract_user_profile(self, user, convs):
        """
        从输入文本中增量提炼目标用户的学术背景。

        输入 convs 的推荐格式（由 profile_watcher 组装）：
          【已确认的用户画像】
          专业：...  研究兴趣：...  擅长方法：...  关键词：...

          【新增对话记录（仅该用户自己的发言）】
          用户名: 发言内容
          ...

        LLM 应只输出相对于已有画像的新增或变更内容。
        禁止从 Bot 回答中提取任何信息（Bot 内容不会出现在输入中）。

        返回 (profile_dict, elapsed)，解析失败时返回空dict。
        profile_dict 键：major / research_interests / methodology / keywords
        """
        import json as _json
        print(f"[DEBUG][extract_user_profile] 目标用户={user}")
        print(f"[DEBUG][extract_user_profile] 输入行数={len(convs.splitlines())} 前200字:\n{convs[:200]}")
        start_time = time.time()
        output = self.generate_agent_response(
            "extract_user_profile",
            ["[user]", "[convs]"],
            [user, convs]
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG][extract_user_profile] LLM原始输出({elapsed:.2f}s): {output[:300]}")
        try:
            # 兼容 LLM 输出带有 markdown 代码块的情况
            clean = output.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:])
            if clean.endswith("```"):
                clean = "\n".join(clean.split("\n")[:-1])
            profile = _json.loads(clean.strip())
            print(f"[DEBUG][extract_user_profile] 解析成功: major={profile.get('major')} "
                  f"interests={profile.get('research_interests')} "
                  f"methods={profile.get('methodology')} "
                  f"keywords={profile.get('keywords')}")
        except Exception as e:
            print(f"[DEBUG][extract_user_profile] JSON解析失败({e})，返回空dict")
            profile = {}
        profile.setdefault("user_id", user)
        profile.setdefault("user_name", user)
        return profile, elapsed

    def extract_user_profile_incremental(self, user: str, convs: str,
                                          existing_major: str = "") -> tuple:
        """
        增量画像提取：只从新对话中提取 research_interests / methodology / keywords，
        不重新生成 major（major 已由用户确认锁定）。
        返回 (dict, elapsed_time)
        """
        import time, json, re
        t0 = time.time()

        prompt = (
            f"请从以下用户发言中，识别新出现的学术背景信息。\n"
            f"注意：该用户的专业已确认为【{existing_major or '未知'}】，请勿修改专业字段。\n"
            f"只提取以下三类新增信息（若无则返回空列表）：\n"
            f"1. research_interests: 新提及的研究兴趣方向\n"
            f"2. methodology: 新提及的研究方法或技能\n"
            f"3. keywords: 新提及的研究关键词\n\n"
            f"用户发言：\n{convs}\n\n"
            f"请严格按JSON格式返回，不要输出其他内容：\n"
            f'{{\"major\": \"{existing_major}\", \"research_interests\": [], '
            f'\"methodology\": [], \"keywords\": []}}'
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            elapsed = time.time() - t0
            print(f"[DEBUG][extract_user_profile_incremental] LLM输出({elapsed:.2f}s): {raw[:200]}")

            # 解析 JSON
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                result = json.loads(m.group())
                # 强制锁定 major
                result["major"] = existing_major
                return result, elapsed
        except Exception as e:
            print(f"[DEBUG][extract_user_profile_incremental] ⚠ 失败: {e}")

        return {"major": existing_major, "research_interests": [], "methodology": [], "keywords": []}, time.time() - t0

    def rewrite_topic_query(self, query, convs, user_profiles=""):
        """
        选题场景查询改写：结合结构化Profile和对话上下文生成4条互联网检索词。
        user_profiles: UserProfileMemory.format_for_prompt() 输出的自然语言字符串。
        返回 (search_queries列表, elapsed)
        """
        print(f"[DEBUG][rewrite_topic_query] query={query!r}")
        print(f"[DEBUG][rewrite_topic_query] user_profiles=\n{user_profiles[:300] if user_profiles else '(空)'}")
        start_time = time.time()
        output = self.generate_agent_response(
            "rewrite_topic_query",
            ["[query]", "[convs]", "[user_profiles]"],
            [query, convs, user_profiles]
        )
        elapsed = time.time() - start_time
        search_queries = [q.strip() for q in output.strip().split("\n") if q.strip()]
        print(f"[DEBUG][rewrite_topic_query] 生成检索词({elapsed:.2f}s): {search_queries}")
        return search_queries, elapsed

    def propose_topics(self, query, convs, user_profile, references):
        """
        RAG选题生成：严格基于检索到的参考文献生成带引用的选题建议。
        对应 prompts/ch/topic_recommendation.txt
        返回 (topics_text, elapsed)
        """
        print(f"[DEBUG][propose_topics] query={query!r}")
        print(f"[DEBUG][propose_topics] references字符数={len(references)} 预览:\n{references[:400]}")
        start_time = time.time()
        output = self.generate_agent_response(
            "topic_recommendation",
            ["[query]", "[convs]", "[user_profile]", "[references]"],
            [query, convs, user_profile, references]
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG][propose_topics] 输出({elapsed:.2f}s):\n{output[:500]}")
        return output, elapsed

    def propose_topics_stream(self, query, convs, user_profile, references,
                               client_slack, channel_id, user_id,
                               search_status_lines: list[str] = None,
                               search_results: list[dict] = None):
        """
        流式选题生成。
        search_results: 原始搜索结果列表，用于生成底部来源卡片。
        返回 (full_text, elapsed, response_ts)
        """
        from utils import send_rag_references, slack_chat_update
        import time

        print(f"[DEBUG][propose_topics_stream] 开始流式选题")
        start_time = time.time()

        search_summary = ""
        if search_status_lines:
            search_summary = "📚 *已完成文献搜索：*\n" + "\n".join(search_status_lines) + "\n\n⏳ 正在生成选题建议…\n"

        init_resp = client_slack.chat_postMessage(
            channel=channel_id,
            text=search_summary + "▌",
        )
        ts = init_resp["ts"]

        last_update_len = 0
        UPDATE_INTERVAL = 80

        def on_chunk(accumulated):
            nonlocal last_update_len
            if len(accumulated) - last_update_len >= UPDATE_INTERVAL:
                last_update_len = len(accumulated)
                try:
                    slack_chat_update(
                        client=client_slack,
                        channel=channel_id,
                        ts=ts,
                        text=search_summary + accumulated + "▌",
                    )
                except Exception as e:
                    print(f"[DEBUG][propose_topics_stream] ⚠ chat_update失败: {e}")

        full_text = self.generate_agent_response_stream(
            "topic_recommendation",
            ["[query]", "[convs]", "[user_profile]", "[references]"],
            [query, convs, user_profile, references],
            chunk_callback=on_chunk,
        )
        elapsed = time.time() - start_time

        # ★ 最终更新：去掉光标，保留纯文本
        try:
            slack_chat_update(
                client=client_slack,
                channel=channel_id,
                ts=ts,
                text=search_summary + full_text,
            )
        except Exception as e:
            print(f"[DEBUG][propose_topics_stream] ⚠ 最终chat_update失败: {e}")

        # ★ 追加来源卡片（单独发一条消息，保持与原 send_rag_answer 一致的体验）
        if search_results:
            try:
                send_rag_references(
                    client=client_slack,
                    channel_id=channel_id,
                    query=query,
                    user_id=user_id,
                    references=search_results,
                )
                print(f"[DEBUG][propose_topics_stream] 来源卡片已追加")
            except Exception as e:
                print(f"[DEBUG][propose_topics_stream] ⚠ 来源卡片发送失败: {e}")

        print(f"[DEBUG][propose_topics_stream] 完成({elapsed:.2f}s)")
        return full_text, elapsed, ts

    def propose_division(self, query, convs, user_profiles, refs=""):
        """
        分工建议生成：结合对话历史中的选题内容和用户画像，输出阶段性任务分工与时间计划。
        对应 prompts/ch/task_division.txt
        占位符：[query][convs][user_profiles][refs]
        refs: 格式化后的参考资料文本（标题+摘要+来源），作为任务生成的唯一依据
        返回：(division_text, elapsed_time)
        """
        print(f"[DEBUG][propose_division] query={query!r}")
        print(f"[DEBUG][propose_division] user_profiles=\n{user_profiles[:300] if user_profiles else '(空)'}")
        print(f"[DEBUG][propose_division] refs字符数={len(refs)} 预览:\n{refs[:300] if refs else '(空)'}")
        start_time = time.time()
        output = self.generate_agent_response(
            "task_division",
            ["[query]", "[convs]", "[user_profiles]", "[refs]"],
            [query, convs, user_profiles, refs]
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG][propose_division] 输出({elapsed:.2f}s):\n{output[:500]}")
        return output, elapsed

    def propose_division_stream(self, query, convs, user_profiles, refs,
                                 client_slack, channel_id, user_id,
                                 search_status_lines: list[str] = None,
                                 search_results: list[dict] = None):
        """
        流式分工生成。
        search_results: 原始搜索结果列表，用于生成底部来源卡片。
        返回 (full_text, elapsed, response_ts)
        """
        from utils import send_rag_references, slack_chat_update
        import time

        print(f"[DEBUG][propose_division_stream] 开始流式分工")
        start_time = time.time()

        search_summary = ""
        if search_status_lines:
            search_summary = "🔍 *已完成资料搜索：*\n" + "\n".join(search_status_lines) + "\n\n⏳ 正在生成分工方案…\n"

        init_resp = client_slack.chat_postMessage(
            channel=channel_id,
            text=search_summary + "▌",
        )
        ts = init_resp["ts"]

        last_update_len = 0
        UPDATE_INTERVAL = 80

        def on_chunk(accumulated):
            nonlocal last_update_len
            if len(accumulated) - last_update_len >= UPDATE_INTERVAL:
                last_update_len = len(accumulated)
                try:
                    slack_chat_update(
                        client=client_slack,
                        channel=channel_id,
                        ts=ts,
                        text=search_summary + accumulated + "▌",
                    )
                except Exception as e:
                    print(f"[DEBUG][propose_division_stream] ⚠ chat_update失败: {e}")

        full_text = self.generate_agent_response_stream(
            "task_division",
            ["[query]", "[convs]", "[user_profiles]", "[refs]"],
            [query, convs, user_profiles, refs],
            chunk_callback=on_chunk,
        )
        elapsed = time.time() - start_time

        try:
            slack_chat_update(
                client=client_slack,
                channel=channel_id,
                ts=ts,
                text=search_summary + full_text,
            )
        except Exception as e:
            print(f"[DEBUG][propose_division_stream] ⚠ 最终chat_update失败: {e}")

        # ★ 追加来源卡片
        if search_results:
            try:
                send_rag_references(
                    client=client_slack,
                    channel_id=channel_id,
                    query=query,
                    user_id=user_id,
                    references=search_results,
                )
                print(f"[DEBUG][propose_division_stream] 来源卡片已追加")
            except Exception as e:
                print(f"[DEBUG][propose_division_stream] ⚠ 来源卡片发送失败: {e}")

        print(f"[DEBUG][propose_division_stream] 完成({elapsed:.2f}s)")
        return full_text, elapsed, ts

    def rewrite_query(self, query, user, convs):
        start_time = time.time()
        output = self.generate_agent_response("rewrite_query", ["[query]", "[user]", "[convs]"],
                                              [query, user, convs])
        return output, time.time() - start_time

    def generate_search_query(self, query, convs, profiles):
        """
        为选题场景生成面向学术搜索的检索词。
        profiles: 字符串，描述各用户的学术背景，例如：
                  "用户A（Ben）：商业法方向，研究兴趣包括公司治理；用户B（Amy）：计算机科学方向，研究兴趣包括NLP"
        返回：(检索词列表, 耗时)，每条检索词对应一次独立搜索
        """
        start_time = time.time()
        output = self.generate_agent_response(
            "generate_search_query",
            ["[query]", "[convs]", "[profiles]"],
            [query, convs, profiles]
        )
        # 每行一条检索词，过滤空行
        search_queries = [q.strip() for q in output.split("\n") if q.strip()]
        return search_queries, time.time() - start_time

    def ask_clarify_query(self, query, user, convs):
        start_time = time.time()
        output = self.generate_agent_response("ask_clarify_query", ["[query]", "[user]", "[convs]"],
                                              [query, user, convs])
        return output, time.time() - start_time

    def rewrite_professional_explain_query(self, query: str, convs: str, term: str = "") -> tuple:
        """
        为专业解释场景生成更适合 Google Scholar 的检索词。
        返回 (raw_output, elapsed)，输出格式：
        检索思路：...
        检索词：...
        """
        start_time = time.time()
        prompt = (
            "你是学术检索词改写助手。目标：把用户口语化的术语求解释问题，改写成 1 条适合 Google Scholar 的检索词。\n"
            "只输出两行：\n"
            "检索思路：...\n"
            "检索词：...\n\n"
            "规则：\n"
            "1) 去掉‘啥是/什么是/是什么东西/啊/呀/能不能解释一下’等口语表达。\n"
            "2) 只保留最核心的 1 个术语或概念，不要写成完整问句。\n"
            "3) 若术语是常见英文缩写，优先补充标准全称，格式可写为 Full Name (ABBR)。\n"
            "4) 优先输出定义、综述、教程、基础介绍类检索词，避免过宽泛，也避免太口语。\n"
            "5) 若上下文能明确术语所指，可利用上下文消歧；否则保守输出术语标准名。\n\n"
            f"候选术语：{term or '无'}\n"
            f"当前问题：{query}\n"
            f"最近对话：\n{convs}\n"
        )
        output = self.generate_openai_response(prompt)
        return output, time.time() - start_time

    def fetch_webpage_source(self, url):
        try:
            # 添加浏览器请求头，避免被反爬虫
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        
            response = requests.get(url, headers=headers, timeout=10)  # 添加超时
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误 ({url}): {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"连接错误 ({url}): {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"超时错误 ({url}): {e}")
            return None
        except Exception as e:
            print(f"抓取错误 ({url}): {e}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP错误 ({url}): {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"连接错误 ({url}): {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"超时错误 ({url}): {e}")
            return None
        except Exception as e:
            print(f"抓取错误 ({url}): {e}")
            return None

    def extract_raw_text(self, webpage):
        try:
            # 清理可能导致解析错误的内容
            if webpage:
                # 移除脚本和样式标签
                import re
                webpage = re.sub(r'<script[^>]*>.*?</script>', '', webpage, flags=re.DOTALL | re.IGNORECASE)
                webpage = re.sub(r'<style[^>]*>.*?</style>', '', webpage, flags=re.DOTALL | re.IGNORECASE)
        
            raw_text = text_extractor.handle(webpage)
            return raw_text
        except Exception as e:
            print(f"HTML解析错误: {e}")
            # 尝试简单的文本提取
            try:
                import re
                text = re.sub(r'<[^>]+>', '', webpage)  # 简单移除所有HTML标签
                return text
            except:
                return None

    def extract_reference(self, query, link, max_text_length=5000):
        webpage = self.fetch_webpage_source(link)
        if not webpage:
            return None

        raw_text = self.extract_raw_text(webpage)
        if not raw_text:
            return None

        raw_text = "\n".join(raw_text.split("\n\n"))
        truncated_text = raw_text[:max_text_length]
        reference = self.generate_agent_response("extract_reference", ["[query]", "[document]"],
                                                 [query, truncated_text])
        if "不存在摘要" in reference or "None" in reference:
            return None

        return reference

    def generate_answer(self, query):
        """普通问答场景：用单条query搜索并生成答案。"""
        search_results = self.search_engine.get_search_results(query)

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            references = list(
                executor.map(lambda result: self.extract_reference(query, result["link"]), search_results))
        extract_time = time.time() - start_time

        search_results = [
            {"title": search_result["title"], "link": search_result["link"], "snippet": ref}
            for search_result, ref in zip(search_results, references) if ref
        ]
        references = [ref for ref in references if ref]
        references = "\n".join([f"[{i + 1}] {ref}" for i, ref in enumerate(references)])

        start_time = time.time()
        answer = self.generate_agent_response("generate_answer", ["[query]", "[references]"],
                                              [query, references])
        answer_time = time.time() - start_time

        return answer, search_results, extract_time, answer_time

    def generate_topic_answer(self, query, search_queries, convs, profiles):
        """
        选题建议场景：
        1. 用多条学术检索词分别搜索，合并去重结果（最多采用前20条结果）
        2. 抽取摘要时同时传入原始 query 保持相关性判断的语义锚点
        3. 调用 topic_recommendation prompt 生成带引用的选题建议
        """
        print(f"[DEBUG][generate_topic_answer] query={query!r}")
        print(f"[DEBUG][generate_topic_answer] 检索词列表({len(search_queries)}条): {search_queries}")

        # 多条检索词分别搜索，合并去重（按 link 去重），最多保留20条
        seen_links = set()
        all_search_results = []
        for sq in search_queries:
            print(f"[DEBUG][generate_topic_answer] 正在搜索: {sq!r}")
            results = self.search_engine.get_search_results(sq)
            added = 0
            for r in results:
                if r["link"] not in seen_links and len(all_search_results) < 20:
                    seen_links.add(r["link"])
                    all_search_results.append(r)
                    added += 1
            print(f"[DEBUG][generate_topic_answer]   → 本轮新增 {added} 条, 累计 {len(all_search_results)} 条")

        print(f"[DEBUG][generate_topic_answer] 搜索阶段完成，共 {len(all_search_results)} 条待抽取")

        # 用原始 query 作为摘要提取的语义锚
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            references = list(
                executor.map(
                    lambda result: self.extract_reference(query, result["link"]),
                    all_search_results
                )
            )
        extract_time = time.time() - start_time
        valid_count = sum(1 for r in references if r)
        print(f"[DEBUG][generate_topic_answer] 摘要抽取完成({extract_time:.2f}s): "
              f"成功={valid_count} 失败={len(references)-valid_count}")

        # 过滤掉无关摘要
        filtered_results = [
            {"title": r["title"], "link": r["link"], "snippet": ref}
            for r, ref in zip(all_search_results, references) if ref
        ]
        references_text = "\n".join([
            f"[{i+1}] 标题：{r['title']}\n摘要：{r['snippet']}\n来源：{r['link']}"
            for i, r in enumerate(filtered_results)
        ])
        print(f"[DEBUG][generate_topic_answer] 最终参考文献条数={len(filtered_results)}")
        print(f"[DEBUG][generate_topic_answer] references_text预览:\n{references_text[:600]}")

        # 生成选题答案（topic_recommendation prompt）
        start_time = time.time()
        answer = self.generate_agent_response(
            "topic_recommendation",
            ["[query]", "[convs]", "[user_profile]", "[references]"],
            [query, convs, profiles, references_text]
        )
        answer_time = time.time() - start_time
        print(f"[DEBUG][generate_topic_answer] 选题输出({answer_time:.2f}s):\n{answer[:600]}")

        return answer, filtered_results, extract_time, answer_time

    def summarize_convs(self, query: str, convs: str) -> str:
        """
        对完整对话记录生成结构化总结（共识 / 行动项 / 讨论状态）。
        对应 prompts/ch/summary.txt
        返回 (summary_text, elapsed)
        """
        print(f"[DEBUG][summarize_convs] query={query!r} 对话行数={len(convs.splitlines())}")
        start_time = time.time()
        output = self.generate_agent_response(
            "summary",
            ["[query]", "[convs]"],
            [query, convs]
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG][summarize_convs] 输出({elapsed:.2f}s):\n{output[:400]}")
        return output, elapsed

    def classify_summary_granularity(self, query: str) -> tuple:
        import time as _time
        print(f"[DEBUG][classify_summary_granularity] query={query!r}")
        t0 = _time.time()

        # 规则兜底：如果是“整体/目前/今天”的总览请求，即使提到多个维度
        # （如选题、分工、下一步），也应走粗度总结而不是专项总结。
        q = (query or "").strip().lower()
        overview_tokens = ["整体", "总体", "目前", "当前", "今天", "进展", "现状", "全局", "整体上", "目前的"]
        dimension_tokens = ["选题", "分工", "安排", "任务", "下一步", "推进", "计划", "研究"]
        has_overview = any(tok in q for tok in overview_tokens)
        dim_hit_count = sum(1 for tok in dimension_tokens if tok in q)
        if has_overview and dim_hit_count >= 2:
            elapsed = _time.time() - t0
            print(
                "[DEBUG][classify_summary_granularity] 规则命中："
                f"overview={has_overview} dim_hit_count={dim_hit_count} -> broad"
            )
            return "broad", "", elapsed

        prompt = (
            "请判断以下总结请求是【细度总结】还是【粗度总结】。\n\n"
            "【细度总结】：用户明确指定了某个具体话题或方向，例如：\n"
            "  - \"总结一下RAG相关的内容\"\n"
            "  - \"帮我总结选题部分的讨论\"\n"
            "  - \"把分工那块总结一下\"\n\n"
            "【粗度总结】：用户没有指定具体话题，泛指整体，例如：\n"
            "  - \"总结一下今天的\"\n"
            "  - \"帮我总结一下\"\n"
            "  - \"总结我们的讨论\"\n\n"
            "补充规则：若请求包含整体视角词（如：目前/今天/整体/当前），并同时提到多个维度"
            "（如：选题、分工、下一步任务），应判定为【粗度总结】；这类请求不是专项总结。\n\n"
            f"用户请求：{query}\n\n"
            "请按以下格式输出，不要输出其他内容：\n"
            "粒度：【细度总结】或【粗度总结】\n"
            "话题：[若为细度总结，提取核心话题词（5字以内）；若为粗度总结，输出空]"
        )

        try:
            raw = self.generate_openai_response(prompt)
            elapsed = _time.time() - t0
            print(f"[DEBUG][classify_summary_granularity] LLM输出({elapsed:.2f}s): {raw!r}")

            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            granularity = "broad"
            topic = ""
            for line in lines:
                if "细度总结" in line:
                    granularity = "focused"
                elif "粗度总结" in line:
                    granularity = "broad"
                if line.startswith("话题：") or line.startswith("话题:"):
                    topic = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            print(f"[DEBUG][classify_summary_granularity] granularity={granularity!r} topic={topic!r}")
            return granularity, topic, elapsed
        except Exception as e:
            print(f"[DEBUG][classify_summary_granularity] ⚠ 失败，默认粗度: {e}")
            return "broad", "", _time.time() - t0

    def summarize_focused(self, query: str, topic: str, focused_convs: str) -> tuple:
        """
        细度总结：针对指定话题生成专项总结。
        对应 prompts/ch/summary_focused.txt
        返回 (summary_text, elapsed)
        """
        import time as _time
        print(f"[DEBUG][summarize_focused] topic={topic!r} 相关行数={len(focused_convs.splitlines())}")
        t0 = _time.time()
        output = self.generate_agent_response(
            "summary_focused",
            ["[topic]", "[focused_convs]", "[query]"],
            [topic, focused_convs, query],
        )
        elapsed = _time.time() - t0
        print(f"[DEBUG][summarize_focused] 输出({elapsed:.2f}s):\n{output[:400]}")
        return output, elapsed

    def chitchat(self, query: str, convs: str) -> tuple:
        """
        【其他】意图：对闲聊/问候/测试等非任务消息给出简短友好的回复。
        对应 prompts/ch/chitchat.txt
        返回 (reply_text, elapsed)
        """
        print(f"[DEBUG][chitchat] query={query!r}")
        start_time = time.time()
        output = self.generate_agent_response(
            "chitchat",
            ["[query]", "[convs]"],
            [query, convs]
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG][chitchat] 回复({elapsed:.2f}s): {output[:200]}")
        return output, elapsed

    def _retrieve_paragraphs(self, query: str, serpapi_key: str) -> tuple:
        """
        调用 scholar_retriever 获取最相关段落。
        返回 (search_results, source)，其中 source 为 scholar 或 google_web。
        """
        print(f"[Retriever] 开始文献检索: {query!r}")
        t0 = time.time()
        search_results, source = retrieve_top_paragraphs(
            query=query,
            api_key=serpapi_key,
            scholar_num=20,
            max_paragraphs_per_pdf=150,
            top_k=20,
        )
        print(f"[Retriever] 检索完成，来源={source} 结果数={len(search_results)} 耗时={time.time()-t0:.2f}s")
        return search_results, source

    def _build_references_text(self, search_results: list[dict]) -> str:
        """
        将检索结果格式化为 prompt 参考文献文本。
        优先使用 paragraph 字段（PDF段落），否则回退 snippet。
        """
        refs = []
        for i, item in enumerate(search_results):
            content = item.get("paragraph") or item.get("snippet", "")
            refs.append(f"[{i + 1}] {content}")
        return "\n".join(refs)

    def run_retrieval_workflow(
        self,
        workflow_type: str,
        query: str,
        serpapi_key: str,
        user_profile: str = "",
        audience_model: str = "",
        escalation_context: str = "",
    ) -> tuple:
        """
        统一检索工作流。
        workflow_type 支持：knowledge_answer / professional_explain / judgment_analysis
        返回 (answer, search_results, source, elapsed)
        """
        start = time.time()
        search_results, source = self._retrieve_paragraphs(query=query, serpapi_key=serpapi_key)
        refs = self._build_references_text(search_results)

        if workflow_type == "professional_explain":
            answer = self.generate_agent_response(
                "professional_explain",
                ["[query]", "[references]", "[user_profile]", "[audience_model]"],
                [query, refs, user_profile, audience_model],
            )
        elif workflow_type == "judgment_analysis":
            answer = self.generate_agent_response(
                "judgment_analysis",
                ["[query]", "[references]", "[escalation_context]"],
                [query, refs, escalation_context],
            )
        else:
            answer = self.generate_agent_response(
                "knowledge_answer",
                ["[query]", "[references]"],
                [query, refs],
            )

        return answer, search_results, source, time.time() - start