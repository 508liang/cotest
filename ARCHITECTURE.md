# CoSearchAgent-master Architecture

## 1. 项目目标

`CoSearchAgent-master` 是一个面向 Slack 协作场景的研究助手系统，核心能力包括：

- 对 `@bot` 消息进行意图识别与检索增强回答（RAG）。
- 维护跨频道的用户学术画像（确认后写库，增量更新）。
- 支持选题、分工、总结、知识解答、专业解释、判断等任务。
- 提供 viewer（后端 API + 前端）用于查看频道、对话和画像数据。

## 2. 总体分层

```text
Slack Workspace
  -> Socket Mode (Slack Bolt)
    -> app.py -> zh_cosearch_agent_app.py (主业务编排)
      -> agents/ (LLM + 检索)
      -> handlers/ (意图处理器/画像确认/总结)
      -> memory/ (MySQL 持久化)
      -> utils.py (Slack消息拼装、用户/频道映射、通用函数)

MySQL
  <- memory/* + utils.py 读写
  <- backend/db_server.py 读取并对 viewer 暴露

viewer/
  -> 调用 backend/db_server.py 的 REST API 展示数据
```

## 3. 目录与职责

### 3.1 入口层

- `app.py`
  - 统一启动入口。
  - 创建 `SocketModeHandler` 并启动 Slack 事件循环。
- `zh_cosearch_agent_app.py`
  - 当前主入口（中文场景）。
  - 负责事件路由、意图分发、状态消息、画像监听触发、RAG 调用、写库。
- `en_cosearch_agent_app.py`
  - 英文入口，能力相对精简。

### 3.2 配置层

- `config.py`
  - 集中加载 `.env` / 环境变量。
  - 提供 `settings`（Slack、LLM、DB、SerpAPI）和启动校验。
  - 关键配置只在此处集中管理，避免散落硬编码。

### 3.3 智能能力层（agents）

- `agents/cosearch_agent.py`
  - LLM 交互封装：意图分类、改写、澄清、回答、画像提取等。
  - 支持主模型 + fallback 模型。
- `agents/search_engine.py`
  - 常规 Web 检索。
- `agents/scholar_retriever.py`
  - Scholar / Web fallback / 段落级 BM25 排序检索。

### 3.4 业务处理层（handlers）

- `handlers/topic_handler.py`：选题流程。
- `handlers/division_handler.py`：分工流程。
- `handlers/summary_handler.py`：总结流程。
- `handlers/profile_watcher.py`：画像后台增量监控。
- `handlers/profile_confirm.py`：画像确认/编辑/恢复挂起意图。
- `handlers/profile_utils.py`：画像比对、合并、变更判断。

### 3.5 持久化层（memory）

- `memory/cosearch_agent_memory.py`：频道对话主表读写。
- `memory/rag_results_memory.py`：检索结果分页状态。
- `memory/click_memory.py`：引用点击行为。
- `memory/user_profile_memory.py`：用户画像（含 `last_confirmed_ts`）。
- `memory/pending_intent_memory.py`：画像确认前挂起的意图。

### 3.6 可观测与运维层

- `backend/db_server.py`：viewer 后端 API（channels/profiles/convs/users）。
- `backend/backfill_channel_info.py`：离线回填 `channel_info`。
- `viewer/`：前端展示页面。

## 4. 核心运行流程

## 4.1 @bot 消息流程（主路径）

1. Slack 事件进入 `handle_message_event`。
2. 解析用户与频道，写入频道对话表（表名为 `channel_id`）。
3. 识别意图（选题/分工/总结/知识解答/专业解释/判断/其他）。
4. 分发到对应 handler 或 RAG 工作流。
5. 回复消息并落库（对话、检索结果、点击行为等）。

## 4.2 非 @bot 消息流程（当前策略）

1. 仅触发画像监听 `watch_profile_in_background`。
2. 不触发主动专业解释/判断检索（避免误触发）。

说明：当前代码中 `_handle_proactive_and_periodic` 仍保留在主入口中，但默认未挂接调用，属于可选能力保留。

## 4.3 用户画像增量流程

1. 读取该用户历史画像和 `last_confirmed_ts`。
2. 从频道对话表筛选“确认后新增证据”。
3. 仅抽取画像相关发言（本人 + 他人提及）。
4. 调用 LLM 提炼画像草稿。
5. 规则补丁将“我对X感兴趣”等短语直接并入兴趣字段（防止 LLM 泛化丢词）。
6. 若有增量，发送确认卡片；用户确认后再写入画像并更新时间戳。

## 4.4 检索类三意图流程（专业解释 / 知识解答 / 判断）

三类意图在主入口统一走 `_run_retrieval_intent`，核心步骤一致，prompt 与附加上下文不同。

### 4.4.1 统一主链路

1. 意图分类得到 `【知识解答】/【专业解释】/【判断】`。
2. 在主入口将意图映射为 workflow：
  - `【知识解答】 -> knowledge_answer`
  - `【专业解释】 -> professional_explain`
  - `【判断】 -> judgment_analysis`
3. 调用 `agent.run_retrieval_workflow(...)`：
  - 先用 `scholar_retriever` 检索并做段落级排序；
  - 再将检索结果拼成 `references` 文本；
  - 最后按 workflow 选用对应 prompt 生成回答。
4. 用 `send_rag_answer` 发送主回复与引用。
5. 写入两类存储：
  - 对话主表 `{channel_id}`（记录 `workflow_type/source/infer_time`）；
  - 检索结果表 `{channel_id}_search`（供 next/previous 翻页）。

### 4.4.2 知识解答（knowledge_answer）

- 触发语义：用户希望获取事实、概念、背景信息。
- 输入上下文：`query + references`。
- 输出目标：给出基于证据的客观知识回答，突出可验证来源。
- 特点：不引入用户画像个性化调参，偏“标准学术问答”。

### 4.4.3 专业解释（professional_explain）

- 触发语义：用户对术语/概念“听不懂、要解释、是什么意思”等。
- 输入上下文：`query + references + user_profile + audience_model`。
- 输出目标：在保证专业性的同时降低理解门槛。
- 个性化策略：
  - `user_profile` 提供专业、兴趣、方法偏好；
  - `audience_model` 强调“先直觉后定义、控制术语密度、用用户熟悉领域类比”。

### 4.4.4 判断分析（judgment_analysis）

- 触发语义：对方案、路径、立场进行比较和取舍。
- 输入上下文：`query + references + escalation_context`。
- 输出目标：对备选项做可比较分析（优缺点、适用条件、建议）。
- 说明：`escalation_context` 用于约束输出风格，强调“可执行建议”。

### 4.4.5 三者差异总结

- 共性：都依赖同一检索引擎与引用链路，均可追溯证据。
- 差异：
  - 知识解答强调“事实回答”；
  - 专业解释强调“用户可理解性”；
  - 判断分析强调“比较与决策建议”。

## 5. 数据模型（MySQL）

## 5.1 关键系统表

- `user_info`
  - `user_id` -> `user_name` 映射。
- `channel_info`
  - `channel_id` -> `channel_name` 映射（viewer 展示名）。
- `user_profile`
  - `user_id`, `major`, `research_interests`, `methodology`, `keywords`, `last_confirmed_ts`。
- `pending_intent`
  - 画像确认前挂起意图参数。
- `seen_channels`
  - 首次见过频道记录。

## 5.2 动态业务表

- `{channel_id}`
  - 频道对话主表（speaker/utterance/query/clarify/search_results/timestamp...）。
- `{channel_id}_search`
  - 检索答案与分页状态。
- `{channel_id}_click`
  - 点击行为日志。

## 6. 配置与启动

## 6.1 配置来源

统一由 `config.py` 读取：

- Slack: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_BOT_ID`
- LLM: `OPENAI_API_KEY`, `OPENAI_API_BASE`, `LLM_MODEL_NAME`, `LLM_FALLBACK_MODEL_NAME`
- Search: `SERPAPI_KEY`
- DB: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`

## 6.2 启动命令

- Bot：`python app.py`
- Viewer 后端：`python backend/db_server.py`
- 回填频道名：`python backend/backfill_channel_info.py --dry-run`

## 7. 架构特性与设计决策

- 配置集中化：敏感信息和模型参数统一管理，便于迁移和运维。
- 画像确认写库：先提炼草稿再确认，降低误更新风险。
- 增量更新窗口：以 `last_confirmed_ts` 为基线提取新增证据。
- 频道隔离：选题/分工画像只使用当前频道活跃用户。
- 展示与存储分离：存储使用 `channel_id`，展示使用 `channel_info.channel_name`。

## 8. 当前技术债与优化建议

1. 主入口 `zh_cosearch_agent_app.py` 体量较大，建议拆分为 router/service/scheduler。
2. 动态表名 SQL 仍有安全风险，建议统一做表名白名单过滤。
3. `utils.py` 职责偏重（消息、映射、DB 辅助混合），建议分拆为 `slack_utils.py` / `db_utils.py`。
4. `_handle_proactive_and_periodic` 目前为保留逻辑，建议改为显式 feature flag 开关。
5. 建议补集成测试：
   - 非@消息不触发专业解释。
   - “我对X感兴趣”进入画像增量。
   - 新频道自动展示可读频道名。

## 9. 合并与协作建议

- 以 `app.py + zh_cosearch_agent_app.py + config.py` 作为主干。
- 新能力统一以 handler 扩展，避免继续堆入口文件。
- 任何新增配置先加 `config.py` 和 `.env.example`，再接入业务代码。
- 变更 viewer 展示逻辑时，优先保持后端 API 字段稳定（`name`, `display_name`, `count`）。
