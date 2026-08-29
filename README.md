# Deep Research Agent — 基于 Ollama + Qwen2.5-3B 的本地深度调研智能体

<p align="center">
  <a href="https://github.com/dhanegle/deep-research-agent/actions/workflows/ci.yml"><img src="https://github.com/dhanegle/deep-research-agent/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/LLM-ollama%20%7C%20qwen2.5--3b-8B5CF6" alt="Ollama">
  <img src="https://img.shields.io/badge/framework-零框架依赖-orange" alt="No framework">
</p>

用 **本地 3B 小模型** 跑通 OpenAI Deep Research 同款形态的调研 Agent：
输入一个调研问题，自动完成 **规划 → 联网搜索 → 网页阅读 → 反思补搜 → 分节成文 → 自审修订**，
产出带引用来源的 Markdown 调研报告。

> **核心命题：大模型人人会调，把 3B 小模型的工程可靠性做上来才是本项目的亮点。**
> 所有架构决策都围绕"把开放推理降级为受限选择"展开，
> Agent 循环、工具调用、结构化输出、评估体系全部手写实现（零 LangChain/LlamaIndex 依赖），
> 并用数据量化每项手段的收益。

## 为什么值得一看

- **双反思闭环**：搜完反思"资料够不够"（阶段C），写完自审"报告对不对"（阶段E）——占位节自动触发补搜+重写，数字存疑逐句裁决
- **三层可靠性防线**：非法工具调用回喂自修复 → JSON-ReAct 降级 → 规则确定性兜底，专治 3B 的参数幻觉与空转
- **搜索质量治理四件套**：大纲空词自动改写、权威域名加权、单位内部页面降权、题库/文档站黑名单
- **可复现评估**：搜索快照磁盘回放（零 API 消耗对比实验）+ 消融实验 + 引用忠实度双层校验
- **消费级显卡可跑**：4GB 显存的入门 GPU 即可全流程本地运行，`local` 搜索源零 API key 开箱即用

## 架构

```
调研问题
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ A 规划 Planner                                               │
│   JSON Schema 约束 + pydantic 校验 + 失败回喂修复重试          │
│   → 大纲 3-5 节 + 初始搜索词 2-4 个（禁止大纲式标题词）         │
├─────────────────────────────────────────────────────────────┤
│ B 信息收集 Researcher（ReAct 工具循环，核心）                  │
│   Ollama 原生 tool calling，仅 2 个单参数工具：               │
│     web_search(query) / read_page(path)                      │
│   防线1 非法调用 → 错误回喂自修复                             │
│   防线2 连续失败 → 降级 JSON-ReAct（提示词动作选择）           │
│   防线3 整轮无产出 → 规则兜底采集（确定性脚本）                │
│   查询词治理：大纲空词自动拼问题核心词改写                     │
│   结果治理：相关性过滤 + 权威加权 + 单位页面降权 + 黑名单      │
│   正文 → 250字 LLM 摘要 → 知识库（上下文压缩）                │
├─────────────────────────────────────────────────────────────┤
│ C 反思 Reflector（搜-反思闭环）                               │
│   「信息够不够」降级为 sufficient + gap_queries 受限输出       │
│   覆盖度规则兜底：任一小节相关来源 <2 条 → 强制补搜            │
├─────────────────────────────────────────────────────────────┤
│ D 写作 Writer                                                │
│   分节生成（每节只注入相关性 top-5 摘要）                      │
│   引用标记 [n] 白名单校验，幻觉引用自动剔除                    │
│   零引用 → 请求补编号；重写失败保留首稿（校验降级而非丢弃）     │
├─────────────────────────────────────────────────────────────┤
│ E 自审 Reviewer（写-反思闭环，对称于 C）                      │
│   ① 规则层(免费)：占位节可补 / 零引用 / 数字存疑              │
│   ② LLM 结构审查(1次)：确认问题 + 为占位节生成补搜词           │
│   ③ 逐句忠实度(3-6次)：存疑节逐句裁决，"不支持"才保留标记      │
│   → 触发补搜（接回 B）+ 重写问题节 → 再过一次引用校验          │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
reports/*.md 调研报告   +   traces/*.jsonl 全链路追踪
```

## 快速开始

**前置要求**：Python 3.10+，[Ollama](https://ollama.com) 已安装。

```bash
# 1. 拉模型（约 1.9GB）
ollama pull qwen2.5:3b

# 2. 安装依赖
git clone https://github.com/dhanegle/deep-research-agent.git
cd deep-research-agent
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/Mac

# 3. 配置（默认 local 搜索源，零 API key 开箱即跑）
cp .env.example .env

# 4. 跑第一个调研（检索 data/ 示例语料，不联网）
.venv\Scripts\python -m agent "2025年国产新能源汽车出口情况" --provider local

# 5. Web UI（推荐）：输入问题 → 实时看到每个阶段 → 渲染报告与指标
.venv\Scripts\python -m agent.web
# 浏览器打开 http://127.0.0.1:8000

# 6. 真实联网调研（三种搜索源见下表）
.venv\Scripts\python -m agent "固态电池产业化最新进展" --provider bocha
```

> 单元测试不依赖 Ollama 和网络，随时可跑：`.venv\Scripts\python -m pytest tests/ -v`

### 三种搜索源（同一 SearchProvider 接口，可插拔）

| 搜索源 | 说明 | 适合 |
|---|---|---|
| `local` | 把 `data/` 文件夹当语料库：关键词滑动切分对文件名+正文打分；支持 md/txt/pdf/docx | **零配置开箱体验**、离线场景 |
| `bocha` | [博查 AI 搜索](https://bochaai.com)（国内），中文召回好，直连无需代理 | **中文调研推荐** |
| `tavily` | [Tavily](https://tavily.com)，英文场景召回优，有免费额度 | 英文调研 |

联网源只需在 `.env` 里填对应 API key，Web UI 下拉框也可直接切换。

## 评估体系

三层设计，保证 **可复现、可对比、可量化**：

1. **磁盘快照回放**：搜索与网页抓取按内容哈希落盘。首次 `CACHE_MODE=record`
   真实请求并快照；此后 `CACHE_MODE=replay` 全程读缓存——零 API 消耗、不依赖网络、
   结果逐位可复现，是提示词对比实验的基础设施。
2. **评测集**：`eval/tasks.jsonl` 每任务附要点评分表（must_cover 关键词）。
3. **指标**：要点覆盖率 / 结构化输出首试成功率 / 工具调用有效率 / 引用忠实度与密度 / 耗时。

```bash
# 首轮：真实搜索并落盘快照（一次性，消耗搜索额度）
CACHE_MODE=record SEARCH_PROVIDER=bocha .venv/Scripts/python eval/run_eval.py

# 之后：回放对比实验（改提示词/关反思等，零成本复跑）
CACHE_MODE=replay .venv/Scripts/python eval/run_eval.py --tag exp2
CACHE_MODE=replay .venv/Scripts/python eval/run_eval.py --no-reflect --tag baseline
```

### 实测指标（local 搜索源 2 任务 / 本机 qwen2.5:3b）

| 指标 | 数值 | 说明 |
|---|---|---|
| 任务成功率 | 2/2 | 全流程无人工干预 |
| 要点覆盖率 | 100%（9/9 关键词） | 评分表关键词全部命中 |
| 结构化输出首试成功率 | 100% | 早期版本实测曾低至 50%，修复重试兜底后始终 100% |
| 工具调用有效率 | 100% | 调优前曾因路径幻觉低至 39% |
| 单份报告 LLM 调用 | 平均 27 次 | 含规划/搜索决策/摘要/反思/自审/分节写作 |
| 单份报告耗时 | 平均 ~40s | 消费级 GPU（4GB 显存） |

> 接入真实搜索后请用 `run_eval.py` 重跑并替换本表——这正是评估体系的设计目的。

### 消融实验（4 配置 × 2 任务 × 3 次重复，local 语料）

复现：`bash eval/ablation.sh && python eval/summarize_ablation.py`

| 配置 | 成功率 | 覆盖率 | 工具调用有效率 | LLM 调用 | 耗时 |
|---|---|---|---|---|---|
| **full**（完整版） | 6/6 | 100% | 86% | 24.2 | 36s |
| no-reflect（去反思） | 6/6 | 100% | 100% | 15.0 | 27s |
| prompt-react（提示词版工具调用） | 6/6 | 97% | **55%** | 30.5 | 31s |

## 为什么这些设计对 3B 模型是必要的

每一条都来自真实 trace 复盘（`traces/*.jsonl` 全链路留痕），不是拍脑袋：

| 3B 的典型失败 | 本项目的对策 | 量化入口 |
|---|---|---|
| 工具调用格式错乱/编造参数（把 query 写成 path） | 少工具(2个)、单参数、错误回喂自修复、JSON-ReAct 降级 | tool_valid_rate / react_fallbacks |
| 给路径编造 `https://` 前缀 | 参数命名 `url`→`path` + 前缀剥离后缀容错匹配（同一任务 39%→100%） | tool_valid_rate |
| 把小节标题原样当搜索词（「政策影响因素」） | 查询词与问题核心词无实义重合时自动拼接改写；planner 提示词禁止大纲式标题词 | trace 中 query 改写记录 |
| 口语词搜索只命中学校/单位通知页 | 官方口径词提示词规则 + 权威域名加权 + 单位内部页面降权 + 题库站黑名单 | 收录来源域名分布 |
| 反复搜索同一关键词而不阅读（实测单轮空转 282 次） | 同轮查询去重直接顶回未读链接 + 单轮 12 次搜索预算（282→4 次） | searches / tool_calls |
| JSON 输出偶尔不合法 | format=json + pydantic 校验 + 失败回喂重试 | json_first_try_rate |
| 读 1 条就"宣布完成" | min_sources 约束 + nudge 顶回 + 反思补搜 | sources 数 / reflect_rounds |
| 零引用被强制重写时摆烂输出"暂缺"，好正文被校验丢弃 | 校验降级而非丢弃：重写失败保留首稿，只剔幻觉引用 | 报告体量 / 暂缺节数 |
| 报告数字与来源对不上（幻觉） | 阶段E 规则层数字比对 + LLM 逐句裁决，"不支持"才触发重写 | faithfulness / num_flags |
| 没有时间概念，"今年/最新"按训练截止期理解 | 每次 LLM 调用注入当前日期；"今年/去年"进管线前确定性替换 | 搜索词年份 |
| 长上下文注意力衰减 | 网页→250字摘要，每节只注入 top-5 相关摘要 | token 用量 |
| 引用编号幻觉 | 引用编号白名单校验，写后剔除 | 报告引用合规 |

## 目录结构

```
agent/
  llm.py          # Ollama 封装：chat / chat_json(校验+修复) / chat_text / stream_text
  planner.py      # 阶段A 结构化规划（大纲+搜索词，禁大纲式标题词）
  researcher.py   # 阶段B ReAct 循环 + 三防线 + 查询词改写 + 结果治理 + 分块摘要
  reflector.py    # 阶段C 搜-反思闭环：受限反思 + 覆盖度规则兜底
  writer.py       # 阶段D 分节写作 + 引用合规 + 校验降级（保首稿）
  reviewer.py     # 阶段E 写-反思闭环：规则层→结构审查→逐句裁决→触发补搜重写
  factcheck.py    # 事实核查原语：数字抽取比对 / 句子提取 / LLM 逐句裁决
  knowledge.py    # 知识库：去重收录、按节相关性选材（4字窗口+二元组排序）
  pipeline.py     # 五阶段编排 + 运行指标
  trace.py        # JSONL 全链路追踪
  cli.py          # rich 实时步骤展示
  web.py          # FastAPI + SSE 实时 Web UI
  static/         # 前端单页
  search/         # SearchProvider 接口 / local / tavily / bocha / 磁盘缓存 / 正文提取
eval/
  tasks.jsonl     # 评测任务集（带要点评分表）
  run_eval.py     # 回放评估入口（默认含引用忠实度校验）
  metrics.py      # 覆盖率与汇总指标
  faithfulness.py # 忠实度评估编排（原语在 agent/factcheck.py，两层共用）
  ablation.sh     # 消融实验脚本
tests/            # 120+ 纯逻辑单元测试（JSON修复/引用校验/路径容错/查询改写/降权/自审/忠实度）
.github/workflows/ci.yml  # GitHub Actions：push/PR 自动跑测试
```

## 常见问题

**Q：报告里有的节写"（本节暂缺相关资料）"？**
设计如此（宁缺毋滥）：资料与该节确实无关时不硬写。阶段E 自审会在"占位但有可用资料"时自动补搜+重写，若仍占位说明现有搜索源确实没找到对口资料——换更具体的提问或换搜索源再试。

**Q：改了代码 Web UI 行为没变？**
Python 进程不会热重载，重启 Web 服务（Ctrl+C 后重新 `python -m agent.web`）。

**Q：一定要联网 / API key 吗？**
不用。`--provider local` 检索 `data/` 示例语料即可完整体验全流程；联网源（博查/Tavily）才需要 key。

**Q：为什么搜到的是小网站而不是权威来源？**
搜索质量治理已内置（官方口径提示词 + 权威加权 + 降权 + 黑名单）。若仍遇到，欢迎开 Issue 并附上 trace 文件——这正是全链路留痕的用途。

**Q：4GB 显存的入门显卡能跑吗？**
能，这正是项目的基准环境：qwen2.5:3b（Q4 量化约 1.9GB）可全量驻留 4GB 显存，单次调用 1-2s、约 40s/报告。模型 + KV 缓存 + 显示占用三者合计不超显存即可。

**Q：换 qwen3 这类更新的小模型行不行？**
显存是硬约束：qwen3:4b（Q4 约 2.5GB）加上 KV 缓存超出 4GB 显存，GPU/CPU 反复腾挪会导致单次调用延迟 3~98s 剧烈波动，实测不可用；显存充裕（8GB+）时改 `.env` 的 `MODEL` 即可尝试。另有一个已写入代码的坑：ollama 0.32 的 `think=False` 参数会把 Qwen3 思维链错路由进正文污染输出，`llm.py` 已按模型族自动改用官方软开关 `/no_think`，换模型无需改代码。

## 贡献

欢迎 Issue 与 PR！提交前请读 [CONTRIBUTING.md](CONTRIBUTING.md)——
特别是"零框架原则"与"校验降级而非丢弃"两条代码约定。

## License

[MIT](LICENSE)
