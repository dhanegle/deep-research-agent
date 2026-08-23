# Deep Research Agent — 基于 Ollama + Qwen2.5-3B 的本地深度调研智能体

用 **本地 3B 小模型** 跑通 OpenAI Deep Research 同款形态的调研 Agent：
输入一个调研问题，自动完成 **规划 → 联网搜索 → 网页阅读 → 反思补搜 → 分节成文**，
产出带引用来源的 Markdown 调研报告。**零框架依赖**（不用 LangChain/LlamaIndex），
Agent 循环、工具调用、结构化输出约束、评估体系全部手写实现。

> 核心命题：大模型人人会调，**把 3B 小模型的工程可靠性做上来**才是本项目的亮点——
> 所有架构决策都围绕"把开放推理降级为受限选择"展开，并用数据量化每项手段的收益。

## 架构

```
调研问题
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ A 规划 Planner                                               │
│   JSON Schema 约束 + pydantic 校验 + 失败回喂修复重试          │
│   → 大纲 3-5 节 + 初始搜索词 2-4 个                          │
├─────────────────────────────────────────────────────────────┤
│ B 信息收集 Researcher（ReAct 工具循环，核心）                  │
│   Ollama 原生 tool calling，仅 2 个单参数工具：               │
│     web_search(query) / read_page(url)                       │
│   防线1 非法调用 → 错误回喂自修复                             │
│   防线2 连续失败 → 降级 JSON-ReAct（提示词动作选择）           │
│   防线3 整轮无产出 → 规则兜底采集（确定性脚本）                │
│   正文分块 → 150字/块 LLM 摘要 → 知识库（上下文压缩）          │
├─────────────────────────────────────────────────────────────┤
│ C 反思 Reflector                                             │
│   「信息够不够」降级为 sufficient + gap_queries 受限输出       │
│   覆盖度规则兜底：任一小节相关来源 <2 条 → 强制补搜            │
├─────────────────────────────────────────────────────────────┤
│ D 写作 Writer                                                │
│   分节生成（每节只注入相关性 top-5 摘要）                      │
│   引用标记 [n] 仅允许指向真实抓取过的来源，幻觉引用自动剔除     │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
reports/*.md 调研报告   +   traces/*.jsonl 全链路追踪
```

## 快速开始

```bash
# 1. 安装依赖（需要 Python 3.10+，Ollama 已运行 qwen2.5:3b）
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/Mac

# 跑单元测试（纯逻辑，不依赖 Ollama/网络）
.venv\Scripts\python -m pytest tests/ -v

# 2. 配置
cp .env.example .env        # 两种搜索源：local（本地文件夹，默认）/ tavily（联网）

# 3. 本地文档调研：把你的 md/txt/pdf/docx 文档放进 data/（LOCAL_DOCS_DIR 可改），然后
.venv\Scripts\python -m agent "2025年国产新能源汽车出口情况" --provider local

# 4. Web UI（推荐）：浏览器打开 http://127.0.0.1:8000
#    输入问题 → 实时看到搜索/阅读/反思/写作每一步 → 渲染完整报告与指标
.venv\Scripts\python -m agent.web

# 5. 真实联网调研（注册 tavily.com 免费 key，每月 1000 次）
#    .env 中：SEARCH_PROVIDER=tavily  TAVILY_API_KEY=tvly-xxx
.venv\Scripts\python -m agent "固态电池产业化最新进展"
```

### 两种搜索源（同一 SearchProvider 接口，可插拔）

| 搜索源 | 说明 |
|---|---|
| `local` | 把指定文件夹（默认 `data/`）当语料库：关键词（长词自动滑动切分）对文件名+正文打分排序；支持 md/txt/pdf/docx（txt 自动兼容 GBK）；返回相对路径作为"链接" |
| `tavily` | 真实联网搜索，英文场景召回优，结果落盘快照供评估回放 |
| `bocha` | 博查 AI 搜索（国内），中文召回优于 Tavily，结果含内容摘要 |
```

## 评估体系

三层设计，保证 **可复现、可对比、可量化**：

1. **磁盘快照回放**：搜索与网页抓取按内容哈希落盘。首次 `CACHE_MODE=record`
   真实请求并快照；此后 `CACHE_MODE=replay` 全程读缓存——零 API 消耗、不依赖网络、
   结果逐位可复现，是提示词对比实验的基础设施。
2. **评测集**：`eval/tasks.jsonl` 每任务附要点评分表（must_cover 关键词）。
3. **指标**：要点覆盖率 / 结构化输出首试成功率 / 工具调用有效率 / 引用合规 / 引用忠实度与密度 / 耗时。

```bash
# 首轮：真实搜索并落盘快照（消耗 Tavily 额度，一次性）
CACHE_MODE=record SEARCH_PROVIDER=tavily .venv/Scripts/python eval/run_eval.py

# 之后：回放对比实验（改提示词/关反思等，零成本复跑）
CACHE_MODE=replay .venv/Scripts/python eval/run_eval.py --tag exp2
CACHE_MODE=replay .venv/Scripts/python eval/run_eval.py --no-reflect --tag baseline
```

### 实测指标（local 搜索源 2 任务 / 本机 qwen2.5:3b）

| 指标 | 数值 | 说明 |
|---|---|---|
| 任务成功率 | 2/2 | 全流程无人工干预 |
| 要点覆盖率 | 100%（9/9 关键词） | 评分表关键词全部命中 |
| 结构化输出首试成功率 | 100% | 早期版本实测曾低至 50%，修复重试兜底后始终 100% 通过 |
| 工具调用有效率 | 100% | 调优前曾因路径幻觉低至 39%，参数命名与容错匹配修复后见右表 |
| 单份报告 LLM 调用 | 平均 27 次 | 含规划/搜索决策/摘要/反思/分节写作 |
| 单份报告耗时 | 平均 ~39s | 消费级 GPU |

> 工程调优的量化对比：工具参数由 `url` 改名 `path` + 路径容错匹配后，
> 同一任务工具有效率 **39% → 100%**（详见"为什么这些设计对 3B 模型是必要的"）。
> 接入真实搜索后请用 `run_eval.py` 重跑并替换本表。

### 消融实验（4 配置 × 2 任务 × 3 次重复，local 语料）

复现：`bash eval/ablation.sh && python eval/summarize_ablation.py`

| 配置 | 成功率 | 覆盖率 | 工具调用有效率 | LLM 调用 | 耗时 |
|---|---|---|---|---|---|
| **full**（完整版） | 6/6 | 100% | 86% | 24.2 | 36s |
| no-reflect（去反思） | 6/6 | 100% | 100% | 15.0 | 27s |
| prompt-react（提示词版工具调用） | 6/6 | 97% | **55%** | 30.5 | 31s |
| naive（朴素提示词+无行为约束） | **5/6** | **83%** | 74% | 21.7 | **106s** |

三个结论：

1. **原生 tool calling 完胜提示词方案**：工具有效率 86% vs 55%——修复重试虽能兜住最终成功率，但每次非法调用都要多一轮"错误回喂+重试"（LLM 调用多 26%）；
2. **行为约束是必要的**：去掉提示词规则与去重/预算后，成功率掉到 5/6、覆盖率 83%，且搜索空转使耗时近 3 倍（106s）；
3. **反思阶段不是免费的**：在语料充足、首轮即可收齐资料的简单任务上，反思是纯开销（+9 次调用）——它的价值在首轮采集不足的稀疏话题上（联网调研 linuxsb 时即靠补搜轮拿回了 4 条来源）。这个负结果同样有价值：说明"反思"应当由覆盖度规则按需触发，而不是默认全开。

### 引用忠实度校验（faithfulness）

引用合规只保证 `[n]` 指向真实存在的来源（格式层）；忠实度校验进一步问**内容层**问题：
被引来源的摘要真的支持那句话吗？`run_eval` 默认执行（`--no-faith` 跳过），两层实现（`eval/faithfulness.py`）：

- **规则层（零成本）**：抽取句中数字与被引摘要比对，找不到依据的记"数字存疑"。
  型号编号（Qwen2.5 / GPT-4 / 5G）与题目自带数字（如年份）不算；允许四舍五入容差
  （来源 106.9 万 ↔ 句子"约107万"不误报）；
- **LLM 层（低成本）**：数字存疑句优先、其余按序补足（上限 12 句/份），判
  支持 / 部分支持 / 不支持，产出 `faithfulness = (支持 + 0.5×部分支持) / 已判定`。
  判定集偏向可疑句，因此是对报告的**保守估计**。

local 语料 2 任务 × 多轮实测：**被引用句忠实度 100%（11/11 支持）、数字存疑 0 句**；
但**引用密度只有 0~14%（0/21 ~ 9/66 句带引用）**——忠实度只说明"被引用的部分没编造"，
3B 写作的引用行为本身不稳定（偶尔通篇不标引用），大量论断无从溯源。
这是评估体系暴露出的下一个改进点（写作提示词强化引用要求），数据如实记录在此。

## 目录结构

```
agent/
  llm.py          # Ollama 封装：chat / chat_json(校验+修复) / chat_text
  planner.py      # 阶段A 结构化规划
  researcher.py   # 阶段B ReAct 循环 + 三条防线 + 分块摘要
  reflector.py    # 阶段C 受限反思 + 覆盖度规则兜底
  writer.py       # 阶段D 分节写作 + 幻觉引用剔除
  knowledge.py    # 知识库：去重收录、按节相关性选材（字符二元组）
  pipeline.py     # 四阶段编排 + 运行指标
  trace.py        # JSONL 全链路追踪
  cli.py          # rich 实时步骤展示
  search/         # SearchProvider 接口 / 本地文件夹检索 / Tavily / 磁盘缓存 / 正文提取
eval/
  tasks.jsonl     # 评测任务集（带要点评分表）
  run_eval.py     # 回放评估入口（默认含引用忠实度校验，--no-faith 跳过）
  metrics.py      # 覆盖率与汇总指标
  faithfulness.py # 引用忠实度：规则层筛数字 + LLM 裁决，产出 faithfulness / 引用密度
tests/            # 纯逻辑单元测试（JSON 修复、引用校验、路径容错、相关性过滤、忠实度规则层、指标）
.github/workflows/ci.yml  # GitHub Actions：push/PR 自动跑测试
```

## 为什么这些设计对 3B 模型是必要的

| 3B 的典型失败 | 本项目的对策 | 量化入口 |
|---|---|---|
| 工具调用格式错乱/编造参数 | 少工具(2个)、单参数、错误回喂自修复、JSON-ReAct 降级 | tool_valid_rate / react_fallbacks |
| 给路径编造 `https://` 前缀 | 参数命名 `url`→`path` + 提示词统一"路径"措辞 + 前缀剥离后缀容错匹配（实测同一任务 39%→100%） | tool_valid_rate |
| 搜索引擎返回无关结果（查专有名词返回泛主题页） | 结果相关性过滤：查询 token 加权覆盖率≥0.35 + **ASCII 专有名词必须命中**（分隔符归一，linuxsb↔linux.sb）+ 按相关性重排；规则兜底采集同样过滤 | trace 中 kept/results |
| 反复搜索同一关键词而不阅读（实测单轮空转 282 次） | 同轮查询去重直接顶回未读链接 + 单轮 12 次搜索预算（282→4 次） | searches / tool_calls |
| JSON 输出偶尔不合法 | format=json + pydantic 校验 + 失败重试 | json_first_try_rate |
| 多步规划跑偏 | 四阶段流水线，把规划拆成单次受限生成 | 覆盖率 |
| 读 1 条就"宣布完成" | min_sources 约束 + nudge 顶回 + 反思补搜 | sources 数 / reflect_rounds |
| 没有时间概念，"今年/最新"按训练截止期理解 | 每次 LLM 调用注入当前日期（`llm.py`）；"今年/去年"进管线前确定性替换为具体年份 | 搜索词年份 / 报告标题 |
| 长上下文注意力衰减 | 网页→150字摘要，每节只注入 top-5 相关摘要 | token 用量 |
| 引用幻觉 | 引用编号白名单校验，写后剔除；评估侧另有忠实度校验兜内容层 | 报告引用合规 / faithfulness、引用密度 |

## 硬件适配实测（RTX 3050 Laptop / 4GB 显存）

| 模型 | 体积(Q4) | 显存驻留 | 实测单次调用延迟 | 结论 |
|---|---|---|---|---|
| qwen2.5:3b | 1.9GB | ✅ 全量进显存 | ~1-2s | 本项目基准配置，40s/报告 |
| qwen3:4b | 2.5GB | ❌ 溢出到内存，GPU/CPU 反复腾挪 | 3~98s 剧烈波动（真实负载 ~95s） | 不可用：单报告预计 45min+ |

两个附带发现（已写入代码）：
- 4GB 显存是本项目的硬边界：模型 + KV 缓存 + 显示占用三者合计不能超，qwen3:4b 差一步之遥；
- ollama 0.32 的 `think=False` 参数会把 Qwen3 思维链错路由进 `content`，官方软开关 `/no_think` 才可靠——`llm.py` 已按模型族自动追加软开关，未来在更大显存机器上切换 qwen3 无需改代码。

## Roadmap

- [x] Web UI：FastAPI + SSE 实时展示 Agent 每一步（`python -m agent.web`）
- [x] 消融实验（原生 tool call vs 提示词 ReAct / 有无反思 / 朴素基线，见上表）
- [x] 引用忠实度自动校验（规则层 + LLM 裁决，产出 faithfulness / 引用密度指标）
- [x] 单元测试 + CI（纯逻辑：JSON 修复、引用校验、路径容错、相关性过滤、忠实度规则层；GitHub Actions）
- [ ] 写作引用密度提升：3B 写作的引用行为不稳定（0~14%），待提示词迭代
- [x] 接入博查搜索（国内 AI 搜索，中文召回优于 Tavily，实现 SearchProvider 接口）
