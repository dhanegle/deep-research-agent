# 贡献指南

感谢关注本项目！欢迎以下形式的贡献：

- 🐛 Bug 报告与修复
- 📖 文档改进
- ✨ 新功能（请先开 Issue 讨论方案，避免方向跑偏后白写）
- 📊 评估补充（新的评测任务、消融配置、指标）

## 开发环境

```bash
git clone https://github.com/dhanegle/deep-research-agent.git
cd deep-research-agent
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/Mac

cp .env.example .env   # 默认 local 搜索源，无需任何 API key 即可跑通
```

## 提交前检查

```bash
# 1. 单元测试必须全过（纯逻辑，不依赖 Ollama / 网络）
.venv\Scripts\python -m pytest tests/ -v

# 2. 实跑一次确认端到端不回归（需要 Ollama 运行 qwen2.5:3b）
.venv\Scripts\python -m agent "2025年国产新能源汽车出口情况" --provider local
```

## 代码约定

- **零框架原则**：不引入 LangChain / LlamaIndex 等 Agent 框架，工具循环、结构化输出、
  上下文压缩保持手写实现——这是本项目的核心命题，新增依赖需要充分理由。
- **受限输出哲学**：面向小模型的设计，任何新的模型交互都应优先考虑
  "把开放推理降级为受限选择"（结构化输出 + pydantic 校验 + 规则兜底），
  而不是指望模型自觉遵守指令。
- **校验降级而非丢弃**：护栏（引用校验、扣题校验）的失败动作应是保留原文或重试，
  不能把已生成的有效内容转成占位符——这是实测踩过的坑（见 README"为什么这些设计"表）。
- **测试锁定行为**：修 bug 或改行为时同步更新 tests/，每个防御性逻辑（容错匹配、
  过滤规则、校验降级）都有对应测试用例。
- **改动留痕**：涉及管线行为的修改请跑一次评估（`eval/run_eval.py`），
  在 PR 中附上前后的指标对比。

## 提交规范

- commit message 用中文或英文均可，格式 `type: 摘要`，
  type 取 `feat` / `fix` / `docs` / `test` / `refactor` / `eval`。
- 一个 PR 聚焦一件事，相关性过滤、写作校验这类"防御性逻辑"的改动
  请在描述里附 trace 或测试输出作为证据。
