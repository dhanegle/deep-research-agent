"""阶段 A：调研规划。输出报告大纲与初始搜索词。

任务是受限生成（结构化输出 + pydantic 校验 + 修复重试），
Qwen2.5-3B 在这类任务上可靠性很高。
"""
from pydantic import BaseModel, Field, field_validator

from .llm import LLM


class Plan(BaseModel):
    outline: list[str] = Field(..., min_length=3, max_length=8)
    search_queries: list[str] = Field(..., min_length=2, max_length=8)

    @field_validator("outline", "search_queries")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        cleaned = [x.strip() for x in v if x and x.strip()]
        if not cleaned:
            raise ValueError("列表不能为空")
        return cleaned


SYSTEM_PROMPT = """\
你是一个严谨的调研规划助手。给定调研问题，请输出：
1. outline：报告大纲，3 到 5 节，每节标题为 4-12 字的名词短语，覆盖问题的不同方面\
（例如：现状数据、主要构成、驱动因素、挑战风险、趋势展望）。
2. search_queries：2 到 4 个适合搜索引擎的查询词，每个 8-15 字、单一主题、\
互相补充不重复，必须包含问题的核心关键词（年份、主题名词等限定词）。\
查统计总量、行业数据时用官方口径词加范围限定词（如「全国普通高校毕业生人数」\
「中国新能源汽车出口量 海关总署」），不用口语缩略（如「毕业总人数」）。\
禁止输出大纲式标题词——像「政策影响因素」「地域分布特点」这种没有主题限定的短语，\
搜索引擎只会返回题库和答疑网站，不会有权威报道。

只输出 JSON，格式：{"outline": ["...", "..."], "search_queries": ["...", "..."]}
"""

FEW_SHOT = """\
示例——
问题：2025 年国产新能源汽车出口情况
输出：{"outline": ["出口总量与增速", "主要出口市场", "头部车企表现", "挑战与风险", "未来趋势"], \
"search_queries": ["2025 中国新能源汽车出口量 数据", "中国新能源汽车 欧洲 东南亚 出口", "比亚迪 奇瑞 新能源 出口 2025"]}
"""


def plan_question(llm: LLM, question: str) -> Plan:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT + "\n现在请规划问题：" + question},
    ]
    plan = llm.chat_json(messages, Plan, tag="planner")
    # 宽松校验 + 代码截断，避免修复重试空转
    return Plan(outline=plan.outline[:5], search_queries=plan.search_queries[:4])
