"""阶段 C：反思补搜。

把「信息是否足够」这类开放判断降级为受限输出（sufficient 布尔 + 补充搜索词），
并用覆盖度规则兜底：任何一节相关来源不足 2 条时，无论模型怎么判断都强制补搜。
"""
from pydantic import BaseModel, Field, field_validator

from .knowledge import KnowledgeBase
from .llm import LLM
from .planner import Plan


class Reflection(BaseModel):
    sufficient: bool
    gap_queries: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("gap_queries")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        return [q.strip() for q in v if q and q.strip()]


SYSTEM_PROMPT = """\
你是调研质量检查员，判断已收录资料是否足以撰写报告大纲的每一节。
只输出 JSON：{"sufficient": true 或 false, "gap_queries": ["补充搜索词", ...]}
信息足够时 sufficient 为 true 且 gap_queries 为空列表；不足时给 1-2 个补充搜索词。
要求：补充搜索词必须是完整的搜索引擎查询式，包含调研主题的核心关键词，\
每个不超过 15 字、单一主题（例如"中国新能源汽车出口 欧洲关税"），\
不能只写小节标题；语言必须与调研问题一致（中文问题给中文搜索词）。
判定标准：每节至少有 2 条相关资料，且事实性内容（数字、事件、机构）不互相缺失。
"""


def reflect(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase) -> Reflection:
    coverage_lines = [f"- {sec}：相关来源 {kb.coverage(sec, question)} 条" for sec in plan.outline]
    collected = "\n".join(f"[{s.id}] {s.title}" for s in kb.sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"调研问题：{question}\n大纲各节资料覆盖情况：\n"
            + "\n".join(coverage_lines)
            + f"\n已收录资料：\n{collected}"
        )},
    ]
    try:
        r = llm.chat_json(messages, Reflection, tag="reflector")
    except ValueError:
        r = Reflection(sufficient=True)  # 反思失败不阻塞主流程
    # 模型常给出 3-5 个补搜词：与其用严格 schema 反复触发修复重试（实测连败 3 次
    # 白烧 9 次调用），不如宽松校验后在代码里截断
    r.gap_queries = r.gap_queries[:2]

    # 规则兜底：覆盖不足时强制补搜，缺的搜索词从最薄弱的小节生成
    weak = [sec for sec in plan.outline if kb.coverage(sec, question) < 2]
    if weak and (r.sufficient or not r.gap_queries):
        gaps = (r.gap_queries or [])[:1] + [f"{question} {weak[0]}"][:1]
        r = Reflection(sufficient=False, gap_queries=gaps[:2])
    return r
