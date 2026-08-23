"""阶段 D：分节写作与引用合规。

- 分节生成而非一次成文：每次只注入该节最相关的少量摘要，上下文小且稳定；
- 引用标记 [n] 只允许指向知识库中真实存在的来源编号，生成后剔除幻觉引用。
- 写作走流式产出（llm.stream_text）：CLI 一次性拼接送进引用校验，
  Web UI 边收边经 SSE 推给前端逐字渲染。
"""
import re

from .knowledge import KnowledgeBase
from .llm import LLM
from .planner import Plan

CITE_RE = re.compile(r"\[(\d+)\]")

INTRO_PROMPT = (
    "你是调研报告撰写者。根据下方资料目录，为报告写一段 80-120 字的导语，"
    "概括本次调研覆盖了什么、主要发现是什么方向。不要使用引用标记，只输出导语正文。"
)

SECTION_PROMPT = """\
你是调研报告撰写者。只依据下方编号资料撰写指定小节，不得编造资料中没有的事实。要求：
1. 只写与本节标题直接相关的内容，其他方面的资料不要写进本节；
2. 从资料中归纳组织成连贯的分析，不要逐条罗列、不要整段照搬摘要；
3. 陈述事实时在句末标注资料编号（如 [1]），不要用资料标题或《》方式引用；
4. 200-400 字；不要输出小节标题，也不要在开头复述标题；不使用表格；
5. 如果资料中确实没有任何与本节标题相关的内容，只写：（本节暂缺相关资料），不要编造凑字。
"""


def _validate_citations(text: str, allowed_ids: set[int]) -> str:
    """剔除指向不存在来源的引用标记（幻觉引用防线）。"""
    def repl(m: re.Match) -> str:
        return m.group(0) if int(m.group(1)) in allowed_ids else ""
    return CITE_RE.sub(repl, text)


def _iter_report(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase):
    """生成器：逐段 yield 报告文本（标题→导语→各节正文→参考来源）。

    每节正文由 llm.stream_text 流式产出，调用方可选择边收边渲染
    （Web UI）或一次性拼接送进引用校验（CLI/eval）。
    """
    catalog = "\n".join(f"[{s.id}] {s.title}" for s in kb.sources)

    yield f"# {question}\n\n"
    for chunk in llm.stream_text([
        {"role": "system", "content": INTRO_PROMPT},
        {"role": "user", "content": f"报告主题：{question}\n资料目录：\n{catalog}"},
    ]):
        yield chunk
    yield "\n\n"

    for section in plan.outline:
        yield f"## {section}\n\n"
        sources = kb.select_for(section, question, k=5)
        if not sources:
            yield "（本节暂缺可用资料。）\n\n"
            continue
        material = "\n\n".join(f"[{s.id}] 《{s.title}》\n{s.digest}" for s in sources)
        for chunk in llm.stream_text([
            {"role": "system", "content": SECTION_PROMPT},
            {"role": "user", "content": f"报告主题：{question}\n本节标题：{section}\n\n编号资料：\n{material}"},
        ]):
            yield chunk
        yield "\n\n"

    yield "## 参考来源\n\n"
    for s in kb.sources:
        yield f"[{s.id}] {s.title} {s.url}\n\n"
    yield "\n"


def _validate_report(report: str, kb: KnowledgeBase) -> str:
    """剔除指向不存在来源的引用标记（每节正文独立处理，保留标题与参考来源页脚）。"""
    all_ids = {s.id for s in kb.sources}
    parts = report.split("## 参考来源", 1)
    body = parts[0]
    footer = ("## 参考来源" + parts[1]) if len(parts) > 1 else ""
    sections = body.split("## ")
    validated = [sections[0]] + [_validate_citations("## " + s, all_ids) for s in sections[1:]]
    return "".join(validated) + footer


def write_report(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase) -> str:
    """非流式：全量收集后做一次幻觉引用剔除。CLI / eval 走这条路。"""
    raw = "".join(_iter_report(llm, question, plan, kb))
    return _validate_report(raw, kb)


def iter_report(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase):
    """流式版：逐段 yield 文本片段，供 Web UI 经 SSE 逐字渲染。

    流式过程中无法中途剔除幻觉引用（已发出的 token 收不回），
    所以校验推迟到整篇完成后在落盘前做一次（见 pipeline._finalize_report）。
    前端观感的微小差异（个别无效引用标记）不影响演示，落盘报告始终是干净的。
    """
    yield from _iter_report(llm, question, plan, kb)
