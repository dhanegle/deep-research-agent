"""阶段 D：分节写作与引用合规。

- 分节生成而非一次成文：每次只注入该节最相关的少量摘要，上下文小且稳定；
- 引用标记 [n] 只允许指向知识库中真实存在的来源编号，生成后剔除幻觉引用；
- 导语按 token 流式产出；各节先在内部收齐（含零引用/跑题的回喂重写）再交出，
  发出去的正文已经过本节编号白名单校验。
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
1. 只写与本节标题直接相关的内容；资料与本节无关时不要硬写，只输出：（本节暂缺相关资料）
2. 从资料中归纳组织成连贯的分析，不要逐条罗列、不要整段照搬摘要；
3. 尽量在事实陈述句末标注资料编号（如 [1]）；不要用资料标题或《》方式引用；
4. 200-400 字；不要输出小节标题，也不要在开头复述标题；不使用表格；
5. 资料不足以支撑本节时，只写：（本节暂缺相关资料），不要用无关资料凑字、不要复述其他小节的内容。
"""

PLACEHOLDER = "（本节暂缺相关资料）"

REWRITE_PROMPT = (
    "你上一段没有标注任何资料编号。请重写本节，给每一句事实陈述末尾补上资料编号引用"
    "（写成[1]或[2]这种，数字必须是上方资料的真实编号）。"
    "不要输出小节标题，不要删除或缩减已有的事实内容，只补编号。"
)

ALIGN_PROMPT = (
    "你写的内容和本节标题对不上。请只写与本节标题直接相关的事实，句末用[1]或[2]标注；"
    "不要把其他小节的内容写进来。资料对不上标题就只输出：（本节暂缺相关资料）"
)


def _validate_citations(text: str, allowed_ids: set[int]) -> str:
    """剔除指向不存在来源的引用标记（幻觉引用防线）。"""
    def repl(m: re.Match) -> str:
        return m.group(0) if int(m.group(1)) in allowed_ids else ""
    return CITE_RE.sub(repl, text)


def _needs_rewrite(body: str) -> bool:
    """有实质内容却零引用 → 值得回喂重写一次。占位/空节不重写。"""
    text = body.strip()
    if not text or "暂缺" in text:
        return False
    return not CITE_RE.search(text)


def _section_aligned(body: str, section: str, question: str = "") -> bool:
    """正文是否扣题：含标题里至少一个二元组（短标题用 1 字）。

    不再要求标题 4 字连续窗口——SECTION_PROMPT 明确禁止复述标题，
    真实输出是释义不是照抄，4 字窗口会把几乎所有正常释义误判为跑题。
    二元组同样能抓住真跑题（"家电出口"写进"贸易伙伴"节时，贸易/伙伴/分析
    等二元组全不命中），但不会误杀"描述毕业生去向"这类对"毕业生总体情况"
    的合法释义。

    不减去 question 的二元组——相减会丢掉"毕业/业生"这种与问题共享、
    却正是释义里唯一能对上的词，把合法释义误判为跑题。
    """
    text = CITE_RE.sub("", body)
    if "暂缺" in text:
        return True
    from .knowledge import _bigrams
    terms = _bigrams(section)
    if not terms:  # 单字标题
        return section in text
    return any(t in text for t in terms)


def _clean_section_body(body: str) -> str:
    """3B 常把提示词里的字面 [n] 抄进正文；整节只是占位时归一为 PLACEHOLDER。

    只在"去掉暂缺短语后几乎不剩内容"时才整节丢弃——正文写满了却捎带一句
    "具体数字暂缺"不该把整节有效内容一起扔掉。
    """
    text = re.sub(r"\[n\]", "", body, flags=re.I).strip()
    if "暂缺" not in text:
        return text
    rest = re.sub(r"[（(][^）)]{0,30}暂缺[^）)]{0,30}[）)]", "", text)
    if len(re.sub(r"\s+", "", rest)) < 30:
        return PLACEHOLDER
    return text


def _material_ids(material: str) -> set[int]:
    """本节实际注入了哪些编号——只认资料条目行首的 [n]，不认摘要正文里的。"""
    return {int(m.group(1)) for m in re.finditer(r"^\[(\d+)\]", material, re.M)}


def _generate_section(llm: LLM, question: str, section: str, material: str,
                      allowed_ids: set[int] | None = None) -> str:
    """生成一节正文。校验降级而非丢弃——宁可留无引用/轻微偏题的正文，也不输出占位符。

    3B 模型很难稳定满足"每句带引用+精确扣题"，把这两条当硬约束会逼它摆烂输出占位符
    （实测 trace 里 5 节全是这条路）。改为：
    - 零引用 → 请求补编号重写一次；重写仍无引用就保留首稿（剥幻觉编号后），不转占位符；
    - 跑题 → 请求对齐重写一次；重写仍跑题才转占位符（真跑题宁可空着）。
    重写只发生在内部，发出去的正文已经过校验。
    """
    ids = _material_ids(material) if allowed_ids is None else allowed_ids

    def draft() -> str:
        return _validate_citations(_clean_section_body("".join(llm.stream_text(messages))), ids)

    messages = [
        {"role": "system", "content": SECTION_PROMPT},
        {"role": "user", "content": f"报告主题：{question}\n本节标题：{section}\n\n编号资料：\n{material}"},
    ]
    body = draft()
    if body == PLACEHOLDER:
        return body
    if _needs_rewrite(body):
        first = body
        messages.append({"role": "assistant", "content": body})
        messages.append({"role": "user", "content": REWRITE_PROMPT})
        body = draft()
        # 重写后仍无引用：保留首稿原文，不转占位符（内容比引用标记更重要）
        if body == PLACEHOLDER or _needs_rewrite(body):
            body = first
    if body != PLACEHOLDER and not _section_aligned(body, section, question):
        messages.append({"role": "assistant", "content": body})
        messages.append({"role": "user", "content": ALIGN_PROMPT})
        body = draft()
        if body != PLACEHOLDER and not _section_aligned(body, section, question):
            return PLACEHOLDER
    return body


def _iter_report(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase):
    """生成器：逐段 yield 报告文本（标题→导语→各节正文→参考来源）。

    导语仍按 token 流式产出；各节先在内部收齐（含可能的一次重写）再 yield，
    保证发出去的正文已经过引用检查。
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
            yield PLACEHOLDER + "\n\n"
            continue
        material = "\n\n".join(f"[{s.id}] 《{s.title}》\n{s.digest}" for s in sources)
        yield _generate_section(llm, question, section, material, {s.id for s in sources})
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


_SECTION_RE = re.compile(r"(^##\s+.+$)", re.M)


def _replace_section(report: str, title: str, new_body: str) -> str:
    """替换报告中指定节的正文（到下一个 ## 或参考来源为止），保留标题行。

    供阶段 E 自审修订使用：对某节重写后，用它把新正文换进报告原位置。
    """
    pattern = re.compile(
        r"(^##\s+" + re.escape(title) + r"\s*\n)(.*?)(?=^##\s|\Z)",
        re.M | re.S,
    )
    m = pattern.search(report)
    if not m:
        return report
    cleaned = _clean_section_body(new_body)
    replacement = m.group(1) + cleaned + "\n\n"
    return report[:m.start()] + replacement + report[m.end():]


def write_report(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase) -> str:
    """非流式：全量收集后做一次幻觉引用剔除。CLI / eval 走这条路。"""
    raw = "".join(_iter_report(llm, question, plan, kb))
    return _validate_report(raw, kb)


def iter_report(llm: LLM, question: str, plan: Plan, kb: KnowledgeBase):
    """流式版：逐段 yield 文本片段，供 Web UI 经 SSE 逐字渲染。

    各节正文已在 _generate_section 内按本节编号白名单校验过；
    导语与整篇结构仍在落盘前由 pipeline 调 _validate_report 兜一次底
    （跨节的幻觉编号、以及非流式路径共用同一道校验）。
    """
    yield from _iter_report(llm, question, plan, kb)
