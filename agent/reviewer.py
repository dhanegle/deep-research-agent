"""阶段 E：报告自审（对称于阶段 C 的反思补搜）。

把「报告质量是否过关」这类开放判断降级为受限输出（sufficient 布尔 + 问题清单），
并用规则层兜底：占位节有可用来源、零引用节、数字存疑等先免费扫一遍，
再让 LLM 做一次结构审查确认，最后对存疑节逐句裁决。

三阶段成本递增，前阶段不命中则后续不触发：
① 规则层（免费）→ ② LLM 结构审查（1 次）→ ③ 逐句忠实度（3-6 次，仅存疑节）
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .factcheck import CITE_RE, cited_sentences, missing_numbers, judge
from .knowledge import KnowledgeBase
from .llm import LLM
from .planner import Plan


class SectionIssue(BaseModel):
    section: str
    problem: Literal["placeholder_fillable", "no_citation", "faithfulness_suspect"]
    detail: str = ""
    gap_query: str = ""

    @field_validator("gap_query")
    @classmethod
    def _clean_gap(cls, v: str) -> str:
        return v.strip()[:60]


class Review(BaseModel):
    """对称于 reflector.Reflection：sufficient + issues。"""
    sufficient: bool
    issues: list[SectionIssue] = Field(default_factory=list, max_length=10)

    @property
    def gap_queries(self) -> list[str]:
        return [i.gap_query for i in self.issues
                if i.problem == "placeholder_fillable" and i.gap_query]


SYSTEM_PROMPT = """\
你是报告质量审查员，判断报告各节是否存在问题。
只输出 JSON：{"sufficient": true或false, "issues": [{"section":"节标题", "problem":"问题类型", "detail":"简述", "gap_query":"搜索词"}]}
问题类型三选一：
- placeholder_fillable：该节为"（本节暂缺相关资料）"但有可用资料可补——必须在 gap_query 给出搜索词（含主题关键词，≤15字，单一主题）
- no_citation：该节有实质内容但没有任何 [n] 引用标记
- faithfulness_suspect：该节数字或事实可能与来源不符（规则层已标记可疑数字）
判定标准：无占位节、无零引用节、无存疑句时 sufficient 为 true。
"""


def _rule_layer(report: str, question: str, plan: Plan, kb: KnowledgeBase) -> list[SectionIssue]:
    """① 规则层（零成本）：占位节查可补、零引用节查缺失、数字查存疑。"""
    issues: list[SectionIssue] = []
    sections = _split_report_sections(report)

    for sec, body in sections.items():
        if sec not in plan.outline:
            continue
        # 占位节：有可用来源才标记可补
        if "暂缺" in body and kb.has_section_material(sec):
            issues.append(SectionIssue(
                section=sec, problem="placeholder_fillable",
                detail="占位节但有可用资料",
            ))
            continue
        # 零引用节：有实质内容却无引用
        if "暂缺" not in body and body.strip() and not CITE_RE.search(body):
            issues.append(SectionIssue(
                section=sec, problem="no_citation",
                detail="有内容但无引用标记",
            ))
        # 数字存疑：句子里的数字不在被引来源摘要里
        for sent, ids in cited_sentences(body):
            digests = " ".join(
                kb.sources[i - 1].digest for i in ids if 1 <= i <= len(kb.sources)
            )
            flagged = missing_numbers(sent, digests, question)
            if flagged:
                issues.append(SectionIssue(
                    section=sec, problem="faithfulness_suspect",
                    detail=f"存疑数字：{','.join(flagged[:3])}",
                ))
                break
    return issues


def _split_report_sections(report: str) -> dict[str, str]:
    """把报告按 ## 标题切成 {标题: 正文}，跳过 # 主标题和 ## 参考来源。"""
    sections: dict[str, str] = {}
    current_title = ""
    current_body: list[str] = []
    for line in report.splitlines():
        if line.startswith("## "):
            if current_title:
                sections[current_title] = "\n".join(current_body).strip()
            current_title = line[3:].strip()
            current_body = []
            if "参考来源" in current_title:
                break
        elif current_title:
            current_body.append(line)
    if current_title and "参考来源" not in current_title:
        sections[current_title] = "\n".join(current_body).strip()
    return sections


def review_report(llm: LLM, question: str, plan: Plan, report: str,
                  kb: KnowledgeBase) -> Review:
    """报告自审入口：规则层 → LLM 结构审查 → 存疑节逐句裁决。"""
    # ① 规则层
    rule_issues = _rule_layer(report, question, plan, kb)

    # ② LLM 结构审查（1 次 chat_json）
    sections = _split_report_sections(report)
    section_summaries = "\n".join(
        f"### {sec}\n{sections.get(sec, '（未生成）')[:200]}" for sec in plan.outline
    )
    rule_hints = "\n".join(
        f"- {i.section}：{i.problem}（{i.detail}）" for i in rule_issues
    ) or "（规则层未发现问题）"
    try:
        review = llm.chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"调研问题：{question}\n报告大纲：{'；'.join(plan.outline)}\n"
                f"各节正文摘要：\n{section_summaries}\n\n"
                f"规则层标记：\n{rule_hints}"
            )},
        ], Review, tag="reviewer")
    except ValueError:
        # LLM 审查失败：回退到仅规则层结果
        return Review(sufficient=not rule_issues, issues=rule_issues)

    # ③ 逐句忠实度裁决：仅对 faithfulness_suspect 的节，限 3-6 句
    suspect_sections = {i.section for i in review.issues
                        if i.problem == "faithfulness_suspect"}
    if suspect_sections:
        by_id = {s.id: s for s in kb.sources}
        confirmed: list[SectionIssue] = []
        for issue in list(review.issues):
            if issue.problem != "faithfulness_suspect":
                confirmed.append(issue)
                continue
            body = sections.get(issue.section, "")
            judged = 0
            refuted = False
            for sent, ids in cited_sentences(body):
                if judged >= 4:
                    break
                digests = " ".join(
                    by_id[i].digest for i in ids if i in by_id
                )
                if not digests:
                    continue
                verdict = judge(llm, sent, digests)
                judged += 1
                if verdict == "不支持":
                    refuted = True
                    issue.detail = f"裁决不支持：{sent[:40]}"
                    break
            if refuted:
                confirmed.append(issue)
            # 裁决通过则丢弃该存疑标记（不冤枉）
        review.issues = confirmed

    review.sufficient = not review.issues
    return review
