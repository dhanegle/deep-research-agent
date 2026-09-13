"""按原文证据审查报告；规则问题、未核验句和资料缺口都保留到最终状态。"""
from typing import Literal

from pydantic import BaseModel, Field

from . import config
from .evidence import content_hash
from .factcheck import CITE_RE, _iter_sentences, missing_numbers, judge, unsupported_citations
from .knowledge import KnowledgeBase
from .llm import LLM
from .planner import Plan


class SectionIssue(BaseModel):
    section: str
    problem: Literal["placeholder_fillable", "missing_evidence", "no_citation",
                     "faithfulness_suspect", "verification_incomplete"]
    detail: str = ""
    gap_query: str = ""


class Review(BaseModel):
    sufficient: bool
    issues: list[SectionIssue] = Field(default_factory=list)
    checked_claims: int = 0
    total_claims: int = 0
    claims: list[dict] = Field(default_factory=list)

    @property
    def gap_queries(self) -> list[str]:
        return list(dict.fromkeys(
            i.gap_query for i in self.issues
            if i.problem in {"placeholder_fillable", "missing_evidence"} and i.gap_query
        ))[:2]


def _split_report_sections(report: str) -> dict[str, str]:
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


def _claims(body: str):
    for sent in _iter_sentences(body):
        if "暂缺" in sent and len(sent) < 60:
            continue
        if len(sent) >= 10 or CITE_RE.search(sent):
            yield sent, sorted({int(i) for i in CITE_RE.findall(sent)})


def _evidence_by_id(kb: KnowledgeBase, sent: str, ids: list[int]) -> dict[int, str]:
    by_id = {s.id: s for s in kb.sources}
    return {i: by_id[i].excerpt(sent, 1800) for i in ids if i in by_id}


def _evidence(kb: KnowledgeBase, sent: str, ids: list[int]) -> str:
    return "\n\n".join(f"[{i}] {ev}" for i, ev in _evidence_by_id(kb, sent, ids).items())


def _rule_layer(report: str, question: str, plan: Plan, kb: KnowledgeBase) -> list[SectionIssue]:
    issues = []
    by_id = {s.id: s for s in kb.sources}
    for sec, body in _split_report_sections(report).items():
        if sec not in plan.outline:
            continue
        if not body or ("暂缺" in body and len(body) < 70):
            fillable = bool(kb.select_for(sec, question, k=1))
            issues.append(SectionIssue(
                section=sec, problem="placeholder_fillable" if fillable else "missing_evidence",
                detail="本节缺少可用证据", gap_query=f"{question} {sec}"[:60],
            ))
            continue
        for sent, ids in _claims(body):
            if not ids or any(i not in by_id for i in ids):
                issues.append(SectionIssue(section=sec, problem="no_citation",
                                           detail=f"缺少有效引用：{sent[:100]}"))
                continue
            per_id = _evidence_by_id(kb, sent, ids)
            flagged = missing_numbers(sent, _evidence(kb, sent, ids), question)
            if flagged:
                issues.append(SectionIssue(
                    section=sec, problem="faithfulness_suspect",
                    detail=f"原文不支持数值、单位或期间 {','.join(flagged)}：{sent[:100]}",
                ))
                continue
            extra = unsupported_citations(sent, per_id, question)
            if extra:
                marks = "".join(f"[{i}]" for i in extra)
                issues.append(SectionIssue(
                    section=sec, problem="faithfulness_suspect",
                    detail=f"来源{marks}不支持本句任何数据，请去掉这些编号只保留真正的出处：{sent[:100]}",
                ))
    return issues


def review_report(llm: LLM, question: str, plan: Plan, report: str,
                  kb: KnowledgeBase, previous: Review | None = None) -> Review:
    issues = _rule_layer(report, question, plan, kb)
    previous_claims = {
        (c["sentence"], c["evidence_hash"]): c
        for c in (previous.claims if previous else [])
        if c.get("verdict") is not None
    }
    records = []
    calls = 0
    by_id = {s.id: s for s in kb.sources}
    for sec, body in _split_report_sections(report).items():
        if sec not in plan.outline:
            continue
        for sent, ids in _claims(body):
            evidence = _evidence(kb, sent, ids)
            fingerprint = content_hash(evidence)
            verdict = None
            method = "unverified"
            if ids and all(i in by_id for i in ids):
                flags = missing_numbers(sent, evidence, question)
                cached = previous_claims.get((sent, fingerprint))
                if flags:
                    verdict, method = "不支持", "numeric"
                elif cached:
                    verdict, method = cached["verdict"], cached["method"]
                elif calls < config.MAX_REVIEW_CLAIMS:
                    verdict, method = judge(llm, sent, evidence), "llm"
                    calls += 1
                if verdict in {"不支持", "部分支持"} and not flags:
                    issues.append(SectionIssue(
                        section=sec, problem="faithfulness_suspect",
                        detail=f"原文{verdict}：{sent[:120]}",
                    ))
                elif verdict is None:
                    issues.append(SectionIssue(
                        section=sec, problem="verification_incomplete",
                        detail=f"尚未完成原文核验：{sent[:100]}",
                    ))
            records.append({
                "section": sec, "sentence": sent, "source_ids": ids,
                "verdict": verdict, "method": method, "evidence_hash": fingerprint,
            })
    unique = {}
    for issue in issues:
        unique.setdefault((issue.section, issue.problem, issue.detail), issue)
    issues = list(unique.values())
    return Review(
        sufficient=not issues, issues=issues, total_claims=len(records),
        checked_claims=sum(c["verdict"] is not None for c in records), claims=records,
    )
