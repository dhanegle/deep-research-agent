"""管线编排：规划 → ReAct 收集 → 反思补搜（≤N 轮）→ 分节写作 → 报告自审修订 → 落盘。

on_step 回调把每个阶段事件实时抛给 CLI / Web UI 展示。
"""
import re
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

from .search import build_provider

from . import config
from .knowledge import KnowledgeBase, source_material
from .llm import LLM
from .planner import Plan, plan_question
from .reflector import reflect
from .researcher import Researcher
from .reviewer import review_report
from .stats import RunStats
from .trace import StepCallback, Trace
from .writer import (add_findings, refresh_references, _dedupe_sections, _generate_section,
                     _replace_section, _validate_report, iter_report)


@dataclass
class RunResult:
    report_path: str
    report: str
    stats: dict
    trace_path: str
    plan: dict
    sources: list  # [{id, url, title, digest}]，引用忠实度校验需要摘要作依据
    evidence_path: str = ""
    quality: dict | None = None


def _slug(text: str, limit: int = 24) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff-]+", "-", text)[:limit].strip("-") or "report"


def _resolve_relative_time(question: str) -> str:
    """模型没有时钟："今年/去年"在进入管线前确定性替换为具体年份，
    让规划出的搜索词自带正确年份，不指望 3B 自己推算。"""
    year = int(time.strftime("%Y"))
    return question.replace("今年", f"{year}年").replace("去年", f"{year - 1}年")


class ResearchPipeline:
    def __init__(self, provider_name: str | None = None, on_step: StepCallback | None = None,
                 on_chunk: Callable[[str], None] | None = None):
        self.stats = RunStats()
        self.trace = Trace()
        self.llm = LLM(self.stats, self.trace)
        self.provider = build_provider(provider_name)
        self.on_step = on_step or (lambda event, data: None)
        self.on_chunk = on_chunk

    def run(self, question: str) -> RunResult:
        question = _resolve_relative_time(question)
        t0 = time.time()
        emit = self.on_step
        self.trace.log("run_start", question=question, provider=type(self.provider).__name__)
        emit("start", {"question": question})

        # 阶段 A：规划
        plan: Plan = plan_question(self.llm, question)
        self.trace.log("plan_done", outline=plan.outline, queries=plan.search_queries)
        emit("plan", {"outline": plan.outline, "queries": plan.search_queries})

        # 阶段 B：信息收集
        kb = KnowledgeBase()
        researcher = Researcher(self.llm, self.provider, kb, self.stats, self.trace, emit)
        emit("research", {"status": "开始收集资料"})
        researcher.run(question, plan.outline, plan.search_queries)
        if len(kb) == 0:
            # 防线 3：模型驱动颗粒无收，规则兜底采集
            emit("research", {"status": "模型采集未果，启用规则兜底"})
            researcher.scripted_collect(plan.search_queries)
        emit("research", {"status": f"完成，收录 {len(kb)} 条来源"})

        # 阶段 C：反思补搜
        for round_i in range(config.MAX_REFLECT_ROUNDS):
            r = reflect(self.llm, question, plan, kb)
            self.stats.reflect_rounds += 1
            self.trace.log("reflect", round=round_i + 1, sufficient=r.sufficient, gaps=r.gap_queries)
            emit("reflect", {"round": round_i + 1, "sufficient": r.sufficient, "gap_queries": r.gap_queries})
            if r.sufficient or not r.gap_queries:
                break
            emit("research", {"status": f"第 {round_i + 1} 轮补搜：{r.gap_queries}"})
            before = len(kb)
            researcher.run(question, plan.outline, r.gap_queries, max_steps=6)
            if len(kb) == before:  # 小循环颗粒无收，退化为确定性采集
                researcher.scripted_collect(r.gap_queries, per_query_pages=1)

        if len(kb) == 0:
            raise RuntimeError("未能收集到任何资料（搜索无结果且兜底采集失败），请更换调研问题或检查搜索配置。")

        # 阶段 D：写作（流式产出，经 on_step 推给前端逐字渲染；落盘前做引用校验）
        emit("write", {"status": "撰写报告中…", "sections": plan.outline})
        chunks = []
        for piece in iter_report(self.llm, question, plan, kb):
            chunks.append(piece)
            if self.on_chunk:
                self.on_chunk(piece)
        body, deduped = _dedupe_sections(_validate_report("".join(chunks), kb), question)
        if deduped:
            self.stats.deduped_sentences += deduped
            emit("write", {"status": f"跨节去重：删除 {deduped} 句重复内容"})

        # 阶段 E：报告自审+修订（固定 1 轮，对称于阶段 C 的搜-反思）
        review = review_report(self.llm, question, plan, body, kb)
        self.stats.review_rounds += 1
        self.stats.review_issues = len(review.issues)
        self.trace.log("review", sufficient=review.sufficient,
                       issues=[i.model_dump() for i in review.issues])
        emit("review", {"sufficient": review.sufficient,
                        "issues": [i.model_dump() for i in review.issues]})
        if not review.sufficient:
            emit("review", {"status": "自审发现不足，修订中…"})
            # E1: 占位节触发补搜（对称于 C 的 researcher.run 补搜）
            gaps = review.gap_queries
            if gaps:
                emit("research", {"status": f"自审补搜：{gaps}"})
                before = len(kb)
                researcher.run(question, plan.outline, gaps, max_steps=4)
                if len(kb) == before:
                    researcher.scripted_collect(gaps, per_query_pages=1)
            # E2: 重写有问题的节
            revised = 0
            handled = set()
            for issue in review.issues:
                if issue.section in handled or issue.problem == "verification_incomplete":
                    continue
                handled.add(issue.section)
                sources = kb.select_for(issue.section, question, k=3)
                if not sources:
                    continue
                material = source_material(sources, f"{question} {issue.section}",
                                           section=issue.section)
                feedback = "\n".join(i.detail for i in review.issues if i.section == issue.section)
                new_body = _generate_section(
                    self.llm, question, issue.section, material, {s.id for s in sources},
                    feedback=feedback[:1500], sources=sources,
                )
                body = _replace_section(body, issue.section, new_body)
                revised += 1
            self.stats.review_revised = revised
            # 重写的节可能又复述了别节的内容，去重要再走一遍（幂等，不重复删）
            body, deduped = _dedupe_sections(
                _validate_report(refresh_references(body, kb), kb), question)
            self.stats.deduped_sentences += deduped
            review = review_report(self.llm, question, plan, body, kb, previous=review)
            self.stats.review_rounds += 1
            self.trace.log("review_revised", revised=revised)
            emit("review", {"status": f"修订完成，重写 {revised} 节"})

        self.stats.review_unresolved = len(review.issues)
        self.stats.claims_checked = review.checked_claims
        self.stats.claims_total = review.total_claims
        self.stats.quality_status = "no_flags" if review.sufficient else "needs_review"
        quality = {"status": self.stats.quality_status, **review.model_dump()}
        self.trace.log("review_final", **quality)
        emit("review", {"status": "核验完成" if review.sufficient else
                       f"仍有 {len(review.issues)} 项待核验或资料缺口"})
        body = add_findings(refresh_references(body, kb), review.claims)
        self.stats.total_seconds = round(time.time() - t0, 1)
        meta = (f"> 模型 `{config.MODEL}` | 来源 {len(kb)} 条 | LLM 调用 {self.stats.llm_calls} 次 | "
                f"耗时 {self.stats.total_seconds}s | trace `{self.trace.run_id}`")
        title_line, _, rest = body.partition("\n")
        status = "未发现核验问题" if review.sufficient else f"待核验草稿：{len(review.issues)} 项未解决"
        quality_note = f"> 核验状态：{status}；已检查 {review.checked_claims}/{review.total_claims} 句。"
        if review.issues:
            pending = list(dict.fromkeys(i.section for i in review.issues))
            quality_note += "\n> 待核验章节：" + "、".join(pending)
        report = f"{title_line}\n\n{meta}\n\n{quality_note}\n{rest}"

        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(question)}.md"
        path.write_text(report, encoding="utf-8")
        sources = [asdict(s) for s in kb.sources]
        evidence_path = path.with_suffix(".evidence.json")
        evidence_path.write_text(json.dumps({
            "question": question, "plan": plan.model_dump(), "sources": sources, "quality": quality,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self.trace.log("run_end", report_path=str(path), stats=self.stats.as_dict())
        emit("finish", {"report_path": str(path), "sources": len(kb)})
        return RunResult(str(path), report, self.stats.as_dict(), str(self.trace.path),
                         plan.model_dump(), sources, str(evidence_path), quality)
