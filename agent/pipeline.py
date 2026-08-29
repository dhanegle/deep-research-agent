"""管线编排：规划 → ReAct 收集 → 反思补搜（≤N 轮）→ 分节写作 → 报告自审修订 → 落盘。

on_step 回调把每个阶段事件实时抛给 CLI / Web UI 展示。
"""
import re
import time
from dataclasses import dataclass
from typing import Callable

from .search import build_provider

from . import config
from .knowledge import KnowledgeBase
from .llm import LLM
from .planner import plan_question
from .reflector import reflect
from .researcher import Researcher
from .reviewer import review_report
from .stats import RunStats
from .trace import StepCallback, Trace
from .writer import _generate_section, _replace_section, _validate_report, iter_report


@dataclass
class RunResult:
    report_path: str
    report: str
    stats: dict
    trace_path: str
    plan: dict
    sources: list  # [{id, url, title, digest}]，引用忠实度校验需要摘要作依据


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
            if len(kb) >= 8:
                break
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
        body = _validate_report("".join(chunks), kb)

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
            for issue in review.issues:
                sources = kb.select_for(issue.section, question, k=5)
                if not sources:
                    continue
                material = "\n\n".join(f"[{s.id}] 《{s.title}》\n{s.digest}" for s in sources)
                new_body = _generate_section(
                    self.llm, question, issue.section, material, {s.id for s in sources}
                )
                body = _replace_section(body, issue.section, new_body)
                revised += 1
            self.stats.review_revised = revised
            body = _validate_report(body, kb)
            self.trace.log("review_revised", revised=revised)
            emit("review", {"status": f"修订完成，重写 {revised} 节"})

        self.stats.total_seconds = round(time.time() - t0, 1)
        meta = (f"> 模型 `{config.MODEL}` | 来源 {len(kb)} 条 | LLM 调用 {self.stats.llm_calls} 次 | "
                f"耗时 {self.stats.total_seconds}s | trace `{self.trace.run_id}`")
        title_line, _, rest = body.partition("\n")
        report = f"{title_line}\n\n{meta}\n{rest}"

        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(question)}.md"
        path.write_text(report, encoding="utf-8")
        self.trace.log("run_end", report_path=str(path), stats=self.stats.as_dict())
        emit("finish", {"report_path": str(path), "sources": len(kb)})
        sources = [{"id": s.id, "url": s.url, "title": s.title, "digest": s.digest} for s in kb.sources]
        return RunResult(str(path), report, self.stats.as_dict(), str(self.trace.path),
                         plan.model_dump(), sources)
