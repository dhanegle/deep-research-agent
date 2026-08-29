"""agent.reviewer 报告自审测试：占位节可补、零引用检测、存疑裁决、节替换。

对称于 reflector 测试——验证"写-反思"闭环的审查逻辑正确识别问题类型，
不误判正常报告，且 _replace_section 正确替换指定节正文。
"""
from agent.reviewer import Review, SectionIssue, _rule_layer, _split_report_sections
from agent.writer import _replace_section, PLACEHOLDER
from agent.knowledge import KnowledgeBase
from agent.planner import Plan
from agent.llm import LLM
from agent.stats import RunStats
from agent.trace import Trace


class ScriptedLLM:
    """按预定回复逐次 yield chat_json 和 stream_text。"""

    def __init__(self, json_replies=None, stream_replies=None, judge_replies=None):
        self.json_replies = list(json_replies or [])
        self.stream_replies = list(stream_replies or [])
        self.judge_replies = list(judge_replies or [])
        self.n_json = 0
        self.n_stream = 0
        self.n_judge = 0

    def chat_json(self, messages, schema, tag):
        self.n_json += 1
        if schema.__name__ == "Judgment":
            self.n_judge += 1
            r = self.judge_replies.pop(0)
            return schema(verdict=r)
        return self.json_replies.pop(0)

    def stream_text(self, messages):
        self.n_stream += 1
        yield self.stream_replies.pop(0)


def _make_kb(*sources):
    kb = KnowledgeBase()
    for title, digest in sources:
        kb.add(f"url-{title}", title, digest)
    return kb


def _make_plan(outline):
    # Plan 要求 outline≥3、search_queries≥2，测试只关注一节时补齐到最小合法
    full_outline = list(outline) + ["补充节B", "补充节C"][: max(0, 3 - len(outline))]
    return Plan(outline=full_outline, search_queries=["测试查询A", "测试查询B"])


class TestSplitReportSections:
    def test_basic_split(self):
        report = "# 标题\n\n## 节A\n正文A\n\n## 节B\n正文B\n\n## 参考来源\n[1] x"
        sections = _split_report_sections(report)
        assert "节A" in sections
        assert "节B" in sections
        assert "参考来源" not in sections
        assert "正文A" in sections["节A"]

    def test_empty_report(self):
        assert _split_report_sections("# 标题\n") == {}


class TestRuleLayer:
    def test_placeholder_fillable_detected(self):
        kb = _make_kb(("毕业生就业数据", "2026届毕业生5191人，落实率95%"))
        plan = _make_plan(["毕业生总体情况"])
        report = f"# q\n\n## 毕业生总体情况\n\n{PLACEHOLDER}\n\n## 参考来源\n\n[1] x"
        issues = _rule_layer(report, "毕业生数据", plan, kb)
        assert any(i.problem == "placeholder_fillable" for i in issues)

    def test_placeholder_no_fillable_source(self):
        kb = _make_kb(("量子计算进展", "量子比特数突破1000"))
        plan = _make_plan(["毕业生总体情况"])
        report = f"# q\n\n## 毕业生总体情况\n\n{PLACEHOLDER}\n\n## 参考来源\n\n[1] x"
        issues = _rule_layer(report, "毕业生数据", plan, kb)
        assert not any(i.problem == "placeholder_fillable" for i in issues)

    def test_no_citation_detected(self):
        kb = _make_kb(("毕业生就业", "2026届5191人毕业"))
        plan = _make_plan(["毕业生总体情况"])
        report = "# q\n\n## 毕业生总体情况\n\n深圳大学5191名学子毕业，落实率95%。\n\n## 参考来源\n\n[1] x"
        issues = _rule_layer(report, "毕业生数据", plan, kb)
        assert any(i.problem == "no_citation" for i in issues)

    def test_cited_section_no_issue(self):
        kb = _make_kb(("毕业生就业", "2026届5191人毕业"))
        plan = _make_plan(["毕业生总体情况"])
        report = "# q\n\n## 毕业生总体情况\n\n深圳大学5191名学子毕业，落实率95%[1]。\n\n## 参考来源\n\n[1] x"
        issues = _rule_layer(report, "毕业生数据", plan, kb)
        assert not any(i.problem == "no_citation" for i in issues)

    def test_faithfulness_suspect_detected(self):
        kb = _make_kb(("就业数据", "2026届5191人毕业，落实率95%"))
        plan = _make_plan(["毕业生总体情况"])
        # 句子含 9999 这个来源里没有的数字
        report = "# q\n\n## 毕业生总体情况\n\n深圳大学9999名学子毕业[1]。\n\n## 参考来源\n\n[1] x"
        issues = _rule_layer(report, "毕业生数据", plan, kb)
        assert any(i.problem == "faithfulness_suspect" for i in issues)


class TestReplaceSection:
    def test_replaces_middle_section(self):
        report = "# 标题\n\n## 节A\n旧A\n\n## 节B\n旧B\n\n## 节C\n旧C\n\n## 参考来源\n[1] x"
        out = _replace_section(report, "节B", "新B正文[1]")
        assert "新B正文" in out
        assert "旧B" not in out
        assert "节A" in out and "节C" in out
        assert "参考来源" in out

    def test_replaces_last_section_before_footer(self):
        report = "# 标题\n\n## 节A\n旧A\n\n## 参考来源\n[1] x"
        out = _replace_section(report, "节A", "新A[1]")
        assert "新A" in out
        assert "旧A" not in out
        assert "参考来源" in out

    def test_no_match_returns_original(self):
        report = "# 标题\n\n## 节A\n旧A\n\n## 参考来源\n[1] x"
        out = _replace_section(report, "不存在节", "新正文")
        assert out == report


class TestReviewReport:
    def test_sufficient_report_no_issues(self):
        kb = _make_kb(("就业数据", "2026届5191人毕业，落实率95%"))
        plan = _make_plan(["毕业生总体情况"])
        report = "# q\n\n## 毕业生总体情况\n\n深圳大学5191名学子毕业，落实率95%[1]。\n\n## 参考来源\n\n[1] x"
        review_json = Review(sufficient=True, issues=[])
        llm = ScriptedLLM(json_replies=[review_json])
        from agent.reviewer import review_report
        review = review_report(llm, "毕业生数据", plan, report, kb)
        assert review.sufficient is True
        assert review.issues == []

    def test_placeholder_triggers_issue_and_gap_query(self):
        kb = _make_kb(("就业数据", "2026届5191人毕业，落实率95%"))
        plan = _make_plan(["毕业生总体情况"])
        report = f"# q\n\n## 毕业生总体情况\n\n{PLACEHOLDER}\n\n## 参考来源\n\n[1] x"
        review_json = Review(sufficient=False, issues=[
            SectionIssue(section="毕业生总体情况", problem="placeholder_fillable",
                         detail="占位节", gap_query="2026年毕业生就业数据"),
        ])
        llm = ScriptedLLM(json_replies=[review_json])
        from agent.reviewer import review_report
        review = review_report(llm, "毕业生数据", plan, report, kb)
        assert review.sufficient is False
        assert review.gap_queries == ["2026年毕业生就业数据"]

    def test_faithfulness_suspect_confirmed_by_judge(self):
        kb = _make_kb(("就业数据", "2026届5191人毕业，落实率95%"))
        plan = _make_plan(["毕业生总体情况"])
        report = "# q\n\n## 毕业生总体情况\n\n深圳大学9999名学子毕业[1]。\n\n## 参考来源\n\n[1] x"
        review_json = Review(sufficient=False, issues=[
            SectionIssue(section="毕业生总体情况", problem="faithfulness_suspect",
                         detail="存疑数字：9999"),
        ])
        # judge 返回"不支持"→ 保留该存疑标记
        llm = ScriptedLLM(json_replies=[review_json], judge_replies=["不支持"])
        from agent.reviewer import review_report
        review = review_report(llm, "毕业生数据", plan, report, kb)
        assert any(i.problem == "faithfulness_suspect" for i in review.issues)
        assert llm.n_judge >= 1

    def test_faithfulness_suspect_cleared_by_judge(self):
        kb = _make_kb(("就业数据", "2026届5191人毕业，落实率95%"))
        plan = _make_plan(["毕业生总体情况"])
        report = "# q\n\n## 毕业生总体情况\n\n深圳大学9999名学子毕业[1]。\n\n## 参考来源\n\n[1] x"
        review_json = Review(sufficient=False, issues=[
            SectionIssue(section="毕业生总体情况", problem="faithfulness_suspect",
                         detail="存疑数字：9999"),
        ])
        # judge 返回"支持"→ 存疑标记被清除
        llm = ScriptedLLM(json_replies=[review_json], judge_replies=["支持"])
        from agent.reviewer import review_report
        review = review_report(llm, "毕业生数据", plan, report, kb)
        assert not any(i.problem == "faithfulness_suspect" for i in review.issues)
