"""agent.writer 引用合规测试：幻觉引用剔除、整篇校验。

引用校验是报告质量的格式层防线——只剔除指向不存在来源编号的 [n] 标记，
合法引用保留不动。这些测试锁定该行为不被回归。
"""
from agent.writer import _validate_citations, _validate_report
from agent.knowledge import KnowledgeBase


class TestValidateCitations:
    def test_keeps_valid_ids(self):
        assert _validate_citations("数据见[1]和[2]", {1, 2}) == "数据见[1]和[2]"

    def test_strips_invalid_ids(self):
        assert _validate_citations("数据见[1]和[99]", {1, 2}) == "数据见[1]和"

    def test_mixed(self):
        text = "[1]成立 [3]不存在 [2]也成立"
        assert _validate_citations(text, {1, 2}) == "[1]成立 不存在 [2]也成立"

    def test_empty_allowed(self):
        assert _validate_citations("[1][2]", set()) == ""

    def test_no_citations(self):
        assert _validate_citations("普通文字", {1}) == "普通文字"


class TestValidateReport:
    def _kb(self, n=2):
        kb = KnowledgeBase()
        for i in range(n):
            kb.add(f"url{i}", f"标题{i}", "摘要")
        return kb

    def test_strips_hallucinated_keeps_valid(self):
        report = "# 标题\n\n## 章节\n\n数据见[1]和[3]。\n\n## 参考来源\n\n[1] t1 u1\n[2] t2 u2\n"
        out = _validate_report(report, self._kb(2))
        # [3] 被剔除，[1] 保留
        assert "[1]" in out
        assert "[3]" not in out
        # 参考来源页脚的 [1] [2] 不受影响
        assert "[2] t2 u2" in out

    def test_preserves_footer_citations(self):
        report = "## 参考来源\n\n[1] t1 u1\n[2] t2 u2\n"
        out = _validate_report(report, self._kb(2))
        assert "[1] t1 u1" in out
        assert "[2] t2 u2" in out

    def test_no_sections(self):
        report = "# 标题\n\n导语，无章节。\n\n## 参考来源\n\n[1] t1 u1\n"
        out = _validate_report(report, self._kb(1))
        assert "# 标题" in out
        assert "[1] t1 u1" in out
