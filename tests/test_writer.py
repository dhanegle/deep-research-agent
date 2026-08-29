"""agent.writer 引用合规测试：幻觉引用剔除、整篇校验。

引用校验是报告质量的格式层防线——只剔除指向不存在来源编号的 [n] 标记，
合法引用保留不动。这些测试锁定该行为不被回归。
"""
from agent.writer import (
    PLACEHOLDER, _clean_section_body, _generate_section, _iter_report,
    _material_ids, _needs_rewrite, _section_aligned, _validate_citations,
    _validate_report,
)
from agent.knowledge import KnowledgeBase
from agent.planner import Plan


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


class TestNeedsRewrite:
    def test_uncited_prose(self):
        assert _needs_rewrite("出口达到343万辆，同比增长70%。") is True

    def test_cited_prose(self):
        assert _needs_rewrite("出口达到343万辆[1]，同比增长70%。") is False

    def test_placeholder(self):
        assert _needs_rewrite(PLACEHOLDER) is False
        assert _needs_rewrite("（本节暂缺可用资料。）") is False

    def test_empty(self):
        assert _needs_rewrite("  \n") is False


class ScriptedLLM:
    """按预定回复逐次 yield，记录调用次数。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.n_calls = 0

    def stream_text(self, messages):
        self.n_calls += 1
        yield self.replies.pop(0)


class TestCleanAndAlign:
    def test_strips_literal_n_cite(self):
        assert "[n]" not in _clean_section_body("（本节暂缺相关资料） [n]")
        assert _clean_section_body("（本节暂缺相关资料） [n]") == PLACEHOLDER

    def test_trade_partner_not_aligned_with_appliance_text(self):
        # 跑题判定：家电出口写进"贸易伙伴分析"节 → 二元组全不命中 → 不对齐
        body = "家电出口突破1000亿元，体育用品出口675亿元。"
        assert _section_aligned(body, "贸易伙伴分析", "2026年出口") is False
        # 扣题判定：正文含标题区别二元组（"出口"是问题词，"产品"是标题独有）→ 对齐
        assert _section_aligned("主要出口产品包括家电[1]。", "主要出口产品", "2026年出口") is True

    def test_paraphrase_aligned_via_bigram(self):
        # 正文是对标题的释义而非照抄，4字窗口会误杀，二元组应判为扣题
        body = "深圳信息职业技术大学5191名学子毕业，去向落实率95.76%。"
        assert _section_aligned(body, "毕业生总体情况", "调研2026年毕业生数据") is True

    def test_keeps_full_body_that_merely_mentions_missing_data(self):
        # 正文写满了，只是捎带一句"暂缺"——不该把整节有效内容丢掉
        body = ("2025年新能源汽车出口达343万辆[1]，同比增长70%[1]；"
                "欧洲与东南亚为主要目的地[2]。分车型的细分数据暂缺。")
        assert _clean_section_body(body) == body

    def test_collapses_placeholder_with_trailing_noise(self):
        assert _clean_section_body("（本节暂缺相关资料）\n\n") == PLACEHOLDER

    def test_material_ids_only_reads_entry_headers(self):
        assert _material_ids("[1] 《甲》\n摘要含[9]\n\n[2] 《乙》\n摘要") == {1, 2}


class TestGenerateSection:
    def test_keeps_cited_first_draft(self):
        llm = ScriptedLLM(["出口总量达到343万辆[1]。"])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert "[1]" in out
        assert llm.n_calls == 1

    def test_rewrites_uncited_once(self):
        # 零引用 → 请求补编号重写；重写带引用 → 用重写稿
        llm = ScriptedLLM(["出口总量达到343万辆。", "出口总量达到343万辆[1]。"])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert "[1]" in out
        assert llm.n_calls == 2

    def test_keeps_first_draft_when_rewrite_still_uncited(self):
        # 零引用 → 请求重写 → 重写仍无引用：保留首稿原文，不转占位符
        llm = ScriptedLLM(["出口总量达到343万辆。", "出口总量达到343万辆。"])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert "343万辆" in out
        assert out != PLACEHOLDER
        assert llm.n_calls == 2

    def test_placeholder_not_rewritten(self):
        llm = ScriptedLLM([PLACEHOLDER])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert out == PLACEHOLDER
        assert llm.n_calls == 1

    def test_hallucinated_id_triggers_rewrite(self):
        # 本节只注入 [1]，模型却写 [7]：剔除后等于零引用，必须回喂重写
        llm = ScriptedLLM(["出口总量达到343万辆[7]。", "出口总量达到343万辆[1]。"])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要", {1})
        assert "[7]" not in out
        assert "[1]" in out
        assert llm.n_calls == 2

    def test_allowed_ids_default_to_material_headers(self):
        material = "[1] 《甲》\n摘要提到[9]这个编号\n\n[2] 《乙》\n摘要"
        llm = ScriptedLLM(["出口总量增长[9]。", "出口总量增长[2]。"])
        out = _generate_section(llm, "出口", "出口总量", material)
        assert "[9]" not in out
        assert llm.n_calls == 2

    def test_off_topic_becomes_placeholder(self):        # 标题是贸易伙伴，正文却写家电 → 对齐失败后暂缺
        llm = ScriptedLLM([
            "家电出口突破1000亿元[1]。",
            "家电和体育用品出口增长[1]。",
        ])
        out = _generate_section(llm, "2026年出口", "贸易伙伴分析", "[1] 家电摘要")
        assert out == PLACEHOLDER
        assert llm.n_calls == 2


class TestIterReportEmptySection:
    def test_unrelated_sources_yield_placeholder(self):
        kb = KnowledgeBase()
        kb.add("u1", "欧洲旅游攻略", "巴黎景点推荐")
        plan = Plan(outline=["量子计算进展", "量子计算挑战", "量子计算趋势"],
                    search_queries=["量子计算 进展", "量子计算 挑战"])
        # select_for 不再用门槛剔除——库不足 k 条时全部返回，各节都会调 LLM。
        # 模型对"量子计算"只看到"巴黎景点"资料：
        #   零引用 → 重写补编号（仍无引用，保留首稿）→ 跑题 → 对齐重写（仍跑题）→ 占位符
        # 每节 3 次调用，3 节 + 导语 = 10 次。
        llm = ScriptedLLM(["导语正文"] + ["巴黎是浪漫之都。"] * 9)
        text = "".join(_iter_report(llm, "量子计算产业化", plan, kb))
        assert text.count(PLACEHOLDER) == 3
        assert "巴黎" not in text.split("## 参考来源")[0]
