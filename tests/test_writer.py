"""agent.writer 引用合规测试：结构化生成、编号白名单、跑题占位、幻觉引用剔除。

写作层是报告质量的格式防线——观点由模型给出，引用由程序排版：
每条观点的来源编号必须落在本节注入资料的白名单内，非法条目回喂重试一次；
整节跑题时纠偏，纠不回则明确占位。这些测试锁定该行为不被回归。
"""
import re

from pydantic import ValidationError

from agent import config
from agent.writer import (
    DEDUP_PLACEHOLDER, PLACEHOLDER, CitedClaim, SectionDraft, _clean_section_body,
    _dedupe_sections, _generate_section, _iter_report, _is_duplicate, _material_ids,
    _norm_sentence, _numbers, _section_aligned, _split_sentences, _validate_citations,
    _validate_report, add_findings, dedupe_sections,
)
from agent.evidence import aspect_terms
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


def _draft(*claims: tuple[str, list[int]]) -> SectionDraft:
    return SectionDraft(claims=[CitedClaim(text=t, source_ids=ids) for t, ids in claims])


class ScriptedLLM:
    """按预定 SectionDraft（或异常）逐次返回，记录调用次数与末次消息。"""

    def __init__(self, drafts: list):
        self.drafts = list(drafts)
        self.n_calls = 0
        self.last_messages = []

    def chat_json(self, messages, schema, tag):
        self.n_calls += 1
        self.last_messages = messages
        item = self.drafts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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

    def test_aspect_synonym_counts_as_aligned(self):
        # 实测误杀：「商品结构与特点」的合法正文写"高技术产品出口"，
        # 标题二元组一个不命中，被判跑题整节丢弃。aspect 组的"产品"应能对上
        body = "高技术产品出口增长35.9%，高端装备与新材料增速领先[7]。"
        assert _section_aligned(body, "商品结构与特点", "2026年我国进出口情况") is True
        # 反例不受影响：家电正文与"贸易伙伴分析"的 aspect 组（伙伴/国别…）同样不命中
        assert _section_aligned("家电出口突破1000亿元。", "贸易伙伴分析", "2026年出口") is False

    def test_aspect_word_inside_book_title_does_not_align(self):
        # 实测误放行：总量复述抄了"《…市场供需…报告》指出"式导语，
        # 书名里的"市场"不代表句子在讲贸易伙伴——aspect 匹配须剔除《…》
        body = ("中国报告大厅发布的《2026-2031年中国外贸行业市场供需及重点企业投资"
                "评估研究分析报告》指出，我国外贸规模实现历史性突破[3]。")
        assert _section_aligned(body, "贸易伙伴分布", "2026年我国进出口情况") is False

    def test_keeps_full_body_that_merely_mentions_missing_data(self):
        # 正文写满了，只是捎带一句"暂缺"——不该把整节有效内容丢掉
        body = ("2025年新能源汽车出口达343万辆[1]，同比增长70%[1]；"
                "欧洲与东南亚为主要目的地[2]。分车型的细分数据暂缺。")
        assert _clean_section_body(body) == body

    def test_collapses_placeholder_with_trailing_noise(self):
        assert _clean_section_body("（本节暂缺相关资料）\n\n") == PLACEHOLDER

    def test_material_ids_only_reads_entry_headers(self):
        assert _material_ids("[1] 《甲》\n摘要含[9]\n\n[2] 《乙》\n摘要") == {1, 2}

    def test_section_specific_terms_required(self):
        # 实测缺口：材料是央媒的总人数口径，正文却写成"就业比例"，
        # 仅靠与问题共享的主题词"毕业"就通过了扣题检查（该节整节跑题）
        off_topic = ("根据《2026年中国大学生就业报告》，2025届本科毕业生在电子电气设备"
                     "制造业的就业比例较2021届提高了0.8个百分点[6]。")
        assert _section_aligned(off_topic, "毕业总人数预测", "2026年的毕业情况") is False
        # 反向不误杀：权威正文写"规模预计1270万人"，不含"总人数"字面，
        # 但 aspect 组里的"规模/万人"能对上
        assert _section_aligned("2026届全国高校毕业生规模预计1270万人[1]。",
                                "毕业总人数预测", "2026年的毕业情况") is True

    def test_section_fully_covered_by_question_stays_loose(self):
        # 小节词全被问题覆盖（无独有词）时保持宽松，避免误杀合法释义
        assert _section_aligned("深圳信息职业技术大学5191名学子毕业，去向落实率95.76%。",
                                "毕业生总体情况", "调研2026年毕业生数据") is True

    def test_statistical_aspect_terms_available(self):
        # 「人数/预测」须入 aspect 表，否则 align_feedback 给不出纠偏提示
        hints = aspect_terms("毕业总人数预测")
        assert "人数" in hints and "预测" in hints and "规模" in hints


class TestGenerateSection:
    def test_valid_first_draft_single_call(self):
        llm = ScriptedLLM([_draft(("出口总量达到343万辆", [1]))])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert "出口总量达到343万辆[1]。" in out
        assert llm.n_calls == 1

    def test_program_cites_every_sentence(self):
        # 模型把两句捆在一条 claim 里：程序按句拆开，每句都补引用
        llm = ScriptedLLM([_draft(("出口总量达到343万辆。同比增长70%", [1]))])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert out.count("[1]") == 2

    def test_strips_model_written_cites_before_relayout(self):
        # 模型违规在 text 里自己写了 [1]：剔除后由程序统一排版，不重复引用
        llm = ScriptedLLM([_draft(("出口总量达到343万辆[1]", [1]))])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert out.count("[1]") == 1

    def test_hallucinated_id_retries_then_uses_valid(self):
        # 本节只注入 [1]，模型却引 [7]：回喂重试，采用重试后的合法稿
        llm = ScriptedLLM([
            _draft(("出口总量达到343万辆", [7])),
            _draft(("出口总量达到343万辆", [1])),
        ])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要", {1})
        assert "[7]" not in out
        assert "[1]" in out
        assert llm.n_calls == 2

    def test_invalid_both_attempts_yields_placeholder(self):
        llm = ScriptedLLM([
            _draft(("出口总量达到343万辆", [7])),
            _draft(("出口总量达到343万辆", [8])),
        ])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要", {1})
        assert out == PLACEHOLDER
        assert llm.n_calls == 2

    def test_title_echo_claim_rejected(self):
        # 3B 偶尔把小节标题当事实句输出（短标题被 schema 最小长度拦下，
        # 长标题靠代码层比对拒绝）：拒绝并重试
        llm = ScriptedLLM([
            _draft(("出口总量与增长趋势", [1])),
            _draft(("出口总量达到343万辆", [1])),
        ])
        out = _generate_section(llm, "出口", "出口总量与增长趋势", "[1] 摘要")
        assert "343万辆" in out
        assert llm.n_calls == 2

    def test_align_retry_names_section_keywords(self):
        # 实测缺口：回喂只说“和标题对不上”，3B 逐字重发上一稿，整节占位。
        # 回喂里要点名本节 aspect 词，模型才知道该回资料里找哪类事实。
        llm = ScriptedLLM([
            _draft(("民营企业进出口14.53万亿元，占外贸总值57%", [1])),
            _draft(("高技术产品出口占比提升至28%", [1])),
        ])
        out = _generate_section(llm, "进出口", "商品结构与特点", "[1] 摘要")
        feedback = llm.last_messages[-1]["content"]
        assert "商品结构与特点" in feedback
        assert "品类" in feedback and "类别" in feedback
        assert "高技术产品出口占比提升至28%" in out

    def test_generic_title_word_does_not_pass_alignment(self):
        # 实测缺口：「主要商品类别」整节写的是贸易伙伴排名，商品/品类/类别
        # 一个不命中，却靠标题泛词"主要"和 aspect 词"占比"通过了扣题检查。
        partner = "2026年1-2月中国对主要贸易伙伴的进出口金额排名中，美国排名第一，占比7.88%[4]。"
        assert _section_aligned(partner, "主要商品类别") is False
        assert _section_aligned(partner, "贸易伙伴分布") is True
        assert _section_aligned("2026年上半年机电产品出口占出口总额的63.5%[3]。",
                                "主要商品类别") is True

    def test_findings_drop_duplicate_sentence(self):
        # 跑题的小节会复述别节的句子，逐字相同的发现被列两遍（实测两份报告）。
        claims = [
            {"section": "贸易伙伴分布", "sentence": "美国排名第一，占比7.88%[4]。", "verdict": "支持"},
            {"section": "主要商品类别", "sentence": "美国排名第一，占比7.88%[4]。", "verdict": "支持"},
        ]
        out = add_findings("# 标题\n\n## 贸易伙伴分布\n\n正文", claims)
        assert out.count("美国排名第一") == 1

    def test_empty_claims_is_placeholder_single_call(self):
        llm = ScriptedLLM([_draft()])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert out == PLACEHOLDER
        assert llm.n_calls == 1

    def test_json_failure_yields_placeholder(self):
        llm = ScriptedLLM([ValueError("结构化输出失败")])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert out == PLACEHOLDER
        assert llm.n_calls == 1

    def test_allowed_ids_default_to_material_headers(self):
        material = "[1] 《甲》\n摘要提到[9]这个编号\n\n[2] 《乙》\n摘要"
        llm = ScriptedLLM([
            _draft(("出口总量增长", [9])),
            _draft(("出口总量增长", [2])),
        ])
        out = _generate_section(llm, "出口", "出口总量", material)
        assert "[9]" not in out
        assert "[2]" in out
        assert llm.n_calls == 2

    def test_keeps_best_draft_when_retry_shrinks(self):
        # 首稿 2 条合法 + 1 条非法；重试只回 1 条：保留更完整的首稿合法部分
        llm = ScriptedLLM([
            _draft(("出口总量达到343万辆", [1]), ("同比增长70%", [1]), ("编号非法的观点", [9])),
            _draft(("出口总量达到343万辆", [1])),
        ])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要", {1})
        assert "343万辆" in out and "70%" in out
        assert llm.n_calls == 2

    def test_off_topic_becomes_placeholder(self):
        # 标题是贸易伙伴，观点却全是家电 → 跑题纠偏失败后占位
        llm = ScriptedLLM([
            _draft(("家电出口突破1000亿元", [1])),
            _draft(("家电和体育用品出口增长", [1])),
        ])
        out = _generate_section(llm, "2026年出口", "贸易伙伴分析", "[1] 家电摘要")
        assert out == PLACEHOLDER
        assert llm.n_calls == 2

    def test_feedback_reaches_prompt(self):
        # 阶段 E 修订：自审发现的问题要注入到重写请求里
        llm = ScriptedLLM([_draft(("出口总量达到343万辆", [1]))])
        _generate_section(llm, "出口", "出口总量", "[1] 摘要",
                          feedback="原文不支持数值 45万亿")
        assert any("45万亿" in m.get("content", "") for m in llm.last_messages)

    def _kb_two_sources(self, second_text: str):
        kb = KnowledgeBase()
        kb.add("u1", "出口再创新高", "摘要",
               text="2025年上半年新能源汽车出口总量约106万辆，同比增长16.2%。")
        kb.add("u2", "第二来源", "摘要", text=second_text)
        return kb

    def test_prunes_citations_unsupported_by_evidence(self):
        # 实测缺口：3B 把本节注入的编号一股脑挂在每句后面（106万辆挂 [1][2]），
        # 阶段 E 能查出来但回喂修不掉，最后只能标成待核验。成文时就按证据剪掉。
        kb = self._kb_two_sources("奇瑞全年出口预计超过120万辆，保持车企出口第一。")
        llm = ScriptedLLM([_draft(("2025年上半年新能源汽车出口总量约106万辆", [1, 2]))])
        out = _generate_section(llm, "新能源汽车出口", "出口总量", "[1] 摘要\n\n[2] 摘要",
                                {1, 2}, sources=kb.sources)
        assert "[1]" in out and "[2]" not in out

    def test_no_retry_when_section_draws_on_one_source_only(self):
        # 试过"整节只用 1 条资料就点名未用来源催补一次"：实测 3B 无视"保留已有
        # 内容"整段重写，程序合并后同一批事实被塞进多个小节（东南亚内容进了
        # 3 节），调用数 +3、耗时 +6s、覆盖率不变。结论是不催补，见 README。
        llm = ScriptedLLM([_draft(("出口总量达到343万辆", [1]))])
        _generate_section(llm, "出口", "出口总量", "[1] 摘要\n\n[2] 摘要", {1, 2})
        assert llm.n_calls == 1

    def test_keeps_multi_citation_when_each_source_supports_it(self):
        # 只剪对本句任何数字都没有支撑的编号：两条来源都写着同一数据的
        # 合法多引用不动，否则会把"各自佐证"的正常引用拆掉
        kb = self._kb_two_sources("海关数据显示出口总量为106万辆。")
        llm = ScriptedLLM([_draft(("2025年上半年新能源汽车出口总量约106万辆", [1, 2]))])
        out = _generate_section(llm, "新能源汽车出口", "出口总量", "[1] 摘要\n\n[2] 摘要",
                                {1, 2}, sources=kb.sources)
        assert "[1]" in out and "[2]" in out

    def test_no_sources_means_no_pruning(self):
        # 不传 sources（如单测与旧调用）时行为不变，模型给的编号原样保留
        llm = ScriptedLLM([_draft(("出口总量达到343万辆", [1, 2]))])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要\n\n[2] 摘要")
        assert "[1]" in out and "[2]" in out


class TestSplitSentences:
    """书名号/引号内的句号不是句界——在《…。》中间断句会把引用编号插进标题。"""

    def test_period_inside_book_title_not_a_boundary(self):
        text = "教育部发布《2026届毕业生就业报告。总卷》并公布数据。第二句独立成句。"
        parts = _split_sentences(text)
        assert parts == ["教育部发布《2026届毕业生就业报告。总卷》并公布数据。", "第二句独立成句。"]

    def test_period_inside_quote_not_a_boundary(self):
        parts = _split_sentences("报告称“出口增长。前景乐观”并给出数据。")
        assert parts == ["报告称“出口增长。前景乐观”并给出数据。"]

    def test_unbalanced_closer_does_not_swallow_rest(self):
        # 模型偶尔只输出后半个书名号：深度被钳在 0，后续句号照常断句
        parts = _split_sentences("残缺》第一句。第二句。")
        assert parts == ["残缺》第一句。", "第二句。"]

    def test_citation_lands_after_title_not_inside(self):
        # 整条链路验证：程序补引用时不得把 [1] 插进《…。》中间
        llm = ScriptedLLM([_draft(("海关发布《进出口统计。月度版》显示出口增长。同比增长70%", [1]))])
        out = _generate_section(llm, "出口", "出口总量", "[1] 摘要")
        assert "《进出口统计。月度版》显示出口增长[1]。" in out
        assert "统计[1]" not in out


class TestSectionDraftSchema:
    def test_claim_requires_source_ids(self):
        try:
            CitedClaim(text="没有来源编号的观点", source_ids=[])
        except ValidationError:
            return
        raise AssertionError("空 source_ids 应当校验失败")


class TestIterReportEmptySection:
    def test_unrelated_sources_yield_placeholder(self):
        kb = KnowledgeBase()
        kb.add("u1", "欧洲旅游攻略", "巴黎景点推荐")
        plan = Plan(outline=["量子计算进展", "量子计算挑战", "量子计算趋势"],
                    search_queries=["量子计算 进展", "量子计算 挑战"])
        # 无证据的章节直接说明缺口，不再浪费调用生成无关内容。
        llm = ScriptedLLM([])
        text = "".join(_iter_report(llm, "量子计算产业化", plan, kb))
        assert text.count(PLACEHOLDER) == 3
        assert "巴黎" not in text.split("## 参考来源")[0]
        assert llm.n_calls == 0


def _report(*sections: tuple[str, str]) -> str:
    body = "".join(f"## {title}\n\n{text}\n\n" for title, text in sections)
    return "# 2026年我国进出口情况\n\n" + body + "## 参考来源\n\n[1] t1 u1\n[2] t2 u2\n[3] t3 u3\n"


class TestDedupeSections:
    """成文后的跨节去重：同一句只在扣题度最高的小节保留一处。

    实测 143 份报告中 59% 存在跨节重复，正文句子重复率 18.0%，最严重的
    一份达 65%；重复对集中在「出口总量与增速」→「未来出口趋势预测」这类
    相邻小节。生成时干预会挤占上下文（见 README 的合并实验），故放在成文后。
    """

    def test_norm_flattens_filler_before_numbers(self):
        # 3B 复述同一事实字面常有出入："预计为1270万人" vs "预计1270万人"
        assert _norm_sentence("规模预计为1270万人[1]。") == _norm_sentence("规模预计1270万人[2]。")
        # 只抹平紧邻数字的虚词，"作为/达到"这类正常用词不受影响
        assert _norm_sentence("作为第一大市场") != _norm_sentence("第一大市场")

    def test_norm_strips_attribution_prefix(self):
        # 3B 习惯写"根据X的报道，事实"，同一事实在别节常换个出处或干脆不写
        assert (_norm_sentence("根据新华网和人民网的报道，2026届毕业生规模预计为1270万人[1]。")
                == _norm_sentence("2026届毕业生规模预计为1270万人[1]。"))
        # 句中没有逗号收尾的"表示/称"不作分界，避免误剥有效内容
        assert _norm_sentence("该企业表示将扩大出口") == "该企业表示将扩大出口"

    def test_removes_duplicate_differing_only_by_attribution(self):
        out = dedupe_sections(_report(
            ("毕业总人数预测",
             "根据新华网和人民网的报道，2026届毕业生规模预计为1270万人[1]。网易预测为1593万[2]。"),
            ("毕业生就业意愿", "2026届毕业生规模预计为1270万人[1]。教育部部署了促就业措施[3]。"),
        ), "2026年的毕业情况")
        assert out.count("1270万人") == 1

    def test_removes_duplicate_from_later_section(self):
        out = dedupe_sections(_report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。同比增长16.2%[1]。"),
            ("未来出口趋势预测", "2025年上半年新能源汽车出口总量约106万辆[1]。预计全年保持增长[2]。"),
        ), "2026年出口")
        assert out.count("出口总量约106万辆") == 1
        assert out.count("同比增长16.2%") == 1 and out.count("预计全年保持增长") == 1

    def test_keeps_duplicate_in_best_aligned_section_not_first(self):
        # 归属由扣题度决定：伙伴排名句该留在「贸易伙伴分布」，即便它先出现在
        # 「主要商品类别」。若一律"先到先得"，会把该句留在错的小节。
        partner = "2026年1-2月中国对主要贸易伙伴的进出口金额排名中，美国排名第一，占比7.88%[3]。"
        out = dedupe_sections(_report(
            ("主要商品类别", "2026年上半年机电产品出口占出口总额的63.5%[1]。" + partner),
            ("贸易伙伴分布", partner),
        ), "2026年我国进出口情况")
        assert out.count("美国排名第一") == 1
        head, _, tail = out.partition("## 贸易伙伴分布")
        assert "美国排名第一" in tail and "美国排名第一" not in head

    def test_fully_duplicated_section_becomes_placeholder(self):
        # 整节都是别节的复述：不保留重复段落，改标注缺口（阶段 E 据此补搜重写）。
        # 实测 3B 把一句政策话塞进「就业市场分析/毕业生就业意愿/政策与经济影响」
        # 三节，保留重复只会让同一段话出现三次。
        out = dedupe_sections(_report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。"),
            ("未来出口趋势预测", "2025年上半年新能源汽车出口总量约106万辆[1]。"),
        ), "2026年出口")
        assert out.count("出口总量约106万辆") == 1
        assert f"## 未来出口趋势预测\n\n{DEDUP_PLACEHOLDER}" in out

    def test_short_but_real_sentence_keeps_section_alive(self):
        # 反向不误标：去重后只剩一句短但真实的句子，不算缺口，不得改成占位
        out = dedupe_sections(_report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。同比增长16.2%[1]。"),
            ("未来出口趋势预测", "2025年上半年新能源汽车出口总量约106万辆[1]。预计增长[2]。"),
        ), "2026年出口")
        assert PLACEHOLDER not in out
        assert "预计增长" in out

    def test_stray_citation_does_not_count_as_content(self):
        # 实测缺口：整节实质句都被判重、只剩一个游离的 [1] 时，若把 [1] 当"还有
        # 内容"，守卫不触发，小节会只剩一个引用编号——看不出内容又像缺证据
        out = dedupe_sections(_report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。"),
            ("未来出口趋势预测", "2025年上半年新能源汽车出口总量约106万辆[1]。\n\n[1]"),
        ), "2026年出口")
        assert out.count("出口总量约106万辆") == 1
        assert f"## 未来出口趋势预测\n\n{DEDUP_PLACEHOLDER}" in out

    def test_findings_section_is_passthrough(self):
        # 「主要发现」是正文摘录，参与去重会把事实从正文小节里删掉
        out = dedupe_sections(_report(
            ("主要发现", "- 2025年上半年新能源汽车出口总量约106万辆[1]。"),
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。"),
        ), "2026年出口")
        assert out.count("出口总量约106万辆") == 2
        assert "## 主要发现" in out

    def test_dedupe_removes_same_section_repeat(self):
        out = dedupe_sections(_report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。"
                              "2025年上半年新能源汽车出口总量约106万辆[1]。"),
        ), "2026年出口")
        assert out.count("出口总量约106万辆") == 1

    def test_distinct_numbers_not_merged(self):
        report = _report(
            ("出口总量与增速", "2026年出口总量预计1270万辆[1]。"),
            ("未来出口趋势预测", "2026年出口总量预计1300万辆[1]。"),
        )
        assert dedupe_sections(report, "2026年出口") == report

    def test_untouched_when_no_duplicates(self):
        report = _report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。"),
            ("未来出口趋势预测", "预计全年出口将保持两位数增长[2]。"),
        )
        assert dedupe_sections(report, "2026年出口") == report

    def test_preserves_footer_and_title(self):
        out = dedupe_sections(_report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。"),
            ("未来出口趋势预测", "2025年上半年新能源汽车出口总量约106万辆[1]。预计增长[2]。"),
        ), "2026年出口")
        assert out.startswith("# 2026年我国进出口情况\n\n## 出口总量与增速")
        assert "## 参考来源" in out and "[1] t1 u1" in out and "[3] t3 u3" in out

    def test_idempotent_and_counted(self):
        report = _report(
            ("出口总量与增速", "2025年上半年新能源汽车出口总量约106万辆[1]。同比增长16.2%[1]。"),
            ("未来出口趋势预测", "2025年上半年新能源汽车出口总量约106万辆[1]。预计全年保持增长[2]。"),
        )
        once, dropped = _dedupe_sections(report, "2026年出口")
        assert dropped == 1
        assert dedupe_sections(once, "2026年出口") == once

    def test_report_without_footer_untouched(self):
        report = "## 出口总量与增速\n\n某句事实[1]。\n\n## 未来出口趋势预测\n\n某句事实[1]。\n"
        assert dedupe_sections(report, "出口") == report


class TestDedupeNearDuplicates:
    """改述型重复：逐字比对抓不到，但同一事实在多个小节各写一遍同样要删。

    实测 3B 把同一句话在 3 个小节各写一遍，措辞差一个"普通"或换个归因说法
    （"…的报道" vs "…发布的消息"），字面不相等却是同一句。判据因此从"字面相同"
    放宽到"高相似 + 数字口径相容"。
    """

    def test_paraphrase_with_inserted_word_merged(self):
        out = dedupe_sections(_report(
            ("毕业总人数预测", "2026届全国普通高校毕业生规模预计为1270万人，同比增加48万人[1]。"),
            ("毕业生就业意愿",
             "2026届全国高校毕业生规模预计为1270万人，同比增加48万人[1]。教育部部署了促就业措施[2]。"),
        ), "2026年的毕业情况")
        assert out.count("1270万人") == 1
        assert "教育部部署了促就业措施" in out

    def test_decimal_point_kept_in_key(self):
        # 归一化若抹掉小数点，"16.2%"与"1.62%"会归一成同一个"162%"（两个不同数值）
        assert _norm_sentence("同比增长16.2%[1]。") != _norm_sentence("同比增长1.62%[1]。")
        report = _report(
            ("出口总量与增速", "2026年上半年出口同比增长16.2%[1]。"),
            ("未来出口趋势预测", "2026年上半年出口同比增长1.62%[1]。"),
        )
        assert dedupe_sections(report, "2026年出口") == report

    def test_numbers_must_be_compatible(self):
        # 数字口径不相容一律不判重：措辞几乎一样但数值不同，是两个事实
        a, b = _norm_sentence("2026年出口总量预计为1270万辆[1]。"), _norm_sentence("2026年出口总量预计为1300万辆[1]。")
        assert not _is_duplicate(a, _numbers(a), b, _numbers(b))

    def test_similarity_threshold_is_configurable(self):
        report = _report(
            ("毕业总人数预测", "2026届全国普通高校毕业生规模预计为1270万人，同比增加48万人[1]。"),
            ("毕业生就业意愿",
             "2026届全国高校毕业生规模预计为1270万人，同比增加48万人[1]。教育部部署了促就业措施[2]。"),
        )
        saved = config.DEDUPE_SIMILARITY
        try:
            config.DEDUPE_SIMILARITY = 1.0  # 关掉近似匹配，只去逐字重复
            assert dedupe_sections(report, "2026年的毕业情况") == report
        finally:
            config.DEDUPE_SIMILARITY = saved

    def test_removal_requires_direct_similarity_to_survivor(self):
        # 并查集把 A≈B≈C 连成一簇时，A 与 C 可能毫不相干——实测把"2024年招收
        # 硕士生118.57万人"和"毕业生规模1270万人"连到一起。删除必须以"与保留句
        # 直接相似"为前提，否则会误删不相关的事实。
        report = _report(
            ("出口总量与增速", "2026年出口总量预计为106万辆，同比增长16%[1]。"),
            ("商品结构与特点",
             "2026年出口总量预计为106万辆，同比增长16%，其中纯电动占比72%[1]。"),
            ("未来出口趋势预测",
             "2026年出口总量预计为106万辆，同比增长16%，其中纯电动占比72%，插混增速接近40%[1]。"),
        )
        out = dedupe_sections(report, "2026年出口")
        before = _sentences_by_section(report)
        survivors = _sentences_by_section(out)
        removed = [(sec, s) for sec, s in before if (sec, s) not in survivors]
        assert removed, "这一组应至少删掉一句"
        for _, sent in removed:
            key = _norm_sentence(sent)
            nums = _numbers(key)
            assert any(
                _is_duplicate(key, nums, _norm_sentence(s2), _numbers(_norm_sentence(s2)))
                for _, s2 in survivors), f"被删句找不到直接相似的保留句：{sent}"


def _sentences_by_section(report: str) -> list[tuple[str, str]]:
    """(小节, 句子) 列表，切句口径与去重一致：按空行分段、段内按句切分。"""
    out = []
    for chunk in report.split("## ")[1:]:
        title, _, body = chunk.partition("\n")
        if "参考来源" in title:
            continue
        for block in re.split(r"\n\s*\n", body):
            if block.strip():
                out.extend((title.strip(), s) for s in _split_sentences(block) if s)
    return out

