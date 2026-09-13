"""Regressions from actual report failures; fixtures remain independent of network/cache."""
import pytest

from agent.evidence import (
    clean_page_text, content_terms, evidence_blocks, page_problem, select_evidence,
)
from agent.factcheck import missing_numbers, cited_sentences
from agent.knowledge import KnowledgeBase, Source, source_material
from agent.search.pages import (
    PageDocument, _checked, _strip_tags, extract_document, trafilatura,
)


CUSTOMS_TABLE = """| Major Export Commodities in Quantity and Value,1.2026 | | | | | |
| | | | | Unit:US$1,000 | |
| Commodity | Quantity Unit | 1 | | Percentage Change | |
| | | Quantity | Value | Quantity | Value |
| Meat(including meat offal) | 10000T | 10 | 224,201 | 83.5 | 23.3 |
| Tea | T | 39,656 | 153,522 | -8.9 | 1.9 |
"""


def test_content_terms_strip_number_with_unit():
    # 「2026年毕业」若只去数字会残留「年」，拼出「年毕」这种跨界二元组——
    # 讲「2026届毕业生」的对题页面永远不含「年毕」，会被相关性门槛误杀
    terms = content_terms("调研2026年毕业情况")
    assert "毕业" in terms
    assert "年毕" not in terms
    assert not any("2026" in t for t in terms)


def test_content_terms_drop_filler_words():
    # 「情况/分析/调研」等泛词不算实义匹配——任何页面都可能含这些词
    terms = content_terms("调研分析出口情况")
    assert "出口" in terms
    assert "情况" not in terms and "分析" not in terms


def test_content_terms_do_not_cross_particles():
    # 「的毕业」不能拼出「的毕」：页面质检要求问题二元组全部命中，
    # 伪二元组会把只写"毕业生"的权威正文整类拒收，反而放行照抄问题措辞的
    # SEO 聚合站——实测导致人民网/新华网正文被拒、小网站被收录。
    terms = content_terms("2026年的毕业情况")
    assert "毕业" in terms
    assert "的毕" not in terms
    # 「在手机上」按虚词切分后仍保留「手机」这一实义二元组
    phone = content_terms("端侧大模型在手机上的应用")
    assert "手机" in phone
    assert "在手" not in phone


def test_recommendation_tail_removed():
    body = "2025年我国进出口总值超过45万亿元。" * 3
    out = clean_page_text(body + "\n为你推荐\n机器人运动会举行。")
    assert out == body


def test_cached_script_page_rejected():
    script = "var glb; window.prototype = function() { return Reflect.construct(); };" * 20
    with pytest.raises(ValueError, match="脚本"):
        _checked(PageDocument(script))


def test_truncated_script_does_not_leak_through_fallback():
    assert _strip_tags("<article><p>有效正文</p></article><script>var injected = 1;") == "有效正文"


def test_ordinary_article_not_rejected():
    assert not page_problem("端侧大模型需要兼顾内存容量和推理速度，量化能够减少权重内存，但仍需评估精度损失与功耗约束。")


def test_html_metadata_kept_separate_from_data_period():
    html = ('<html><head><meta property="article:published_time" content="2026-01-14"/>'
            '<title>2025年进出口</title></head><body><article><p>'
            + "2025年我国进出口总值超过45万亿元，全年进出口保持增长。" * 3
            + "</p></article></body></html>")
    page = extract_document(html, "https://example.com/report")
    assert "2025年" in page.text
    assert page.publisher
    assert page.retrieved_at


def _article_html(meta_date: str) -> str:
    return ('<html><head>'
            f'<meta property="article:published_time" content="{meta_date}"/>'
            '<title>高校毕业生如何赢得就业主动权</title></head><body><article><p>'
            + "高校毕业生就业状况关系社会万千家庭的获得感与安全感，也影响教育投入能否转化为人力资本优势。" * 3
            + "</p></article></body></html>")


@pytest.mark.skipif(trafilatura is None, reason="需要 trafilatura 解析 meta 日期")
def test_meta_date_used_when_year_agrees_with_url():
    # 年份一致时采信 meta（保留到日的精度）
    page = extract_document(_article_html("2026-04-15"),
                            "https://www.news.cn/202606/30/x.html")
    assert page.published_at == "2026-04-15"


def test_meta_date_overridden_by_url_when_year_conflicts():
    # 实测回归：光明日报电子版 URL 是 /202606/30/，页脚版权让解析器得出
    # 2025-01-01，并被原样写进报告"发布于 2025-01-01"。URL 是发布系统按日期
    # 生成的路径，更可信，年份矛盾时以它为准。
    page = extract_document(
        _article_html("2025-01-01"),
        "https://epaper.gmw.cn/gmrb/html/content/202606/30/content_17988.html")
    assert page.published_at == "2026-06-30"


def test_published_at_empty_when_no_date_anywhere():
    page = extract_document(_article_html(""),
                            "https://example.com/a/b.html")
    assert page.published_at == ""


def test_customs_units_and_columns_are_decoded_without_model():
    evidence = "\n".join(evidence_blocks(CUSTOMS_TABLE))
    assert "2026年1月出口商品" in evidence
    assert "数量100000吨" in evidence
    assert "金额224201000美元" in evidence
    assert "数量同比-8.9%" in evidence
    assert "金额同比1.9%" in evidence
    assert "224,201百万美元" not in evidence


@pytest.mark.parametrize("claim,source", [
    ("2026年出口45万亿元[1]", "2025年出口45万亿元，2026年1月14日公布。"),
    ("肉类价值224,201百万美元[1]", "肉类价值224,201千美元。"),
    ("出口30万辆[1]", "出口130万辆。"),
    ("茶叶金额下降8.9%[1]", "茶叶数量同比-8.9%；金额同比1.9%。"),
    ("同比增长8.9%[1]", "同比下降8.9%。"),
])
def test_wrong_numeric_claims_are_flagged(claim, source):
    assert missing_numbers(claim, source, claim)


@pytest.mark.parametrize("claim,source", [
    ("肉类金额2.24201亿美元[1]", "肉类金额224201000美元。"),
    ("约107万辆[1]", "出口106.9万辆。"),
    ("5191名学子毕业[1]", "5191人毕业。"),
    ("茶叶金额同比1.9%[1]", CUSTOMS_TABLE),
])
def test_equivalent_units_and_valid_values_are_kept(claim, source):
    assert missing_numbers(claim, source, "") == []


def test_citation_after_period_belongs_to_preceding_claim():
    sentences = cited_sentences("第一句出口增长。[1] 第二句进口下降。[2]")
    assert sentences == [("第一句出口增长[1] 。".replace(" ", ""), [1]),
                         ("第二句进口下降[2]。", [2])]


def test_duplicate_republication_is_one_source():
    kb = KnowledgeBase()
    first = kb.add("u1", "标题一", "摘要一", text="2025年出口45万亿元。")
    second = kb.add("u2", "标题二", "摘要二", text="2025年出口45万亿元。")
    assert first == second
    assert len(kb) == 1
    assert kb._by_url["u2"] == first


def test_writer_material_uses_original_evidence_not_bad_digest():
    kb = KnowledgeBase()
    kb.add("u1", "统计公报", "2026年出口45万亿元", text="2025年出口45万亿元。")
    material = source_material(kb.sources, "出口")
    assert "2025年出口45万亿元" in material
    assert "2026年出口45万亿元" not in material


def test_excerpt_keeps_sentence_and_unit():
    source = "汽车出口金额增长。\n" + "无关背景。" * 100 + "\n肉类金额224201000美元。"
    excerpt = select_evidence(source, "肉类金额", 100)
    assert "肉类金额224201000美元。" in excerpt


def test_section_terms_outweigh_question_terms_in_excerpt():
    # 实测缺口：问题词在长文的每个段落都命中，把小节特征词淹没——「未来展望」
    # 从展望专稿里只选出总量段落，模型看不到前瞻内容，整节占位。
    # 干扰段命中全部问题词；展望段只沾一个问题词，靠前瞻标记胜出。
    # 权重 3 意味着一个标记词抵三个问题词——只含单个标记的句子仍会打平，
    # 与真实展望段落一致地放两个标记（实测那段含"下半年""后续""接下来"）。
    text = ("\n".join(f"2026年我国进出口总值增长，出口规模扩大，进出口第{i}段。" for i in range(20))
            + "\n从下半年的形势判断来看，我国外贸压力仍在，后续需打造新的增长引擎。")
    question, section = "2026年我国进出口情况", "未来展望"
    plain = select_evidence(text, f"{question} {section}", 200)
    boosted = source_material(
        [Source(1, "u", "贸易形势展望", "", text)], f"{question} {section}", 200, section=section)
    assert "从下半年的形势判断来看" not in plain
    assert "从下半年的形势判断来看" in boosted
