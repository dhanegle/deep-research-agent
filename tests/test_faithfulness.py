"""eval.faithfulness 规则层测试：数字抽取、数字存疑检测、引用句提取。

规则层是零成本的忠实度防线——抽取句子中的数字与被引来源比对，
找不到依据的记为存疑。这些测试锁定型号编号跳过、四舍五入容差、
题目自带数字豁免等关键行为。
"""
from eval.faithfulness import (
    extract_numbers, missing_numbers, cited_sentences, _iter_sentences,
)


class TestExtractNumbers:
    def test_plain_numbers(self):
        assert extract_numbers("出口107万辆，增长35.2%") == ["107", "35.2"]

    def test_skips_model_numbers(self):
        # Qwen2.5 / GPT-4 / 5G 紧邻字母 → 型号编号，跳过
        nums = extract_numbers("Qwen2.5和GPT-4以及5G技术")
        assert nums == []

    def test_skips_citation_brackets(self):
        # [1] 引用标记本身不算数字
        nums = extract_numbers("出口增长35%[1]")
        assert nums == ["35"]

    def test_mixed(self):
        nums = extract_numbers("2025年出口107万辆，Qwen2.5无关")
        assert nums == ["2025", "107"]

    def test_no_numbers(self):
        assert extract_numbers("没有数字的文字") == []


class TestMissingNumbers:
    def test_number_in_digest(self):
        assert missing_numbers("出口107万辆[1]", "出口107万辆", "") == []

    def test_number_in_question(self):
        # 题目自带的年份不算存疑
        assert missing_numbers("2025年情况[1]", "摘要无年份", "2025年出口情况") == []

    def test_rounded_tolerance(self):
        # 来源 106.9 万，句子写"约107万" → 四舍五入容差，不算存疑
        assert missing_numbers("约107万辆[1]", "出口106.9万辆", "") == []

    def test_genuinely_missing(self):
        # 来源没有 30，题目也没有 → 存疑
        assert missing_numbers("关税升至30%[1]", "欧洲关税壁垒", "") == ["30"]

    def test_decimal_in_sentence_no_tolerance(self):
        # 句子给了小数还匹配不上，不再宽容
        assert missing_numbers("增长35.2%[1]", "增长30%", "") == ["35.2"]


class TestCitedSentences:
    def test_extracts_cited(self):
        report = "# 标题\n\n> meta\n\n数据见[1]和[2]。\n\n## 参考来源\n\n[1] t\n[2] t\n"
        sents = cited_sentences(report)
        assert len(sents) == 1
        assert sents[0][1] == [1, 2]

    def test_skips_headings_and_meta(self):
        report = "# 标题\n> meta行\n## 章节\n正文[1]。\n## 参考来源\n[1] t\n"
        sents = cited_sentences(report)
        assert len(sents) == 1
        assert "正文" in sents[0][0]

    def test_empty_citation_no_chinese(self):
        # 只剩引用标记 [1] 单独成行 → 无可核验论断，跳过
        report = "## 章节\n正文。\n[1]\n## 参考来源\n[1] t\n"
        sents = cited_sentences(report)
        assert len(sents) == 0

    def test_stops_at_references(self):
        report = "## 章节\n正文[1]。\n## 参考来源\n[1] t\n[2] 也算句子[3]\n"
        sents = cited_sentences(report)
        assert len(sents) == 1  # 参考来源页脚里的不提取


class TestIterSentences:
    def test_splits_on_punctuation(self):
        report = "## 章节\n第一句。[1] 第二句。[2] 第三句。"
        sents = list(_iter_sentences(report))
        assert len(sents) == 3

    def test_skips_empty(self):
        report = "## 章节\n第一句。 。 第二句。"
        sents = list(_iter_sentences(report))
        assert len(sents) == 2

    def test_stops_at_references(self):
        # 真实报告中 "## 参考来源" 独占一行
        report = "## 章节\n正文。\n\n## 参考来源\n[1] 这里不应出现。"
        sents = list(_iter_sentences(report))
        assert any("正文" in s for s in sents)
        assert all("不应出现" not in s for s in sents)
