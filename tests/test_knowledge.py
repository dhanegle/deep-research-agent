"""agent.knowledge 相关性过滤测试：token 覆盖率、ASCII 必命中、bigram 选材。

搜索结果相关性过滤是报告质量的关键——查专有名词时搜索引擎常返回泛主题页，
3B 照单全收会跑题。这些测试锁定过滤规则的核心行为。
"""
from agent.knowledge import (
    KnowledgeBase, _bigrams, query_tokens, token_coverage, ascii_token_hit,
)


class TestQueryTokens:
    def test_ascii_word_weighted_higher(self):
        toks = query_tokens("linuxsb 介绍")
        d = dict(toks)
        assert d["linuxsb"] == 1.5  # ASCII 专有名词高权重
        assert d["介绍"] == 1.0      # 中文段标准权重

    def test_year_is_ascii_token(self):
        toks = query_tokens("2025 新能源汽车出口")
        d = dict(toks)
        assert "2025" in d and d["2025"] == 1.5

    def test_pure_chinese(self):
        toks = query_tokens("固态电池")
        d = dict(toks)
        assert d["固态电池"] == 1.0
        assert not any(k.isascii() for k in d)


class TestTokenCoverage:
    def test_full_match(self):
        toks = query_tokens("linuxsb")
        assert token_coverage("linuxsb 社区", toks) == 1.0

    def test_no_match(self):
        toks = query_tokens("linuxsb")
        assert token_coverage("欧洲旅游攻略", toks) == 0.0

    def test_partial_chinese(self):
        # query_tokens 按空格切段，≥7 字的中文段才走 4 元组部分命中
        toks = query_tokens("固态电池量产进展与挑战")
        cov = token_coverage("固态电池量产情况", toks)
        assert 0 < cov < 1.0

    def test_empty_tokens(self):
        assert token_coverage("anything", []) == 1.0

    def test_separator_normalization(self):
        # linux.sb 的分隔符被归一后应匹配 linuxsb
        toks = query_tokens("linuxsb")
        assert token_coverage("linux.sb 论坛", toks) == 1.0


class TestAsciiTokenHit:
    def test_ascii_present_and_hit(self):
        toks = query_tokens("linuxsb")
        assert ascii_token_hit("linuxsb 论坛", toks) is True

    def test_ascii_present_not_hit(self):
        toks = query_tokens("linuxsb")
        # 含 ASCII token 但结果不含它 → False（泛词不放过）
        assert ascii_token_hit("Linux 操作系统介绍", toks) is False

    def test_no_ascii_tokens(self):
        # 纯中文查询恒为 True（不强制 ASCII 命中）
        toks = query_tokens("固态电池")
        assert ascii_token_hit("任意文本", toks) is True

    def test_separator_normalization(self):
        toks = query_tokens("linux.sb")
        assert ascii_token_hit("linuxsb 论坛", toks) is True


class TestBigrams:
    def test_basic(self):
        assert "固态" in _bigrams("固态电池")
        assert "态电" in _bigrams("固态电池")
        assert "电池" in _bigrams("固态电池")

    def test_single_char(self):
        assert _bigrams("a") == set()


class TestKnowledgeBaseSelect:
    def _kb(self):
        kb = KnowledgeBase()
        kb.add("u1", "固态电池量产进展", "硫化物路线装车")
        kb.add("u2", "欧洲旅游攻略", "巴黎景点推荐")
        return kb

    def test_select_relevant(self):
        kb = self._kb()
        selected = kb.select_for("固态电池", "固态电池产业化", k=5)
        assert selected[0].title == "固态电池量产进展"

    def test_has_section_material(self):
        kb = self._kb()
        assert kb.has_section_material("固态电池") is True
        assert kb.has_section_material("量子计算") is False
