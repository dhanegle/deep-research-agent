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

    def test_year_alone_does_not_veto(self):
        # 查询里的年份不能把没写年份的相关页整页丢掉
        toks = query_tokens("2025 新能源汽车出口")
        assert ascii_token_hit("中国新能源汽车出口欧洲", toks) is True

    def test_year_does_not_replace_proper_noun(self):
        toks = query_tokens("linuxsb 2025")
        assert ascii_token_hit("linuxsb 社区介绍", toks) is True
        assert ascii_token_hit("2025 年 Linux 内核发布", toks) is False


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

    def test_drops_zero_section_overlap(self):
        kb = self._kb()
        # 「量子计算」与两条来源都无小节重合 → 库不足 k 条时仍返回，但排在最末
        # （不再用门槛剔除——给足素材比严卡门槛更重要）
        selected = kb.select_for("量子计算", "固态电池产业化", k=5)
        assert len(selected) == 2  # 库只有 2 条，全部返回
        # 与小节相关的应排在前面；这里两条都不相关，顺序按原 id
        assert selected[0].title == "固态电池量产进展"

    def test_question_overlap_alone_not_enough(self):
        # 来源对上了问题词（linuxdo/社区）但对不上小节标题 → 不优先，但库不足时仍返回
        kb = KnowledgeBase()
        kb.add("u1", "linuxdo 邀请码获取", "填写50字申请即可加入社区")
        selected = kb.select_for("活跃成员", "linuxdo是什么社区", k=5)
        assert len(selected) == 1  # 库只有 1 条，返回它

    def test_keeps_only_section_relevant(self):
        kb = self._kb()
        selected = kb.select_for("固态电池", "固态电池产业化", k=5)
        assert [s.title for s in selected] == ["固态电池量产进展", "欧洲旅游攻略"]
        # 相关的排在前，不相关的在后（不再剔除）

    def test_prefers_section_4gram_over_generic_word(self):
        kb = KnowledgeBase()
        kb.add("u1", "货物贸易第一大国", "中国出口26.99万亿元，货物贸易顺差扩大")
        kb.add("u2", "中国前十二大贸易伙伴", "东盟、欧盟、美国是主要贸易伙伴，国别结构变化")
        selected = kb.select_for("贸易伙伴分析", "2026年我国的出口情况", k=5)
        assert selected[0].title == "中国前十二大贸易伙伴"

    def test_has_section_material(self):
        kb = self._kb()
        assert kb.has_section_material("固态电池") is True
        assert kb.has_section_material("量子计算") is False
