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
        # 资料不足也不以无关来源补足 k。
        selected = kb.select_for("量子计算", "固态电池产业化", k=5)
        assert selected == []

    def test_question_overlap_alone_not_enough(self):
        # 只对上问题词，不足以支持“活跃成员”这个小节。
        kb = KnowledgeBase()
        kb.add("u1", "linuxdo 邀请码获取", "填写50字申请即可加入社区")
        selected = kb.select_for("活跃成员", "linuxdo是什么社区", k=5)
        assert selected == []

    def test_keeps_only_section_relevant(self):
        kb = self._kb()
        selected = kb.select_for("固态电池", "固态电池产业化", k=5)
        assert [s.title for s in selected] == ["固态电池量产进展"]

    def test_prefers_section_4gram_over_generic_word(self):
        kb = KnowledgeBase()
        kb.add("u1", "货物贸易第一大国", "中国出口26.99万亿元，货物贸易顺差扩大")
        kb.add("u2", "中国前十二大贸易伙伴", "东盟、欧盟、美国是主要贸易伙伴，国别结构变化")
        selected = kb.select_for("贸易伙伴分析", "2026年我国的出口情况", k=5)
        assert selected[0].title == "中国前十二大贸易伙伴"

    def test_aspect_fallback_selects_structure_source(self):
        # 实测缺口：「商品结构与特点」的对口来源只命中标题二元组"结构"一个，
        # 旧 aspect 表没有商品/结构组，兜底条件不成立被整个排除
        kb = KnowledgeBase()
        kb.add("u1", "2026年中国外贸市场分析", "2026年上半年我国进出口总值25.47万亿元，高技术产品出口增长，外贸结构优化升级")
        selected = kb.select_for("商品结构与特点", "2026年我国进出口情况", k=5)
        assert [s.title for s in selected] == ["2026年中国外贸市场分析"]
        # 无关来源仍进不来：aspect 命中但问题词不足两个
        kb2 = KnowledgeBase()
        kb2.add("u2", "家居产品选购指南", "沙发与床垫的品类结构对比")
        assert kb2.select_for("商品结构与特点", "2026年我国进出口情况", k=5) == []

    def test_title_hit_outranks_diluted_body_aspect(self):
        # 实测缺口：「未来展望」节里，标题写明“展望”的光明网专稿与两条排名页
        # 同分（正文各偶然出现一次“趋势/预计”），靠 id 小被 k=2 截掉，整节占位。
        # 标题命中不会被长正文稀释，应排在正文 aspect 命中之前。
        kb = KnowledgeBase()
        kb.add("u1", "全国进出口贸易额及各国排名", "2026年进出口排名。" + "行业预计保持增长，趋势向好。" * 40)
        kb.add("u2", "中国外贸市场分析及发展趋势", "2026年我国进出口规模扩大。" + "未来趋势值得关注。" * 40)
        kb.add("u3", "2026年我国贸易形势分析及展望", "我国进出口全年有望保持增长，出口结构持续改善。")
        selected = kb.select_for("未来展望", "2026年我国进出口情况", k=2)
        assert "展望" in selected[0].title

    def test_has_section_material(self):
        kb = self._kb()
        assert kb.has_section_material("固态电池") is True
        assert kb.has_section_material("量子计算") is False


class TestSelectPrefersAuthority:
    """写作选材的权威优先：实测「毕业总人数预测」一节把人民网/新华网挡在门外，
    报告因此引用了自媒体的错误数字（1250 万）而非权威的 1270 万。"""

    def test_authority_gate_relaxed(self):
        # 权威媒体用编辑措辞（"规模预计1270万人"），字面命中低于照抄查询词的聚合站，
        # 门槛对权威放宽到 sec_hits>=1
        kb = KnowledgeBase()
        kb.add("https://edu.people.com.cn/n1/2025/1121/c.html",
               "2026届全国高校毕业生规模预计1270万人 - 人民网教育",
               "2026届全国高校毕业生规模预计1270万人，同比增加48万人。")
        kb.add("https://view.inews.qq.com/a/2025",
               "明年大学毕业生的人数比今年跳涨371万？这张图表数据错得离谱",
               "大学毕业生的人数 毕业生人数 数据")
        titles = [s.title for s in kb.select_for("毕业总人数预测", "2026年的毕业情况", k=3)]
        assert any("人民网" in t for t in titles)

    def test_authority_outranks_ordinary_when_comparable(self):
        kb = KnowledgeBase()
        kb.add("https://www.sohu.com/a/1", "2026年毕业生就业数据报告", "毕业生就业数据 分析")
        kb.add("https://www.news.cn/2025/x.html", "2026届高校毕业生就业数据发布", "毕业生就业数据发布")
        selected = kb.select_for("就业数据", "2026年毕业生就业情况", k=5)
        assert selected[0].title.startswith("2026届高校毕业生就业数据发布")

    def test_ordinary_sources_still_gated(self):
        # 门槛放宽只对权威生效：普通来源仍须满足原有相关性要求
        kb = KnowledgeBase()
        kb.add("https://www.gzchanjiao.com/a", "某培训机构就业分析", "毕业生就业情况概述")
        assert kb.select_for("量子计算", "2026年的毕业情况", k=5) == []
