"""agent.researcher 路径容错测试：_resolve_path 的前缀剥离与后缀匹配。

这是 3B 模型最频繁的失败模式之一（给路径编造 https:// 前缀或截断），
参数命名 url→path + 容错匹配曾把同一任务的工具有效率从 39% 拉到 100%。
测试锁定该行为。
"""
from agent.researcher import Researcher, _date_tag, _strip_question_verbs
from agent.search.base import SearchResult


def _seen(*urls):
    return {u: f"标题{i}" for i, u in enumerate(urls)}


class _Trace:
    """最小 trace 桩：_filter_results 会记录被拒收的单位内部页。"""

    def __init__(self):
        self.events = []

    def log(self, event, **data):
        self.events.append((event, data))


def _bare(**attrs):
    """裸实例：绕过 __init__，只挂上被测方法真正用到的属性。"""
    r = Researcher.__new__(Researcher)
    r.trace = _Trace()
    for k, v in attrs.items():
        setattr(r, k, v)
    return r


class TestResolvePath:
    def test_exact_match(self):
        seen = _seen("data/doc.md")
        assert Researcher._resolve_path("data/doc.md", seen) == "data/doc.md"

    def test_strips_http_prefix(self):
        seen = _seen("data/doc.md")
        assert Researcher._resolve_path("https://example.com/data/doc.md", seen) == "data/doc.md"

    def test_strips_https_prefix(self):
        seen = _seen("data/doc.md")
        assert Researcher._resolve_path("https://example.com/data/doc.md", seen) == "data/doc.md"

    def test_suffix_match(self):
        seen = _seen("data/2025/出口情况.md")
        # 模型只给了后半段
        assert Researcher._resolve_path("出口情况.md", seen) == "data/2025/出口情况.md"

    def test_ambiguous_suffix_returns_none(self):
        seen = _seen("data/a/doc.md", "data/b/doc.md")
        # 两个都以后缀匹配，无法确定是哪个 → None
        assert Researcher._resolve_path("doc.md", seen) is None

    def test_no_match(self):
        seen = _seen("data/doc.md")
        assert Researcher._resolve_path("nonexistent.md", seen) is None

    def test_cleaned_exact(self):
        seen = _seen("docs/note.txt")
        # 剥掉协议域名后恰好命中
        assert Researcher._resolve_path("http://x.com/docs/note.txt", seen) == "docs/note.txt"


def _filter(query, results, question=None):
    r = _bare(_question=question or query)
    return r._filter_results(results, query)


def _anchor(query, question):
    r = _bare()
    r._core = _strip_question_verbs(question)[:14]
    return r._anchor_query(query)


class TestFilterResults:
    def test_keeps_proper_noun_even_if_coverage_low(self):
        # "linuxsb 主要 讨论 领域"：只命中 linuxsb 时覆盖率 1.5/4.5 < 0.35，
        # 旧逻辑会丢掉真正的主页。专有名词命中即应保留。
        results = [
            SearchResult("烧饼社区", "https://linux.sb", "人人都有饼吃的AI社区"),
            SearchResult("Linux 内核介绍", "https://example.com/linux", "什么是Linux操作系统"),
        ]
        kept = _filter("linuxsb 主要 讨论 领域", results)
        urls = [r.url for r in kept]
        assert "https://linux.sb" in urls
        assert "https://example.com/linux" not in urls

    def test_drops_generic_pages_without_proper_noun(self):
        results = [
            SearchResult("个人介绍网站", "https://github.com/foo/site", "网站介绍 个人主页"),
            SearchResult("Linux SKB", "https://blog.csdn.net/skb", "Linux 内核网络包"),
        ]
        kept = _filter("linuxsb 网站 介绍", results)
        assert kept == []

    def test_chinese_query_uses_coverage_floor(self):
        results = [
            SearchResult("出口总量再创新高", "u1", "2025年中国新能源汽车出口欧洲"),
            SearchResult("巴黎旅游攻略", "u2", "埃菲尔铁塔门票"),
        ]
        kept = _filter("新能源汽车出口 欧洲", results)
        assert [r.url for r in kept] == ["u1"]

    def test_keeps_partial_long_chinese_query(self):
        # 「中国出口贸易伙伴 国别分析」只命中「贸易伙伴」时覆盖率 0.25，旧门槛 0.35 会全灭
        results = [
            SearchResult("中国前十二大贸易伙伴", "https://news.example.com/partners", "东盟欧盟美国"),
            SearchResult("巴黎旅游攻略", "https://travel.example.com/paris", "埃菲尔铁塔"),
        ]
        kept = _filter("中国出口贸易伙伴 国别分析", results, question="2026年我国的出口情况")
        assert [r.url for r in kept] == ["https://news.example.com/partners"]

    def test_question_coverage_saves_paraphrase(self):
        # 搜索词「中国出口未来展望」整串对不上，但页面对调研问题很相关
        results = [
            SearchResult("出口高增长能延续吗", "https://news.example.com/outlook",
                         "2026年我国出口情况展望，增长能否延续"),
        ]
        kept = _filter("中国出口未来展望", results, question="2026年我国的出口情况")
        assert len(kept) == 1

    def test_drops_document_farms(self):
        results = [
            SearchResult("中国贸易伙伴", "https://www.doc88.com/p-123.html", "贸易伙伴国别分析 2026出口"),
        ]
        kept = _filter("中国出口贸易伙伴 国别分析", results, question="2026年我国的出口情况")
        assert kept == []


class TestFailedUrlTracking:
    """抓取失败/被拒收的 URL 跨轮拉黑：反爬 412 页面反复出现在补搜结果里，
    不拉黑会让每轮补搜都在同一批死链上空转（实测 13 次搜索零收录）。"""

    def test_failed_url_dropped_from_results(self):
        r = _bare(_question="2026年我国的出口情况",
                  _failed_urls={"https://customs.example.com/blocked"})
        results = [
            SearchResult("海关总署出口统计", "https://customs.example.com/blocked", "2026年出口总值统计"),
            SearchResult("出口总值再创新高", "https://news.example.com/ok", "2026年我国出口情况"),
        ]
        kept = r._filter_results(results, "2026年出口总值")
        assert [x.url for x in kept] == ["https://news.example.com/ok"]

    def test_no_failed_set_keeps_all(self):
        # __new__ 构造的裸实例没有 _failed_urls 属性时不报错、不误删
        results = [SearchResult("出口总值再创新高", "u1", "2026年我国出口情况")]
        assert [x.url for x in _filter("2026年出口总值", results, "2026年我国的出口情况")] == ["u1"]


class TestAnchorQuery:
    """大纲空词兜底改写：小节标题原样当搜索词会被题库站字面命中。"""

    Q = "调研2026年的毕业情况"

    def test_outline_word_gets_anchored(self):
        # 「政策影响因素」与问题核心词零重合 → 拼上核心再搜
        out = _anchor("政策影响因素", self.Q)
        assert out != "政策影响因素"
        assert "2026" in out and "政策影响因素" in out

    def test_specific_query_untouched(self):
        # 已含主题限定词（2026/毕业）→ 不改写
        assert _anchor("2026年毕业总人数", self.Q) == "2026年毕业总人数"

    def test_partial_overlap_untouched(self):
        # 与核心有足够重合（≥2 个二元组）→ 不改写
        out = _anchor("毕业生就业率统计", self.Q)
        assert out == "毕业生就业率统计"

    def test_core_substring_untouched(self):
        # 查询本身包含核心词 → 不重复拼接
        out = _anchor("2026年的毕业情况统计", self.Q)
        assert out == "2026年的毕业情况统计"

    def test_strip_question_verbs(self):
        assert _strip_question_verbs("调研一下2026年的毕业情况") == "2026年的毕业情况"
        assert _strip_question_verbs("帮我分析固态电池产业") == "固态电池产业"
        assert _strip_question_verbs("新能源汽车出口") == "新能源汽车出口"


class TestAuthorityAndBlocklist:
    def test_drops_quiz_farm(self):
        # 题库站字面完美命中查询词，也必须整站拒绝
        results = [
            SearchResult("影响政策执行的因素包括", "https://www.jutiku.cn/shiti/1.html",
                         "政策决定因素 政策资源因素 政策环境因素"),
        ]
        kept = _filter("政策影响因素", results, question="2026年毕业生政策环境")
        assert kept == []

    def test_authority_ranks_first_on_tie(self):
        # 相关性打平时，权威域名（gov.cn）排在普通站前面
        results = [
            SearchResult("毕业生就业数据发布", "https://news.example.com/a", "2026年毕业生就业数据发布"),
            SearchResult("毕业生就业数据分析", "https://www.moe.gov.cn/b", "2026年毕业生就业数据分析"),
        ]
        kept = _filter("毕业生就业数据", results, question="2026年的毕业情况")
        assert kept[0].url == "https://www.moe.gov.cn/b"

    def test_drops_school_internal_notice(self):
        # 学校公示页整条拒收：正文只有名单，没有任何统计口径，
        # 字面密度却很高，只降权会让它继续占着候选位（实测被收录 3 次）
        results = [
            SearchResult("2026届毕业生毕业资格及学位授予资格审查结果公示",
                         "https://jiaowuchu.xcu.edu.cn/info/1.html",
                         "2026届毕业生 名单公示 5281人"),
            SearchResult("2026届全国普通高校毕业生规模预计1270万人",
                         "https://news.example.com/guojia",
                         "教育部发布 2026届全国普通高校毕业生规模 1270万人"),
        ]
        kept = _filter("2026年毕业生人数规模", results, question="调研2026年的毕业情况")
        assert [x.url for x in kept] == ["https://news.example.com/guojia"]

    def test_gov_notice_kept(self):
        # 权威域的「通知」是部委文件，恰恰是最佳来源——不参与内部页拒收
        results = [
            SearchResult("关于做好2026届全国高校毕业生就业创业工作的通知",
                         "https://www.moe.gov.cn/srcsite/A17/notice.html",
                         "教育部 2026届高校毕业生 就业创业 通知"),
            SearchResult("某高校就业新闻", "https://news.example.com/x", "毕业生就业"),
        ]
        kept = _filter("毕业生就业通知", results, question="2026年的毕业情况")
        assert kept[0].url == "https://www.moe.gov.cn/srcsite/A17/notice.html"

    def test_drops_non_edu_internal_page(self):
        # 非教育话题同样成立：公司放假通知与调研无关，直接丢弃
        results = [
            SearchResult("关于2026年春节放假安排的通知", "https://hr.example.com/notice1",
                         "2026年春节 放假 通知 安排"),
            SearchResult("2026年春节假期全国消费数据解读", "https://news.example.com/c",
                         "2026年春节 全国消费 消费数据"),
        ]
        kept = _filter("2026年春节消费数据", results, question="2026年春节消费情况如何")
        assert [x.url for x in kept] == ["https://news.example.com/c"]

    def test_internal_page_drop_is_traced(self):
        # 拒收要留痕：否则线上只能看到"搜到了却没收录"，无法归因
        results = [
            SearchResult("关于2026年春节放假安排的通知", "https://hr.example.com/notice1",
                         "2026年春节 放假 通知 安排"),
        ]
        r = _bare(_question="2026年春节消费情况如何")
        assert r._filter_results(results, "2026年春节消费数据") == []
        assert any(e == "internal_page_dropped" for e, _ in r.trace.events)


class TestFreshnessPreference:
    """时效偏好：搜索结果此前不带日期，排序没有时间维度。

    实测候选里 2026 年 546 条、2025 年 102 条、2024 年及更早 50 条，
    旧稿与新闻同权，旧稿还常因是央媒排得更前。
    设计为**只奖新鲜、不罚旧**——见 test_old_and_unknown_treated_alike。
    """

    def test_fresher_wins_when_equally_relevant(self):
        results = [
            SearchResult("2026年毕业生就业数据分析", "https://news.example.com/old",
                         "2026年毕业生就业数据 分析", published_at="2024-01-01"),
            SearchResult("2026年毕业生就业数据分析", "https://news.example.com/new",
                         "2026年毕业生就业数据 分析", published_at="2026-08-01"),
        ]
        kept = _filter("2026年毕业生就业数据", results, question="2026年的毕业情况")
        assert kept[0].url == "https://news.example.com/new"

    def test_authority_not_displaced_by_freshness(self):
        # 离线回放发现的回归：若时效惩罚旧内容，新普通站会挤掉旧权威。
        # 权威分层必须优先于时效。
        results = [
            SearchResult("2026年毕业生就业数据发布", "https://www.moe.gov.cn/a",
                         "2026年毕业生就业数据发布", published_at="2025-11-20"),
            SearchResult("2026年毕业生就业数据解读", "https://news.example.com/b",
                         "2026年毕业生就业数据解读", published_at="2026-09-01"),
        ]
        kept = _filter("2026年毕业生就业数据", results, question="2026年的毕业情况")
        assert kept[0].url == "https://www.moe.gov.cn/a"

    def test_old_and_unknown_treated_alike(self, monkeypatch):
        # 旧内容与日期未知都不获奖，因此时效开关不应改变两者的相对顺序。
        # 这是"只奖不罚"的关键性质：否则未知日期的政府来源会反超旧权威。
        from agent import config

        results = [
            SearchResult("2026年毕业生就业数据分析", "https://news.example.com/old",
                         "2026年毕业生就业数据 分析", published_at="2024-01-01"),
            SearchResult("2026年毕业生就业数据分析", "https://news.example.com/nodate",
                         "2026年毕业生就业数据 分析"),
        ]
        on = [x.url for x in _filter("2026年毕业生就业数据", results,
                                     question="2026年的毕业情况")]
        monkeypatch.setattr(config, "FRESH_DAYS", 0)
        off = [x.url for x in _filter("2026年毕业生就业数据", results,
                                      question="2026年的毕业情况")]
        assert on == off

    def test_disabled_still_keeps_all(self, monkeypatch):
        from agent import config
        monkeypatch.setattr(config, "FRESH_DAYS", 0)
        results = [
            SearchResult("2026年毕业生就业数据分析", "https://news.example.com/old",
                         "2026年毕业生就业数据 分析", published_at="2025-01-01"),
            SearchResult("2026年毕业生就业数据分析", "https://news.example.com/new",
                         "2026年毕业生就业数据 分析", published_at="2026-08-01"),
        ]
        kept = _filter("2026年毕业生就业数据", results, question="2026年的毕业情况")
        assert len(kept) == 2


class TestDateTag:
    def test_tag_rendered(self):
        assert _date_tag("2026-06-30") == "（2026-06-30）"

    def test_unknown_date_not_fabricated(self):
        assert _date_tag("") == ""

    def test_search_result_defaults_to_no_date(self):
        assert SearchResult("标题", "https://x.com/a", "摘要").published_at == ""
