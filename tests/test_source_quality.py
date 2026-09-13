"""来源质量治理测试：权威分层、垃圾站黑名单、单位内部页识别与拒收、权威定向补搜。

锁定本次改造的行为：
- 权威分层（中央媒体 2 / 政府统计机构 1 / 普通 0）；
- 行业报告聚合站整站拒绝（实测最大污染源，51 份报告被收录 24 次）；
- 单位内部页正则覆盖「关于开展…的通知」（旧规则只认「关于做好…」）；
- 非权威域的单位内部页整条拒收（原为 -0.15 降权，实测拦不住）；
- 常规搜索无权威来源时触发权威定向检索，且受预算约束。
"""
import threading

import pytest

from agent import config
from agent.researcher import Researcher
from agent.search.base import SearchResult
from agent.source_quality import (
    authority_domains, authority_tier, is_blocked, is_internal_page,
    is_internal_source, is_low_quality,
)
from agent.stats import RunStats


class TestAuthorityTier:
    def test_central_media_is_top_tier(self):
        assert authority_tier("news.cctv.com") == 2
        assert authority_tier("www.news.cn") == 2
        assert authority_tier("www.people.com.cn") == 2
        assert authority_tier("m.gmw.cn") == 2

    def test_government_is_middle_tier(self):
        assert authority_tier("www.moe.gov.cn") == 1
        assert authority_tier("www.customs.gov.cn") == 1
        assert authority_tier("www.gov.cn") == 1

    def test_ordinary_site_is_zero(self):
        assert authority_tier("news.qq.com") == 0
        assert authority_tier("m.chinabgao.com") == 0
        assert authority_tier("blog.csdn.net") == 0

    def test_industry_body_is_authority(self):
        # 中汽协/中证网等一手数据发布方
        assert authority_tier("www.cada.cn") == 1


class TestBlocklist:
    def test_report_farms_blocked(self):
        # 实测最高频污染源
        assert is_blocked("m.chinabgao.com")
        assert is_blocked("www.chinabaogao.com")
        assert is_blocked("www.fxbaogao.com")
        assert is_blocked("51w2c.com")
        assert is_blocked("www.hlsok.com")

    def test_document_farms_blocked(self):
        assert is_blocked("www.doc88.com")
        assert is_blocked("wenku.baidu.com")

    def test_authority_not_blocked(self):
        assert not is_blocked("news.cctv.com")
        assert not is_blocked("www.moe.gov.cn")

    def test_path_rule_matches_url(self):
        assert is_blocked("eepw.com", "https://www.eepw.com/shiti/123.html")
        assert not is_blocked("eepw.com", "https://www.eepw.com/news/1.html")


class TestInternalPage:
    def test_kaizhan_notice_matched(self):
        # 旧规则只认「关于做好…的通知」，此类整类漏网（实测被收录）
        assert is_internal_page("关于开展2026届毕业生就业意向和进展调查（3月）的通知")

    def test_zuohao_notice_matched(self):
        assert is_internal_page("关于做好2026届毕业生就业创业工作的通知")

    def test_publicity_and_audit_matched(self):
        assert is_internal_page("2026届毕业生毕业资格审查结果公示")
        assert is_internal_page("拟聘用人员名单公示")

    def test_normal_report_not_matched(self):
        assert not is_internal_page("2026届全国普通高校毕业生规模预计1270万人")
        assert not is_internal_page("新能源汽车出口再创新高")

    def test_self_media_low_quality(self):
        assert is_low_quality("blog.csdn.net")
        assert is_low_quality("zhuanlan.zhihu.com")
        # 门户新闻频道有真实采编内容，不降权
        assert not is_low_quality("news.qq.com")
        assert not is_low_quality("finance.sina.com.cn")


class TestInternalSourceDropped:
    """非权威域的单位内部页整条拒收（此前只 -0.15 降权，实测拦不住）。"""

    def test_university_notice_dropped(self):
        # 实测案例：北京体育大学就业指导中心的调查通知被收录 3 次，
        # 正文只有调查安排，没有任何统计数据
        assert is_internal_source(
            "jy.bsu.edu.cn", "关于开展2026届毕业生就业意向和进展调查（3月）的通知")
        assert is_internal_source("jiaowuchu.xcu.edu.cn", "2026届毕业生毕业资格审查结果公示")

    def test_non_edu_internal_page_dropped(self):
        assert is_internal_source("hr.example.com", "关于2026年春节放假安排的通知")

    def test_authority_notice_exempt(self):
        # 部委「通知」是统计类问题的最佳一手来源，必须放行
        assert not is_internal_source(
            "www.moe.gov.cn", "关于做好2026届全国高校毕业生就业创业工作的通知")
        assert not is_internal_source(
            "www.stats.gov.cn", "关于开展2026年全国人口变动情况抽样调查的通知")

    def test_normal_page_not_dropped(self):
        assert not is_internal_source(
            "news.example.com", "2026届全国普通高校毕业生规模预计1270万人")

    def test_local_corpus_without_host_exempt(self):
        # 本地语料 url 是相对路径（无 hostname），不该由来源治理二次判断
        assert not is_internal_source("", "关于做好2026年工作的通知")


class TestAuthorityDomains:
    def test_non_empty_and_no_bare_gov(self):
        domains = authority_domains()
        assert len(domains) > 20
        # 博查 include 不支持泛域 gov.cn（实测返回 0 条），白名单不能含裸 gov.cn
        assert "gov.cn" not in domains
        assert "moe.gov.cn" in domains
        assert "cctv.com" in domains


class _Trace:
    def __init__(self):
        self.events = []

    def log(self, event, **payload):
        self.events.append((event, payload))


class _Provider:
    """普通检索返回 plain，带 include_domains 时返回 auth。"""

    def __init__(self, plain, auth):
        self.plain, self.auth, self.calls = plain, auth, []

    def search(self, query, max_results=5, include_domains=None):
        self.calls.append((query, max_results, include_domains))
        return self.auth if include_domains else self.plain


def _researcher(provider):
    r = Researcher.__new__(Researcher)
    r.provider = provider
    r.stats = RunStats()
    r.trace = _Trace()
    r.emit = lambda event, data: None
    r._lock = threading.Lock()
    r._authority_passes = 0
    r._question = "2026年的毕业情况"
    r._failed_urls = set()
    return r


CCTV = SearchResult("2026届全国普通高校毕业生规模预计1270万人",
                    "https://news.cctv.com/2026/03/19/x.html",
                    "教育部发布 2026届高校毕业生规模 1270万人")
PLAIN = SearchResult("毕业生就业数据解读", "https://news.example.com/a",
                     "2026年毕业生就业数据 解读")


class TestAuthorityPass:
    def test_triggered_and_promoted(self):
        provider = _Provider([PLAIN], [CCTV])
        r = _researcher(provider)
        merged = r._authority_pass("2026年毕业生就业数据", [PLAIN], {})
        # 权威结果被前置
        assert merged[0].url == CCTV.url
        # 普通结果保留在后
        assert PLAIN.url in [x.url for x in merged]
        # include_domains 确实传给了提供方
        assert provider.calls and provider.calls[0][2]

    def test_skipped_when_disabled(self, monkeypatch):
        monkeypatch.setattr(config, "AUTHORITY_PASS", False)
        provider = _Provider([PLAIN], [CCTV])
        r = _researcher(provider)
        merged = r._authority_pass("2026年毕业生就业数据", [PLAIN], {})
        assert merged == [PLAIN]
        assert provider.calls == []

    def test_respects_budget(self, monkeypatch):
        monkeypatch.setattr(config, "AUTHORITY_PASS_BUDGET", 1)
        provider = _Provider([PLAIN], [CCTV])
        r = _researcher(provider)
        r._authority_pass("q1", [PLAIN], {})
        r._authority_pass("q2", [PLAIN], {})
        assert len(provider.calls) == 1  # 第二次被预算拦住

    def test_stats_counted(self):
        provider = _Provider([PLAIN], [CCTV])
        r = _researcher(provider)
        r._authority_pass("q", [PLAIN], {})
        assert r.stats.authority_searches == 1
        assert r.stats.searches == 1

    def test_provider_without_support_degrades(self):
        class OldProvider:
            def search(self, query, max_results=5):
                return [PLAIN]

        r = _researcher(OldProvider())
        merged = r._authority_pass("q", [PLAIN], {})  # 不应抛异常
        assert merged == [PLAIN]


class TestTieredRanking:
    def test_authority_promoted_over_relevant_small_site(self):
        r = _researcher(_Provider([], []))
        results = [
            SearchResult("2026届毕业生就业数据发布", "https://www.hlsok.com/a",
                         "2026年毕业生就业数据发布 就业率"),
            SearchResult("2026届高校毕业生就业形势", "https://news.cctv.com/b",
                         "2026届高校毕业生就业形势分析"),
        ]
        kept = r._filter_results(results, "2026年毕业生就业数据")
        assert kept[0].url == "https://news.cctv.com/b"

    def test_report_farm_dropped(self):
        r = _researcher(_Provider([], []))
        results = [
            SearchResult("2026年毕业生就业数据报告", "https://m.chinabgao.com/a",
                         "2026年毕业生就业数据 分析报告"),
        ]
        assert r._filter_results(results, "2026年毕业生就业数据") == []

    def test_irrelevant_authority_not_promoted(self):
        # 权威项相关性明显落后时不顶置（RELEVANCE_FLOOR 约束）
        r = _researcher(_Provider([], []))
        results = [
            SearchResult("2026年毕业生就业数据发布", "https://news.example.com/a",
                         "2026年毕业生就业数据发布 就业率 就业市场 分析"),
            SearchResult("某市政府采购意向公告", "https://www.gov.cn/x",
                         "政府采购 意向公告"),
        ]
        kept = r._filter_results(results, "2026年毕业生就业数据")
        # 明显跑题的政府页要么被相关性门槛挡掉，要么不排在首位
        assert not kept or kept[0].url == "https://news.example.com/a"

    def test_school_internal_page_dropped_but_authority_notice_kept(self):
        # 同一条查询下：学校公示页拒收，部委通知照常保留
        r = _researcher(_Provider([], []))
        results = [
            SearchResult("关于开展2026届毕业生就业意向和进展调查的通知",
                         "https://jy.bsu.edu.cn/info/1.html",
                         "2026届毕业生 就业意向 调查 通知"),
            SearchResult("关于做好2026届全国高校毕业生就业创业工作的通知",
                         "https://www.moe.gov.cn/srcsite/A17/notice.html",
                         "2026届高校毕业生 就业创业 通知"),
        ]
        kept = r._filter_results(results, "2026届毕业生就业通知")
        assert [x.url for x in kept] == ["https://www.moe.gov.cn/srcsite/A17/notice.html"]
        assert any(e == "internal_page_dropped" for e, _ in r.trace.events)
