"""日期归一化与提取测试。

锁定改造行为：日期在搜索结果层曾完全缺失、网页 meta 日期会出错且无人校验。
- 各来源写法归一为 YYYY-MM-DD；
- URL 内嵌日期可提取，且不会把长数字 ID 误判为日期；
- meta 与 URL 矛盾时以 URL 为准（实测光明日报电子版案例）；
- `is_recent` 只用于正向加分，日期未知一律返回 False。
"""
from datetime import date

from agent.dates import extract_url_date, is_recent, normalize, resolve_published

TODAY = date(2026, 9, 13)


class TestNormalize:
    def test_iso_with_time_and_timezone(self):
        assert normalize("2026-04-15T10:00:00+08:00") == "2026-04-15"

    def test_plain_iso(self):
        assert normalize("2026-03-09") == "2026-03-09"

    def test_rfc2822_from_tavily(self):
        # Tavily topic=news 的 published_date 形态
        assert normalize("Fri, 03 Apr 2026 00:00:00 GMT") == "2026-04-03"

    def test_chinese_written(self):
        assert normalize("2026年3月9日") == "2026-03-09"

    def test_slash_form(self):
        assert normalize("2026/03/09") == "2026-03-09"

    def test_compact_form(self):
        assert normalize("20260309") == "2026-03-09"

    def test_empty_and_garbage(self):
        assert normalize("") == ""
        assert normalize("未标明") == ""
        assert normalize("今天") == ""

    def test_invalid_month_rejected(self):
        # 宁可缺失也不给错值
        assert normalize("2026-13-09") == ""


class TestExtractUrlDate:
    def test_slash_year_month_day(self):
        assert extract_url_date("https://epaper.gmw.cn/gmrb/html/content/202606/30/x.html") == "2026-06-30"

    def test_dashed(self):
        assert extract_url_date("http://auto.cyol.com/gb/articles/2026-08/19/content_x.html") == "2026-08-19"

    def test_compact_path(self):
        # 新华网 /20251120/xxxx/c.html
        assert extract_url_date("https://www.news.cn/20251120/ead0f25d/c.html") == "2025-11-20"

    def test_year_month_only_defaults_to_first(self):
        assert extract_url_date("https://www.moe.gov.cn/jyb_xwfb/202603/t1.html") == "2026-03-01"

    def test_embedded_id_style(self):
        assert extract_url_date("https://www.zaobao.com.sg/news/china/story20251121-7854118") == "2025-11-21"

    def test_long_numeric_id_not_a_date(self):
        # 推特/雪球等平台的雪花 ID 不能当日期
        assert extract_url_date("https://x.com/AsiaStock/status/2058163934876537065") == ""
        assert extract_url_date("https://www.163.com/dy/article/KDS1MECQ0511B3FV.html") == ""

    def test_no_date_returns_empty(self):
        assert extract_url_date("http://www.customs.gov.cn/customs/xwfb34/302330/index.html") == ""

    def test_far_future_rejected(self):
        assert extract_url_date("https://example.com/2099/01/01/x.html") == ""


class TestResolvePublished:
    def test_url_wins_when_year_conflicts(self):
        # 实测案例：光明日报电子版 URL 是 2026-06-30，trafilatura 从页脚
        # 版权解析出 2025-01-01，原样写进报告"发布于 2025-01-01"
        out = resolve_published(
            "2025-01-01",
            "https://epaper.gmw.cn/gmrb/html/content/202606/30/content_17988.html")
        assert out == "2026-06-30"

    def test_meta_kept_when_year_agrees(self):
        out = resolve_published(
            "2026-04-15T10:00:00+08:00", "https://www.news.cn/20260415/x/c.html")
        assert out == "2026-04-15"

    def test_meta_only(self):
        assert resolve_published("2026-02-11", "https://m.chinabgao.com/a/b.html") == "2026-02-11"

    def test_url_only(self):
        assert resolve_published("", "https://www.news.cn/20251120/x/c.html") == "2025-11-20"

    def test_neither(self):
        assert resolve_published("", "https://example.com/a/b.html") == ""


class TestIsRecent:
    def test_recent(self):
        assert is_recent("2026-09-01", 180, TODAY)

    def test_old_is_not_recent(self):
        assert not is_recent("2025-11-20", 180, TODAY)

    def test_unknown_is_not_recent(self):
        # 只奖不罚：未知一律不获奖，但也不会被当成旧内容惩罚（由调用方保证）
        assert not is_recent("", 180, TODAY)

    def test_boundary_inside_window(self):
        assert is_recent("2026-03-18", 180, TODAY)

    def test_boundary_outside_window(self):
        assert not is_recent("2026-03-16", 180, TODAY)
