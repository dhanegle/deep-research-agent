"""日期归一化与提取：搜索结果层与网页抓取层共用的时间解析。

改造动机（实测数据）：
- **搜索结果层此前完全不带日期**。`SearchResult` 只有 title/url/snippet 三个字段，
  于是排序 key（权威层级/相关度/覆盖率）与给模型的候选列表都没有时间维度：
  2025-11 的旧稿与 2026-06 的新稿同权，而旧稿因是央媒（tier 2）反而排得更前，
  模型读到旧数字就写进报告。
- **网页 meta 解析出的日期会出错且无人校验**。光明日报电子版的 URL 是
  `/gmrb/html/content/202606/30/...`，trafilatura 却解析出 `2025-01-01`
  （页脚版权年份），并被原样写进报告"发布于 2025-01-01"。错误日期比缺失更有害。

各来源的实测覆盖率（3358 条候选 / 233 条已收录来源）：

| 来源 | 覆盖率 | 说明 |
|---|---|---|
| 博查 `datePublished` | 100% | ISO 8601，此前被 bocha.py 丢弃 |
| Tavily `published_date` | 仅 `topic=news` 有，`general` 为 0 | 实测两 topic 召回重叠仅 12.5%，不能整体切换 |
| URL 内嵌日期 | 全量 21%，**权威域≈100%** | news.cn 26/26、moe.gov.cn 16/16、xinhuanet 14/14 |
| 网页 meta（trafilatura） | 42.9%，含错值 | 需与 URL 交叉校验 |

因此把"从哪拿日期、怎么归一、怎么判可信"集中到这里，供搜索层与抓取层共用。
统一输出 `YYYY-MM-DD`；无法确定时返回空串（**不猜**——宁可"未标明"，不给错值）。
"""
import re
from datetime import date, timedelta

# 通用归一化：API 字段与 meta 标签的各种写法
_ISO_RE = re.compile(r"(20\d{2})-(\d{1,2})-(\d{1,2})")
_SLASH_RE = re.compile(r"(20\d{2})/(\d{1,2})/(\d{1,2})")
_CN_RE = re.compile(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")
_RFC_RE = re.compile(r"(\d{1,2})\s+([A-Za-z]{3})\s+(20\d{2})")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"))}

# URL 专用：按特异性从高到低尝试，命中即返回（(模式, 捕获组数)）
_URL_PATTERNS = (
    (re.compile(r"/(20\d{2})/(\d{2})/(\d{2})/"), 3),    # /2026/08/04/
    (re.compile(r"/(20\d{2})-(\d{2})-(\d{2})"), 3),     # /2026-08-19
    (re.compile(r"/(20\d{2})(\d{2})/(\d{2})/"), 3),     # /202606/30/  光明日报电子版
    (re.compile(r"/(20\d{2})-(\d{2})/(\d{2})"), 3),     # /2026-08/19  中青报
    (re.compile(r"/(20\d{2})/(\d{2})(\d{2})/"), 3),     # /2026/0304/
    (re.compile(r"/(20\d{2})(\d{2})(\d{2})/"), 3),      # /20260324/
    (re.compile(r"[^0-9](20\d{2})(\d{2})(\d{2})[^0-9]"), 3),  # story20251121-
    (re.compile(r"/(20\d{2})[-_/](\d{2})/"), 2),        # /2026-08/  /2026_08/
    (re.compile(r"/(20\d{2})(\d{2})/"), 2),             # /202603/
)

# 合理性边界：URL 里的数字串可能碰巧长得像日期
_MIN_YEAR = 2000


def _make(year: str, month: str, day: str) -> str:
    """构造 YYYY-MM-DD；非法月/日返回空串（宁可缺失也不给错值）。"""
    y, m, d = int(year), int(month), int(day)
    if not (_MIN_YEAR <= y <= date.today().year + 1):
        return ""
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return ""
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return ""


def normalize(raw: str) -> str:
    """把各种日期写法归一为 YYYY-MM-DD，无法识别返回空串。

    覆盖：ISO 8601（含 T 时间与时区）、2026/03/09、2026年3月9日、
    20260309、RFC 2822（Tavily 的 `Fri, 03 Apr 2026 00:00:00 GMT`）。
    """
    text = (raw or "").strip()
    if not text:
        return ""
    for pat in (_ISO_RE, _SLASH_RE, _CN_RE, _COMPACT_RE):
        m = pat.search(text)
        if m:
            out = _make(*m.groups())
            if out:
                return out
    m = _RFC_RE.search(text)
    if m:
        month = _MONTHS.get(m.group(2).lower())
        if month:
            return _make(m.group(3), str(month), m.group(1))
    return ""


def extract_url_date(url: str) -> str:
    """从 URL 路径提取发布日期；取不到返回空串。

    日期型 URL 是发布系统的固定路径规则（`/2026/0403/`、`/20251120/`），
    比正文里的"© 2025"可靠。实测权威域覆盖接近 100%：
    news.cn 26/26、moe.gov.cn 16/16、xinhuanet 14/14、edu.cctv.com 12/12。
    仅命中两位（年+月）时按当月 1 日处理。
    """
    text = url or ""
    for pat, parts in _URL_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        g = m.groups()
        day = g[2] if parts == 3 else "01"
        out = _make(g[0], g[1], day)
        if out:
            return out
    return ""


def resolve_published(meta_date: str, url: str) -> str:
    """网页发布日期：meta 与 URL 交叉校验，矛盾时以 URL 为准。

    两者都是发布方产物，但出错方式不同：meta 会被页脚版权年份污染
    （实测光明日报电子版解析出 2025-01-01，实际为 2026-06-30），
    URL 路径则是发布系统按日期生成的，不易被正文改动影响。
    因此年份矛盾时采信 URL；只有一方时用那一方；都没有则留空。
    """
    meta = normalize(meta_date)
    from_url = extract_url_date(url)
    if meta and from_url:
        return from_url if meta[:4] != from_url[:4] else meta
    return from_url or meta


def is_recent(published: str, days: int, today: date | None = None) -> bool:
    """发布日期是否在最近 `days` 天内（无日期视为否——不奖励未知）。

    只用于**正向**加分：调用方不应拿它惩罚旧内容。实测多数政府一手来源的
    URL 不含日期，一旦"未知"相对占优，会把确知较旧的央视/新华网挤出候选。
    """
    if not published:
        return False
    try:
        d = date.fromisoformat(published)
    except ValueError:
        return False
    ref = today or date.today()
    return ref - timedelta(days=days) <= d <= ref + timedelta(days=1)
