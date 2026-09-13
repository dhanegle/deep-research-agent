"""来源质量治理：权威域名分层、垃圾站黑名单、低质与单位内部页识别。

集中管理"什么样的来源值得收录"，供 researcher（检索/过滤/补搜）与评估共用。

改造动机（实测数据，51 份报告 / 504 份搜索快照）：
- 收录来源中权威域名仅占 4.6%，央媒仅占 6.9%；47/51 份报告无任何权威来源；
- 原始搜索结果里央媒只占 2.8%，仅 12.9% 的查询能搜到任何央媒——
  说明"等着排序把央视顶上来"不够，必须主动做权威定向检索；
- 高频污染源是行业报告聚合站（chinabgao 等），标题堆砌查询词但无一手正文。

因此把原先的"单一 +0.15 加权"改为**分层**：中央媒体 > 政府/统计机构 > 普通站点，
并新增定向检索所需的权威域名白名单。
"""
import re

# 第一层：中央媒体与权威新闻机构（用户期望的"央视网"一类来源）
CENTRAL_MEDIA = (
    "cctv.com", "cctv.cn", "news.cn", "xinhuanet.com", "people.com.cn",
    "chinadaily.com.cn", "gmw.cn", "cnr.cn", "china.com.cn", "thepaper.cn",
    "ce.cn", "youth.cn", "chinanews.com.cn", "cri.cn", "stdaily.com",
    "huanqiu.com", "bjnews.com.cn", "yicai.com", "nbd.com.cn", "21jingji.com",
)

# 第二层：政府与统计机构。注意博查 include 不支持泛域 "gov.cn"（实测返回 0 条），
# 必须逐个列出具体站点，因此这里显式枚举中央部委与统计机构。
GOVERNMENT = (
    "www.gov.cn", "moe.gov.cn", "stats.gov.cn", "customs.gov.cn",
    "miit.gov.cn", "ndrc.gov.cn", "mofcom.gov.cn", "mof.gov.cn",
    "pbc.gov.cn", "samr.gov.cn", "mot.gov.cn", "nea.gov.cn",
    "nhc.gov.cn", "mohurd.gov.cn", "cac.gov.cn", "ncha.gov.cn",
    "sasac.gov.cn", "safe.gov.cn", "csrc.gov.cn", "gov.cn",
)

# 权威行业机构/交易所：统计数据的一手发布方（如中汽协、中证网）
INDUSTRY = (
    "cada.cn", "caam.org.cn", "cninfo.com.cn", "cs.com.cn",
    "sse.com.cn", "szse.cn", "chinaclear.cn", "cnautonews.com",
)

# 定向检索白名单：include_domains / 博查 include 用。
# 必须排除裸域 "gov.cn"——博查实测不支持泛域（返回 0 条），
# 但裸域保留在 GOVERNMENT 里用于层级匹配（任何 *.gov.cn 都是 tier 1）。
AUTHORITY_DOMAINS = CENTRAL_MEDIA + tuple(d for d in GOVERNMENT if d != "gov.cn") + INDUSTRY

# 纯 SEO 内容聚合/文库/题库站：无一手正文，收录后摘要为空壳或二手转述。
# 前两项是历史规则；后半段是本次按实测高频污染源补充的行业报告聚合站
# （chinabgao 报告大厅在 51 份报告里被收录 24 次，是最大的单一污染源）。
BLOCKED_HOSTS = (
    # 文库/文档分享站
    "docin.com", "doc88.com", "taodocs.com", "book118.com",
    "360doc.com", "wenku.baidu.com", "max.book118.com",
    # 题库/答疑/范文站：大纲式空词的完美字面命中大户
    "jutiku.cn", "xilvlaw.com", "wkda.cn", "027art.com",
    "renrendoc.com", "cooco.net.cn", "eepw.com/shiti",
    # 行业报告聚合站：标题堆砌查询词，正文是付费报告的营销摘要
    "chinabgao.com", "chinabaogao.com", "fxbaogao.com", "baogaox.com",
    "51w2c.com", "hlsok.com", "vzkoo.com", "chyxx.com", "qianzhan.com",
    "askci.com", "zhengceku.com", "hangyebaogao.com", "bg.qianzhan.com",
)

# UGC/自媒体平台：内容质量方差大，小幅降权而非整站拒绝。
# 不含 news.qq.com / finance.sina.com.cn 这类门户新闻频道——它们有真实采编内容。
LOW_QUALITY_HOSTS = (
    "csdn.net", "cnblogs.com", "zhihu.com", "jianshu.com",
    "toutiao.com", "baijiahao.baidu.com", "bilibili.com",
    "xiaohongshu.com", "douban.com", "weibo.com", "mp.weixin.qq.com",
)

# 单位内部页面标题特征（公示/资格审查/放假通知……任何话题都成立）：
# 这类页面字面密度高、更新勤，总量统计类查询下常压过权威报道。
# 实测遗漏："关于开展2026届毕业生就业意向和进展调查的通知"——
# 旧规则只匹配"关于做好…的通知"，"关于开展…"整类漏网。
# 命中后由 is_internal_source 决定是否整条丢弃（非权威域直接拒收，见该函数）。
INTERNAL_PAGE_RE = re.compile(
    r"公示|教务处|资格审查|录取名单|成绩查询|放假安排|校历|返校"
    r"|关于[^，。！？；]{0,24}的通知"
    r"|招聘公告|招标公告|采购公告|中标公告|询价公告|竞争性磋商"
    r"|招生简章|任前公示|拟聘用人员|拟录用人员|考察对象公示"
)


def _matches(host: str, domains) -> bool:
    """域名匹配：支持裸域、带点前缀（.gov.cn）与带路径规则（eepw.com/shiti）。"""
    for d in domains:
        if "/" in d:
            continue
        bare = d.lstrip(".")
        if host == bare or host.endswith("." + bare):
            return True
    return False


def authority_tier(host: str) -> int:
    """权威层级：2=中央媒体/权威新闻机构，1=政府与统计机构/权威行业机构，0=其他。"""
    host = (host or "").lower()
    if _matches(host, CENTRAL_MEDIA):
        return 2
    if _matches(host, GOVERNMENT) or _matches(host, INDUSTRY):
        return 1
    return 0


def is_authority(host: str) -> bool:
    return authority_tier(host) > 0


def is_blocked(host: str, url: str = "") -> bool:
    """整站拒绝：域名命中黑名单，或 URL 命中带路径规则（如 eepw.com/shiti）。"""
    host = (host or "").lower()
    if _matches(host, BLOCKED_HOSTS):
        return True
    low = (url or "").lower()
    return any("/" in d and d in low for d in BLOCKED_HOSTS)


def is_low_quality(host: str) -> bool:
    return _matches((host or "").lower(), LOW_QUALITY_HOSTS)


def is_internal_page(title: str) -> bool:
    return bool(INTERNAL_PAGE_RE.search(title or ""))


def is_internal_source(host: str, title: str) -> bool:
    """单位内部页是否应整条丢弃（不只是降权）。

    原先只做 -0.15 降权，实测拦不住：北京体育大学就业指导中心的
    《关于开展2026届毕业生就业意向和进展调查（3月）的通知》在 51 份报告里
    被收录 3 次，正文全是调查安排、没有一个统计数据，却因为字面密度高挤进了
    候选前 5。这类页面（公示/资格审查/放假通知/采购公告）任何话题下都不含
    一手数据，属于"收录了也没用"，直接拒收比降权更省一轮抓取与摘要。

    两条豁免：
    - **权威域**：部委/统计机构的「通知」恰恰是统计类问题的最佳一手来源
      （实测 227 条收录来源里，权威域内部页仅 1 条且是有价值的部委文件）；
    - **无域名的本地语料**（url 是相对路径）：那是用户自己放进 docs 目录的文档，
      不该由来源治理二次判断。
    """
    if not host:
        return False
    return authority_tier(host) == 0 and is_internal_page(title)


def authority_domains() -> tuple[str, ...]:
    """定向检索白名单（Tavily include_domains / 博查 include 共用）。"""
    return AUTHORITY_DOMAINS
