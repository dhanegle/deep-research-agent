"""知识库：收录已读网页的摘要，并按报告小节做相关性选材。

这是上下文压缩的核心：正文被压成短摘要存库，写作时每个小节
只注入按相关性挑选的少量摘要，保证 3B 模型的上下文始终很小。
相关性用字符二元组重合度计算，中文场景下无需引入分词依赖。
"""
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from .evidence import aspect_terms, content_hash, content_terms, evidence_blocks, select_evidence
from .source_quality import authority_tier


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _windows4(s: str) -> list[str]:
    """小节标题的 4 字窗口：比二元组更能区分「货物贸易」和「贸易伙伴」。"""
    s = re.sub(r"\s+", "", s)
    if len(s) < 4:
        return [s] if s else []
    return [s[i:i + 4] for i in range(len(s) - 3)]


def query_tokens(query: str) -> list[tuple[str, float]]:
    """把搜索词切成带权重的匹配元。

    ASCII 整词（如 "linuxsb"、"2025"）通常是专有名词/年份，权重更高；
    中文按空格分段，≥2 字的段各为一个 token。
    """
    tokens: list[tuple[str, float]] = []
    for part in query.split():
        for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{1,}", part):
            tokens.append((w.lower(), 1.5))
        cjk = re.sub(r"[A-Za-z0-9_.\-\s]+", " ", part)
        tokens.extend((seg, 1.0) for seg in cjk.split() if len(seg) >= 2)
    return tokens


def _is_ascii_proper(tok: str) -> bool:
    """真正的专有名词：ASCII 且不是纯数字（年份不能一票否决整页）。"""
    return tok.isascii() and not tok.isdigit()


def ascii_proper_tokens(tokens: list[tuple[str, float]]) -> list[str]:
    return [tok for tok, _ in tokens if _is_ascii_proper(tok)]


def ascii_token_hit(text: str, tokens: list[tuple[str, float]]) -> bool:
    """查询含 ASCII 专有名词（如 "linuxsb"）时的强判别：

    结果必须至少命中其中一个，否则视为无关。防止"网站/介绍"这类泛词
    让 GitHub 上的"个人介绍网站.html"之类的页面混过覆盖率门槛。
    纯中文查询、或 ASCII 只是年份（2025）时恒为 True——年份用于排序加权，
    不该把没写年份的相关页整页丢掉。
    """
    proper = ascii_proper_tokens(tokens)
    if not proper:
        return True
    low = text.lower()
    compact = re.sub(r"[._\-]", "", low)
    for tok in proper:
        tok_c = re.sub(r"[._\-]", "", tok)
        if tok in low or tok in compact or tok_c in compact:
            return True
    return False


def token_coverage(text: str, tokens: list[tuple[str, float]]) -> float:
    """文本对查询 token 的加权覆盖率，0~1。

    ASCII token（专有名词/年份）必须精确匹配——部分命中会让
    "linuxsb" 被 "linux" 泛页面误命中；长中文段才允许 4 元组部分命中。
    """
    if not tokens:
        return 1.0
    low = text.lower()
    compact = re.sub(r"[._\-]", "", low)  # 域名分隔符归一：linuxsb 可匹配 linux.sb
    total = sum(w for _, w in tokens)
    got = 0.0
    for tok, w in tokens:
        tok_c = re.sub(r"[._\-]", "", tok) if tok.isascii() else tok
        if tok in low or tok in compact or tok_c in compact:
            got += w
        elif len(tok) >= 4 and not tok.isascii():
            grams = [tok[i:i + 4] for i in range(0, len(tok) - 3, 2)]
            if any(g in low for g in grams):
                got += w * 0.5
    return got / total


@dataclass
class Source:
    id: int
    url: str
    title: str
    digest: str
    text: str = ""
    published_at: str = ""
    publisher: str = ""
    retrieved_at: str = ""
    content_hash: str = ""

    @property
    def evidence(self) -> str:
        return self.text or self.digest

    def excerpt(self, query: str, max_chars: int = 2400,
                boost: set[str] | None = None) -> str:
        return select_evidence(self.evidence, query, max_chars, boost)


class KnowledgeBase:
    def __init__(self):
        self.sources: list[Source] = []
        self._by_url: dict[str, int] = {}
        self._by_content: dict[str, int] = {}

    def __len__(self):
        return len(self.sources)

    def add(self, url: str, title: str, digest: str, *, text: str = "",
            published_at: str = "", publisher: str = "", retrieved_at: str = "") -> int:
        """按 URL 去重；重复收录返回已有编号。"""
        if url in self._by_url:
            return self._by_url[url]
        fingerprint = content_hash(text) if text else ""
        if fingerprint and fingerprint in self._by_content:
            sid = self._by_content[fingerprint]
            self._by_url[url] = sid
            return sid
        sid = len(self.sources) + 1
        self.sources.append(Source(sid, url, title, digest, text, published_at,
                                   publisher, retrieved_at, fingerprint))
        self._by_url[url] = sid
        if fingerprint:
            self._by_content[fingerprint] = sid
        return sid

    def _hits(self, source: Source, terms: set[str]) -> int:
        text = f"{source.title} {' '.join(evidence_blocks(source.evidence))}".lower()
        return sum(1 for t in terms if t in text)

    def relevance(self, source: Source, section: str, question: str) -> int:
        return self._hits(source, _bigrams(section) | _bigrams(question))

    def section_match(self, source: Source, section: str, question: str) -> tuple[int, int]:
        """返回 (标题 4 字窗口命中数, 小节区别于问题的二元组命中数)。

        4 字窗口用来把「货物贸易」和「贸易伙伴」分开；二元组作次级信号。
        """
        text = f"{source.title} {' '.join(evidence_blocks(source.evidence))}"
        w4 = sum(1 for w in _windows4(section) if w in text)
        distinctive = _bigrams(section) - _bigrams(question)
        d2 = sum(1 for t in distinctive if t in text) if distinctive else self._hits(source, _bigrams(section))
        return w4, d2

    def _title_hits(self, source: Source, terms: set[str]) -> int:
        return sum(1 for t in terms if t in source.title.lower())

    def select_for(self, section: str, question: str, k: int = 5) -> list[Source]:
        """只选择有小节证据的来源；问题泛词或年份不能替代小节覆盖。

        标题命中排在正文命中之前：正文里的 aspect 命中会被泛词稀释——
        长文只要出现过一次"趋势"就得分，与专门讲展望的报道同分。标题不会，
        写明"…形势分析及展望"的来源就是在讲展望（实测：光明网展望专稿
        与两条排名页同分，靠 id 小被 k 截掉，"未来展望"整节因此占位）。

        标题命中再分两级：小节标题自己的词（"展望"）强于泛同义词（"趋势"）——
        否则"…及发展趋势"的综述会和展望专稿在标题层继续打平。

        权威分层（本次改造）：权威来源放宽相关性门槛（sec_hits≥1 即可），
        并在排序里高于普通站点。理由：央媒/部委用编辑措辞（"规模预计1270万人"），
        字面命中天然低于照抄查询词的聚合站——实测「毕业总人数预测」一节，
        人民网/新华网 sec_hits=1 被门槛整条挡掉，而标题写"数据错得离谱"的
        自媒体 sec_hits=4 入选，报告最终引用了错误的 1250 万而非权威的 1270 万。
        相关性仍由 w4（小节 4 字窗口）作首要信号，权威只在同层级内优先。
        """
        sec_terms = content_terms(section)
        if not sec_terms:
            sec_terms = content_terms(question)
        aspects = aspect_terms(section)
        # 小节区别于问题的二元组才有判别力：问题词每条来源都有。
        distinctive = _bigrams(section) - _bigrams(question) or _bigrams(section)
        need = min(2, len(sec_terms))
        scored = []
        for s in self.sources:
            tier = authority_tier(_host(s.url))
            w4, d2 = self.section_match(s, section, question)
            sec_hits = self._hits(s, sec_terms)
            topic_hits = self._hits(s, content_terms(question))
            aspect_hits = self._hits(s, aspects)
            relevant = (sec_hits >= need
                        or (topic_hits >= 2 and aspect_hits > 0)
                        or (tier > 0 and sec_hits >= 1))
            if not sec_terms or not relevant:
                continue
            title_own = self._title_hits(s, distinctive)
            title_aspect = self._title_hits(s, aspects - distinctive)
            scored.append((w4, tier, title_own, title_aspect, aspect_hits, sec_hits, s.id, s))
        scored.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3], -x[4], -x[5], x[6]))
        return [s for *_, s in scored[:k]]

    def has_section_material(self, section: str) -> bool:
        """是否存在与小节标题直接相关的来源（仅用小节自身二元组判断）。"""
        return bool(self.select_for(section, "", k=1))

    def coverage(self, section: str, question: str, threshold: int = 3) -> int:
        """与小节有实质相关性的来源数；反思阶段的规则兜底依据。"""
        return len(self.select_for(section, question, k=len(self.sources)))


def source_material(sources: list[Source], query: str, max_chars: int = 1800,
                    section: str = "") -> str:
    """section 给出后，该节的特征词在片段筛选中加权（见 select_evidence）。"""
    boost = (content_terms(section) | aspect_terms(section)) if section else None
    return "\n\n".join(
        f"[{s.id}] 《{s.title}》\n发布机构：{s.publisher or '未标明'}；"
        f"发布日期：{s.published_at or '未标明'}（不是数据所属期间）\n"
        f"原文证据：\n{s.excerpt(query, max_chars, boost)}" for s in sources
    )
