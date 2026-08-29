"""知识库：收录已读网页的摘要，并按报告小节做相关性选材。

这是上下文压缩的核心：正文被压成短摘要存库，写作时每个小节
只注入按相关性挑选的少量摘要，保证 3B 模型的上下文始终很小。
相关性用字符二元组重合度计算，中文场景下无需引入分词依赖。
"""
import re
from dataclasses import dataclass


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


class KnowledgeBase:
    def __init__(self):
        self.sources: list[Source] = []
        self._by_url: dict[str, int] = {}

    def __len__(self):
        return len(self.sources)

    def add(self, url: str, title: str, digest: str) -> int:
        """按 URL 去重；重复收录返回已有编号。"""
        if url in self._by_url:
            return self._by_url[url]
        sid = len(self.sources) + 1
        self.sources.append(Source(sid, url, title, digest))
        self._by_url[url] = sid
        return sid

    def _hits(self, source: Source, terms: set[str]) -> int:
        text = f"{source.title} {source.digest}"
        return sum(1 for t in terms if t in text)

    def relevance(self, source: Source, section: str, question: str) -> int:
        return self._hits(source, _bigrams(section) | _bigrams(question))

    def section_match(self, source: Source, section: str, question: str) -> tuple[int, int]:
        """返回 (标题 4 字窗口命中数, 小节区别于问题的二元组命中数)。

        4 字窗口用来把「货物贸易」和「贸易伙伴」分开；二元组作次级信号。
        """
        text = f"{source.title} {source.digest}"
        w4 = sum(1 for w in _windows4(section) if w in text)
        distinctive = _bigrams(section) - _bigrams(question)
        d2 = sum(1 for t in distinctive if t in text) if distinctive else self._hits(source, _bigrams(section))
        return w4, d2

    def select_for(self, section: str, question: str, k: int = 5) -> list[Source]:
        """挑选与小节最相关的 k 条来源，始终返回 k 条（库不足时返回全部）。

        4 字窗口与区别于问题的二元组只作排序加权，不作入选门槛——
        实测把窗口当门槛会让每节只剩 1 条素材，3B 模型既要扣题又要带引用
        难度过高，反而更倾向输出占位符。给足素材让模型有得选更重要。
        """
        sec_terms = _bigrams(section)
        scored = []
        for s in self.sources:
            w4, d2 = self.section_match(s, section, question)
            sec_hits = self._hits(s, sec_terms)
            scored.append((w4, d2, sec_hits, s.id, s))
        scored.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
        return [s for *_, s in scored[:k]]

    def has_section_material(self, section: str) -> bool:
        """是否存在与小节标题直接相关的来源（仅用小节自身二元组判断）。"""
        terms = _bigrams(section)
        if not terms:
            return bool(self.sources)
        return any(self._hits(s, terms) > 0 for s in self.sources)

    def coverage(self, section: str, question: str, threshold: int = 3) -> int:
        """与小节有实质相关性的来源数；反思阶段的规则兜底依据。"""
        terms = _bigrams(section) | _bigrams(question)
        return sum(1 for s in self.sources if self._hits(s, terms) >= threshold)
