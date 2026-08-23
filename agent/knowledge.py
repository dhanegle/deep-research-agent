"""知识库：收录已读网页的摘要，并按报告小节做相关性选材。

这是上下文压缩的核心：正文被压成短摘要存库，写作时每个小节
只注入按相关性挑选的少量摘要，保证 3B 模型的上下文始终很小。
相关性用字符二元组重合度计算，中文场景下无需引入分词依赖。
"""
import re
from dataclasses import dataclass


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


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


def ascii_token_hit(text: str, tokens: list[tuple[str, float]]) -> bool:
    """查询含 ASCII 专有名词（如 "linuxsb"、"2025"）时的强判别：

    结果必须至少命中其中一个，否则视为无关。防止"网站/介绍"这类泛词
    让 GitHub 上的"个人介绍网站.html"之类的页面混过覆盖率门槛。
    不含 ASCII token 的纯中文查询恒为 True。
    """
    low = text.lower()
    compact = re.sub(r"[._\-]", "", low)
    for tok, _ in tokens:
        if not tok.isascii():
            continue
        tok_c = re.sub(r"[._\-]", "", tok)
        if tok in low or tok in compact or tok_c in compact:
            return True
    return not any(tok.isascii() for tok, _ in tokens)


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
        elif len(tok) >= 7 and not tok.isascii():
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

    def select_for(self, section: str, question: str, k: int = 5) -> list[Source]:
        """挑选与小节最相关的 k 条来源（按二元组重合度降序）。"""
        ranked = sorted(self.sources, key=lambda s: (-self.relevance(s, section, question), s.id))
        return ranked[:k]

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
