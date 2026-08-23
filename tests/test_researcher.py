"""agent.researcher 路径容错测试：_resolve_path 的前缀剥离与后缀匹配。

这是 3B 模型最频繁的失败模式之一（给路径编造 https:// 前缀或截断），
参数命名 url→path + 容错匹配曾把同一任务的工具有效率从 39% 拉到 100%。
测试锁定该行为。
"""
from agent.researcher import Researcher


def _seen(*urls):
    return {u: f"标题{i}" for i, u in enumerate(urls)}


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
