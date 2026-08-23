"""磁盘快照缓存：搜索结果与网页正文按内容哈希落盘。

- record: 未命中 → 真实请求并写盘（日常使用）
- replay: 只读，未命中直接抛错（评估回放：可复现、零 API 消耗、不依赖网络）
- off:    不使用缓存
"""
import hashlib
import json

from .. import config


class CacheMiss(KeyError):
    pass


class DiskCache:
    def __init__(self, subdir: str):
        self.mode = config.CACHE_MODE
        self.dir = config.CACHE_DIR / subdir
        if self.mode != "off":
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str):
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def get(self, key: str):
        if self.mode == "off":
            return None
        path = self._path(key)
        if not path.exists():
            if self.mode == "replay":
                raise CacheMiss(f"replay 模式下缓存未命中: {key[:80]}")
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value) -> None:
        if self.mode == "off":
            return
        self._path(key).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
