"""全链路 JSONL 追踪：每次 LLM 调用与工具执行都落盘，用于调试与评估复盘。"""
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from . import config

# (事件类型, 摘要信息) —— CLI/Web UI 通过它做实时步骤展示
StepCallback = Callable[[str, dict], None]


class Trace:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        config.TRACES_DIR.mkdir(parents=True, exist_ok=True)
        self.path: Path = config.TRACES_DIR / f"{self.run_id}.jsonl"

    def log(self, event: str, **payload: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
