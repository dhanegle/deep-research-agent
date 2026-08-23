"""Web UI：FastAPI + SSE 实时展示 Agent 每一步。

启动：
    python -m agent.web
然后浏览器打开 http://127.0.0.1:8000

流程：管线在后台线程运行，on_step 事件推入队列；
SSE 端点把队列内容实时推给浏览器，结束时附带完整报告与运行指标。
"""
import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from .pipeline import ResearchPipeline

app = FastAPI(title="Deep Research Agent")
INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX.read_text(encoding="utf-8")


@app.get("/api/research")
def research(q: str, provider: str | None = None):
    if not q.strip():
        return StreamingResponse(iter(["data: " + json.dumps(
            {"event": "error", "data": {"message": "问题不能为空"}}, ensure_ascii=False) + "\n\n"]),
            media_type="text/event-stream")

    events: queue.Queue = queue.Queue()

    def emit(event: str, data: dict) -> None:
        events.put((event, data))

    def emit_chunk(text: str) -> None:
        events.put(("chunk", {"text": text}))

    def worker() -> None:
        try:
            result = ResearchPipeline(provider, on_step=emit, on_chunk=emit_chunk).run(q)
            events.put(("done", {
                "report": result.report,
                "report_path": result.report_path,
                "stats": result.stats,
                "trace_path": result.trace_path,
            }))
        except Exception as e:  # 管线任意失败都要推给前端并结束流
            events.put(("error", {"message": str(e)[:300]}))
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                break
            event, data = item
            yield "data: " + json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    print("Deep Research Agent Web UI → http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
