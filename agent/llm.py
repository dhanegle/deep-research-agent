"""Ollama 客户端封装。

三条调用路径，均针对 Qwen2.5-3B 的可靠性做了约束设计：
- chat():      原生 tool calling（ReAct 循环用）
- chat_json(): format=json + pydantic 校验 + 失败回喂修复（结构化输出用）
- chat_text(): 普通补全（网页摘要、报告写作）
"""
import json
import time
from typing import Type

from ollama import Client
from pydantic import BaseModel, ValidationError

from . import config
from .stats import RunStats
from .trace import Trace


def extract_json(text: str) -> str:
    """模型偶尔输出 ```json 围栏或前后夹带说明文字，先做鲁棒提取。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return text


def _no_think_needed() -> bool:
    """Qwen3 系列默认开思维链，对受限管线是纯开销，自动关闭。"""
    mode = config.THINK_MODE.lower()
    if mode == "on":
        return False
    if mode == "off":
        return True
    return config.MODEL.lower().startswith("qwen3")


def _append_no_think(messages: list[dict]) -> list[dict]:
    """在末条用户消息追加官方软开关 /no_think。

    实测（ollama 0.32）：chat(think=False) 会把思维链错路由进 content 污染正文，
    而软开关能把思维链留在独立的 thinking 字段，content 保持干净。
    """
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] += " /no_think"
            break
    return msgs


def _clock_prefix() -> str:
    """模型没有时钟，且训练数据截止较早：注入当前日期作为唯一时间基准。"""
    now = time.localtime()
    week = "日一二三四五六"[int(time.strftime("%w", now))]
    return (f"当前日期：{time.strftime('%Y-%m-%d', now)}（星期{week}）。"
            '凡涉及"今年/最近/当前/最新"等时间表述，一律以当前日期为准推算。')


def _with_clock(messages: list[dict]) -> list[dict]:
    """把当前日期并入首条 system 消息（没有则插入一条），全阶段调用生效。"""
    msgs = [dict(m) for m in messages]
    line = _clock_prefix()
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = line + "\n" + msgs[0]["content"]
    else:
        msgs.insert(0, {"role": "system", "content": line})
    return msgs


class LLM:
    def __init__(self, stats: RunStats, trace: Trace):
        self.stats = stats
        self.trace = trace
        self.client = Client(host=config.OLLAMA_HOST, timeout=config.LLM_TIMEOUT)

    # ---- 基础调用 -------------------------------------------------------
    def _chat(self, messages: list[dict], tools: list | None = None, json_mode: bool = False) -> dict:
        messages = _with_clock(messages)
        if _no_think_needed():
            messages = _append_no_think(messages)
        kwargs: dict = dict(
            model=config.MODEL,
            messages=messages,
            options={"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX},
        )
        if tools:
            kwargs["tools"] = tools
        if json_mode:
            kwargs["format"] = "json"
        if config.THINK_MODE.lower() == "on":
            kwargs["think"] = True
        t0 = time.time()
        try:
            resp = self.client.chat(**kwargs)
        except Exception as e:  # ollama 长连接偶发挂起，超时类错误重试一次
            if "timed out" not in str(e).lower() and "timeout" not in str(e).lower():
                raise
            self.trace.log("llm_timeout_retry", error=str(e)[:200])
            resp = self.client.chat(**kwargs)
        elapsed = round(time.time() - t0, 2)
        self.stats.llm_calls += 1
        self.stats.llm_seconds += elapsed
        message = resp["message"]
        if hasattr(message, "model_dump"):  # ollama SDK 返回 pydantic Message，统一转为纯 dict
            message = message.model_dump()
        self.trace.log(
            "llm_call",
            seconds=elapsed,
            tools=bool(tools),
            json_mode=json_mode,
            messages=messages,
            response=message,
        )
        return message

    def chat_text(self, messages: list[dict]) -> str:
        return (self._chat(messages).get("content") or "").strip()

    def stream_text(self, messages: list[dict]):
        """流式补全：逐段 yield 文本。写作阶段用于前端逐字渲染。

        与 chat_text 共享时钟注入/超时重试/统计/trace 逻辑，
        差异仅在于 stream=True 逐 chunk yield content。
        """
        messages = _with_clock(messages)
        if _no_think_needed():
            messages = _append_no_think(messages)
        kwargs: dict = dict(
            model=config.MODEL, messages=messages,
            options={"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX},
            stream=True,
        )
        if config.THINK_MODE.lower() == "on":
            kwargs["think"] = True
        t0 = time.time()
        try:
            stream = self.client.chat(**kwargs)
        except Exception as e:
            if "timed out" not in str(e).lower() and "timeout" not in str(e).lower():
                raise
            self.trace.log("llm_timeout_retry", error=str(e)[:200])
            stream = self.client.chat(**kwargs)
        self.stats.llm_calls += 1
        chunks: list[str] = []
        try:
            for chunk in stream:
                if hasattr(chunk, "model_dump"):
                    chunk = chunk.model_dump()
                content = (chunk.get("message") or {}).get("content") or ""
                if content:
                    chunks.append(content)
                    yield content
        finally:
            elapsed = round(time.time() - t0, 2)
            self.stats.llm_seconds += elapsed
            self.trace.log("llm_call", seconds=elapsed, tools=False, json_mode=False,
                           messages=messages, response={"content": "".join(chunks)})

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        return self._chat(messages, tools=tools)

    # ---- 结构化输出 -------------------------------------------------------
    def chat_json(self, messages: list[dict], schema: Type[BaseModel], tag: str):
        """带 pydantic 校验与自动修复的 JSON 调用，返回 schema 实例。

        修复策略：把校验错误原文回喂给模型重试（首试 + 最多 2 次修复）。
        首试成功率计入 stats，是核心评估指标之一。
        """
        self.stats.json_calls += 1
        conv = list(messages)
        text = ""
        last_err = "unknown"
        for attempt in range(3):
            try:
                text = (self._chat(conv, json_mode=True).get("content") or "").strip()
                model = schema.model_validate_json(extract_json(text))
                if attempt == 0:
                    self.stats.json_first_try_ok += 1
                else:
                    self.trace.log("json_repaired", tag=tag, attempts=attempt)
                return model
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:500]
                if attempt == 0:
                    self.stats.json_repairs += 1
                self.trace.log("json_invalid", tag=tag, attempt=attempt, error=last_err, raw=text[:300])
                conv = conv + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": (
                        f"你上一条输出不符合要求的格式，错误信息：{last_err}\n"
                        "请重新输出。只输出一个合法的 JSON 对象，不要任何解释文字或代码块标记。"
                    )},
                ]
        raise ValueError(f"结构化输出 3 次尝试后仍失败 [{tag}]: {last_err}")


def tool_call_name(tc: dict) -> str:
    return (tc.get("function") or {}).get("name", "")


def tool_call_args(tc: dict) -> dict:
    """解析工具调用参数；ollama 有时把 arguments 给成 JSON 字符串，做兼容。"""
    args = (tc.get("function") or {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}
