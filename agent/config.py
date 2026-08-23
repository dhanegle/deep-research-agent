"""全局配置：从 .env 读取，未配置时使用对 Qwen2.5-3B 友好的默认值。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen2.5:3b")
# 上下文窗口：资料已做摘要压缩，8192 足够容纳，且比更大窗口的提示处理更快
NUM_CTX = _int("NUM_CTX", 8192)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
# 思考模式（Qwen3 系列）：auto=qwen3 自动关闭（受限管线不需要思维链开销）
# off=强制关闭 on=强制开启
THINK_MODE = os.getenv("THINK_MODE", "auto")

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "local")  # local | tavily | bocha
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")
# basic=快但相关性差（1 credit）| advanced=相关性更好（2 credits），实测对中文/专有名词查询提升明显
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "advanced")
# 是否走系统代理（如 Clash）。实测系统代理不稳时 Tavily 会出现 SSL 断连/403，
# 而直连正常，因此默认直连（trust_env=False）；确需代理时改为 1。
USE_SYSTEM_PROXY = os.getenv("USE_SYSTEM_PROXY", "0") == "1"
CACHE_MODE = os.getenv("CACHE_MODE", "record")  # record | replay | off
# local 搜索源的文档文件夹（相对路径基于项目根目录）
LOCAL_DOCS_DIR = (ROOT / os.getenv("LOCAL_DOCS_DIR", "data")).resolve()

CACHE_DIR = ROOT / "cache"
REPORTS_DIR = ROOT / "reports"
TRACES_DIR = ROOT / "traces"

MAX_STEPS = _int("MAX_STEPS", 10)              # Researcher 单轮 ReAct 最大步数
MAX_REFLECT_ROUNDS = _int("MAX_REFLECT_ROUNDS", 2)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 300)
# 工具调用路径: native=原生 tool calling | prompt=提示词 JSON-ReAct（消融实验用）
REACT_MODE = os.getenv("REACT_MODE", "native")
# 朴素基线开关: 最简提示词 + 关闭行为约束（min_sources/nudge/搜索去重），消融实验用
NAIVE = os.getenv("NAIVE", "0") == "1"
