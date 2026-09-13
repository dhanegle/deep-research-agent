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


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
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
# general=全站召回 | news=只召回新闻站点。实测两者召回几乎不重叠（重叠 4/32）：
# news 独有央视财经/人民日报，general 独有海关总署/商务部等一手来源。
# news 下 Tavily 才返回 published_date（general 实测 0/32），但切过去会损失政府
# 站点召回，因此默认仍是 general，发布日期由 URL 内嵌日期兜底（权威域覆盖≈100%）。
TAVILY_TOPIC = os.getenv("TAVILY_TOPIC", "general")
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
MAX_SOURCES = _int("MAX_SOURCES", 12)
MAX_SEARCHES = _int("MAX_SEARCHES", 12)
MAX_REVIEW_CLAIMS = _int("MAX_REVIEW_CLAIMS", 24)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 300)
# 工具调用路径: native=原生 tool calling | prompt=提示词 JSON-ReAct（消融实验用）
REACT_MODE = os.getenv("REACT_MODE", "native")
# 权威定向补搜：常规搜索没有权威来源时，额外对权威域名做一次定向检索并前置。
# 实测原始召回里央媒仅占 2.8%，被动排序不够；关闭可省搜索额度（AUTHORITY_PASS=0）
AUTHORITY_PASS = os.getenv("AUTHORITY_PASS", "1") == "1"
# 单次调研最多执行几次权威定向检索（每次消耗 1 次搜索额度）
AUTHORITY_PASS_BUDGET = _int("AUTHORITY_PASS_BUDGET", 3)
# 时效偏好：发布在最近 N 天内的来源获得 FRESH_BONUS 加分（**只奖不罚**）。
# 此前排序只有权威层级与相关度，没有时间维度——实测候选里 2026 年 546 条、
# 2025 年 102 条、2024 年及更早 50 条，旧稿与新闻同权，旧稿还常因是央媒排更前。
# 注意不能改成"惩罚旧内容"：多数政府一手来源的 URL 不含日期，惩罚旧的会让
# "日期未知"相对占优，反而把确知较旧的央视/新华网挤出候选（离线回放实测 22 例）。
# FRESH_DAYS=0 可关闭时效偏好（退回纯权威+相关度排序）。
FRESH_DAYS = _int("FRESH_DAYS", 180)
FRESH_BONUS = _float("FRESH_BONUS", 0.05)
# 成文后跨节去重的相似度阈值：1.0=只去逐字重复，0.85 能抓到"差一个词"的改述
# （实测 3B 会把同一事实在 3 个小节各写一遍，措辞略异）。数字口径不相容的句子
# 一律不判重，因此调低阈值不会拿不同数据互相覆盖。
DEDUPE_SIMILARITY = _float("DEDUPE_SIMILARITY", 0.85)
# 朴素基线开关: 最简提示词 + 关闭行为约束（min_sources/nudge/搜索去重），消融实验用
NAIVE = os.getenv("NAIVE", "0") == "1"
