"""单次运行指标：既是运行时观测，也是评估体系的数据来源。"""
from dataclasses import asdict, dataclass


@dataclass
class RunStats:
    llm_calls: int = 0
    llm_seconds: float = 0.0
    json_calls: int = 0          # chat_json 调用次数（结构化输出）
    json_first_try_ok: int = 0   # 其中首试即通过 pydantic 校验的次数
    json_repairs: int = 0        # 触发回喂修复的次数
    tool_calls: int = 0          # 模型发起的工具调用总数
    tool_invalid: int = 0        # 其中非法（未知工具/缺参数/编造链接）的次数
    react_fallbacks: int = 0     # 原生 tool call 连续失败后降级 JSON-ReAct 的步数
    searches: int = 0
    pages_fetched: int = 0
    digest_calls: int = 0
    reflect_rounds: int = 0
    review_rounds: int = 0      # 报告自审轮次（0 或 1）
    review_issues: int = 0     # 自审发现的问题数
    review_revised: int = 0     # 自审实际重写的节数
    total_seconds: float = 0.0

    @property
    def json_first_try_rate(self) -> float:
        return self.json_first_try_ok / self.json_calls if self.json_calls else 1.0

    @property
    def tool_valid_rate(self) -> float:
        return 1 - self.tool_invalid / self.tool_calls if self.tool_calls else 1.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["json_first_try_rate"] = round(self.json_first_try_rate, 3)
        d["tool_valid_rate"] = round(self.tool_valid_rate, 3)
        return d
