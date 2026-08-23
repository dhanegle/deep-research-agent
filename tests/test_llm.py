"""agent.llm 纯逻辑测试：JSON 鲁棒提取、/no_think 软开关、时钟注入。

这些函数不依赖 ollama，可以离线跑、可复现。
"""
from agent.llm import extract_json, _append_no_think, _with_clock


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fenced_plain(self):
        assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_with_prose(self):
        raw = '好的，结果如下：\n{"outline": ["a"], "queries": ["b"]}\n以上。'
        assert extract_json(raw) == '{"outline": ["a"], "queries": ["b"]}'

    def test_no_json_returns_input(self):
        # 无 {} 时原样返回（调用方后续 pydantic 校验会失败）
        assert extract_json("hello world") == "hello world"

    def test_nested_braces(self):
        raw = 'prefix {"a": {"b": 2}, "c": 3} suffix'
        assert extract_json(raw) == '{"a": {"b": 2}, "c": 3}'


class TestAppendNoThink:
    def test_appends_to_last_user(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _append_no_think(msgs)
        assert out[-1]["content"] == "hi /no_think"

    def test_appends_to_last_user_among_many(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        out = _append_no_think(msgs)
        assert out[-1]["content"] == "q2 /no_think"
        # 不动非末条 user 消息
        assert out[1]["content"] == "q1"

    def test_no_user_message(self):
        msgs = [{"role": "system", "content": "sys"}]
        out = _append_no_think(msgs)
        # 没有 user 消息则原样返回（不崩）
        assert out == msgs

    def test_does_not_mutate_input(self):
        msgs = [{"role": "user", "content": "hi"}]
        _append_no_think(msgs)
        assert msgs[0]["content"] == "hi"  # 原列表未改


class TestWithClock:
    def test_prepends_to_system(self):
        msgs = [{"role": "system", "content": "你是助手"}]
        out = _with_clock(msgs)
        assert out[0]["role"] == "system"
        assert "当前日期" in out[0]["content"]
        assert "你是助手" in out[0]["content"]

    def test_inserts_system_when_absent(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = _with_clock(msgs)
        assert out[0]["role"] == "system"
        assert "当前日期" in out[0]["content"]
        assert out[1] == msgs[0]

    def test_includes_weekday(self):
        out = _with_clock([{"role": "system", "content": "x"}])
        # 星期后跟"日一二三四五六"之一
        assert any(f"星期{d}" in out[0]["content"] for d in "日一二三四五六")
