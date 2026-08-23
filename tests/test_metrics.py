"""eval.metrics 覆盖率与汇总测试。"""
from eval.metrics import coverage, aggregate


class TestCoverage:
    def test_all_hit(self):
        r = coverage("中国新能源汽车出口欧洲，比亚迪领跑", ["出口", "欧洲", "比亚迪"])
        assert r["rate"] == 1.0
        assert len(r["missed"]) == 0

    def test_partial_hit(self):
        r = coverage("出口欧洲", ["出口", "欧洲", "比亚迪"])
        assert r["rate"] == round(2 / 3, 3)
        assert "比亚迪" in r["missed"]

    def test_no_hit(self):
        r = coverage("无关内容", ["出口", "欧洲"])
        assert r["rate"] == 0.0

    def test_empty_must_cover(self):
        assert coverage("text", [])["rate"] == 1.0


class TestAggregate:
    def test_basic(self):
        results = [
            {"ok": True, "coverage": {"rate": 1.0}, "stats": {"json_first_try_rate": 1.0,
             "tool_valid_rate": 0.86, "llm_calls": 20, "total_seconds": 36}},
            {"ok": True, "coverage": {"rate": 0.5}, "stats": {"json_first_try_rate": 1.0,
             "tool_valid_rate": 1.0, "llm_calls": 15, "total_seconds": 30}},
        ]
        agg = aggregate(results)
        assert agg["tasks"] == 2
        assert agg["success"] == 2
        assert agg["coverage_avg"] == 0.75
        assert agg["tool_valid_rate_avg"] == 0.93

    def test_with_failed_task(self):
        results = [
            {"ok": True, "coverage": {"rate": 1.0}, "stats": {"json_first_try_rate": 1.0,
             "tool_valid_rate": 1.0, "llm_calls": 10, "total_seconds": 30}},
            {"ok": False, "coverage": {"rate": 0.0}, "stats": {}},
        ]
        agg = aggregate(results)
        assert agg["tasks"] == 2
        assert agg["success"] == 1
        # 失败任务按零覆盖计入分母
        assert agg["coverage_avg"] == 0.5

    def test_empty(self):
        assert aggregate([]) == {}

    def test_faithfulness_avg(self):
        results = [
            {"ok": True, "coverage": {"rate": 1.0}, "stats": {"json_first_try_rate": 1.0,
             "tool_valid_rate": 1.0, "llm_calls": 10, "total_seconds": 30},
             "faith": {"faithfulness": 1.0, "citation_density": 0.14}},
            {"ok": True, "coverage": {"rate": 1.0}, "stats": {"json_first_try_rate": 1.0,
             "tool_valid_rate": 1.0, "llm_calls": 10, "total_seconds": 30},
             "faith": {"faithfulness": 0.5, "citation_density": 0.0}},
        ]
        agg = aggregate(results)
        assert agg["faithfulness_avg"] == 0.75
        assert agg["citation_density_avg"] == 0.07
