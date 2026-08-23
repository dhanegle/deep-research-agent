"""conftest：让测试既支持 pytest 也支持无框架运行（python -m tests.run）。

把项目根加入 sys.path，使 agent/ 和 eval/ 可直接 import。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
