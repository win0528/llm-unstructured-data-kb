"""
Web 界面冒烟测试

用 Streamlit 官方的 AppTest 无头执行 app.py，逐个切换页面并断言没有抛异常。
避免"改了后端逻辑、界面一打开就报错"这类问题。

用法:
  python tests/test_app_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402


PAGES = ["文件上传", "数据处理", "清洗效果预览", "知识库检索", "统计报告"]


def main() -> int:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if at.exception:
        print(f"[FAIL] 首页渲染异常: {[e.value for e in at.exception]}")
        return 1
    print("  [OK] 首页渲染正常（文件上传）")

    for page in PAGES[1:]:
        at.sidebar.radio[0].set_value(page)
        at.run()
        if at.exception:
            print(f"[FAIL] 页面「{page}」渲染异常: {[e.value for e in at.exception]}")
            return 1
        print(f"  [OK] 页面「{page}」渲染正常")

    # 侧边栏应当能读到索引 manifest
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    if "知识库状态" in sidebar_text:
        print("  [OK] 侧边栏已显示知识库状态")

    print("\n界面冒烟测试通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
