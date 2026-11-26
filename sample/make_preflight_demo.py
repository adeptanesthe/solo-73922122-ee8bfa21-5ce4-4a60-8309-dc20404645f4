"""生成反白细线预检演示图 sample/preflight_demo.png（8 位灰度 PNG）。

图像只含纯黑(0)/纯白(255)，Floyd–Steinberg 抖动对它是恒等映射，因此
“成稿”与本图逐像素相同，预检结果完全可预期：

- 上半区一条直角转折的 6px 粗反白线，中间夹一段 3px 细颈
  → 最高可承受扩墨档为 1；可在细颈处强制黑/白观察重测；
- 下半区一条带 2px 断口的反白线 → 零档即断，用于演示“零档已断”提示。

尺寸 256x256。运行：python sample/make_preflight_demo.py
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

W = H = 256


def _rect(a: np.ndarray, x0: int, y0: int, x1: int, y1: int, v: int = 255) -> None:
    """填充半开区间矩形 [x0,x1) x [y0,y1)。"""

    a[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = v


def build() -> np.ndarray:
    a = np.zeros((H, W), dtype=np.uint8)  # 黑底

    # ---------- 上半区：Z 形反白通道（主演示） ----------
    # 上横段：行 56..61（6px 粗），列 32..122
    _rect(a, 32, 56, 122, 62)
    # 中间 3px 细颈：列 117..119，行 56..88（连接上下横段）
    _rect(a, 117, 56, 120, 88)
    # 下横段：行 82..87（6px 粗），列 117..226
    _rect(a, 117, 82, 227, 88)
    # 两个建议端点（放在粗段中央、远离断头）：A=(44,59) B=(214,85)

    # 细颈旁的“改稿切口”提示块：颈右侧留一块可安全下强制黑的纯黑区域，
    # 这里仅用稍亮? 不，保持纯黑，坐标写进 README。

    # ---------- 下半区：带断口的反白线（零档已断演示） ----------
    # 横段行 176..181（6px 粗），中间 x=126..129 留 3px 断口
    _rect(a, 32, 176, 126, 182)
    _rect(a, 129, 176, 227, 182)
    # 建议端点 C=(44,179) D=(214,179)

    # ---------- 左下角一条更粗的直线（高档位对照，可选） ----------
    _rect(a, 28, 214, 120, 224)   # 10px 粗 → 理论最高档 4
    return a


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(out_dir, "preflight_demo.png")
    img = Image.fromarray(build(), mode="L")
    img.save(path, format="PNG", optimize=True)
    print(f"预检演示图已生成：{path} ({img.size[0]}x{img.size[1]}, mode={img.mode})")


if __name__ == "__main__":
    main()
