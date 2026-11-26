"""反白预检测试：曼哈顿距离、BFS 最短通道、取数边界、参数校验与 API。

运行：python -I tests/test_preflight.py
"""

from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import preflight as PF  # noqa: E402
from app.server import app  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        raise AssertionError(name)


def brute_manhattan(out: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """对 sources 布尔数组中的黑点做朴素 O(N*K) 曼哈顿距离。"""

    h, w = out.shape
    ys, xs = np.nonzero(sources)
    dist = np.full((h, w), PF._INF, dtype=np.int64)
    for r in range(h):
        for c in range(w):
            if len(xs):
                dist[r, c] = int(np.min(np.abs(xs - c) + np.abs(ys - r)))
    return dist


def test_distance_exact() -> None:
    print("test_distance_exact")
    rng = np.random.default_rng(3)
    for trial in range(6):
        h = w = 40 + 12 * trial
        out = np.where(rng.random((h, w)) < 0.18, 0, 255).astype(np.uint8)
        x0, y0, x1, y1 = 5, 6, min(w - 2, 26 + trial), min(h - 2, 30)
        dist, cy0, cx0 = PF.manhattan_black_distance(out, x0, y0, x1, y1)
        crop = out[cy0:cy0 + dist.shape[0], cx0:cx0 + dist.shape[1]]
        expect = brute_manhattan(crop, crop == 0)
        check(f"trial{trial} 裁剪区内距离与朴素实现完全一致",
              np.array_equal(dist, expect))

    # 无黑点：距离保持 INF，不溢出
    blank = np.full((40, 40), 255, np.uint8)
    dist, _, _ = PF.manhattan_black_distance(blank, 4, 4, 30, 30)
    check("全白图距离为 INF 且无溢出", bool(np.all(dist >= PF._INF - 40)))


def test_read_boundary() -> None:
    print("test_read_boundary")
    # 框 [20,32)x[20,32)。黑点 (4,20) 距框角 (20,20)=16 → 影响第 16 档；
    # 黑点 (3,20) 距离 17 → 对 0..16 档判定完全无影响。
    img16 = np.full((60, 60), 255, np.uint8)
    img16[4, 20] = 0
    r16 = PF.analyze(img16, 20, 20, 32, 32, [[20, 20], [31, 31]])
    check("距离恰好 16 的黑点影响第 16 档（最高档=15）", r16["max_level"] == 15)
    check("read_region 为框外扩 16 并裁剪",
          r16["read_region"] == {"x0": 4, "y0": 4, "x1": 48, "y1": 48})

    img17 = np.full((60, 60), 255, np.uint8)
    img17[3, 20] = 0
    r17 = PF.analyze(img17, 20, 20, 32, 32, [[20, 20], [31, 31]])
    check("距离 17 的框外黑点不影响任何 0..16 档（最高档=16）",
          r17["max_level"] == 16)

    # 贴图像边界的框：read_region 被裁剪到 0，不报错
    edge = np.full((40, 40), 255, np.uint8)
    re = PF.analyze(edge, 0, 0, 10, 10, [[1, 1], [8, 8]])
    check("贴边框 read_region 裁剪到图像范围",
          re["read_region"]["x0"] == 0 and re["read_region"]["y0"] == 0)


def _stripe(thickness: int, gap: bool = False):
    h = w = 60
    a = np.zeros((h, w), np.uint8)
    y = h // 2
    a[y - thickness // 2:y - thickness // 2 + thickness, :] = 255
    if gap:
        a[y - thickness // 2:y - thickness // 2 + thickness, 28:33] = 0
    return a, y


def test_channel_levels_and_path() -> None:
    print("test_channel_levels_and_path")
    expectations = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}
    for t, want in expectations.items():
        out, y = _stripe(t)
        r = PF.analyze(out, 4, y - 8, 56, y + 9, [[10, y], [50, y]])
        check(f"{t}px 线最高档={want}（实际 {r['max_level']}）",
              r["max_level"] == want)
        check("始终给出 17 个档位状态", len(r["levels"]) == 17)
        connected = [l["connected"] for l in r["levels"]]
        check("档位连通性单调（前缀 True 后全 False）",
              all(connected[:r["max_level"] + 1]) and
              not any(connected[r["max_level"] + 1:]))

        # 校验返回路径：端点、四邻接、当档全部可通行
        path = r["path"]
        check("路径首尾为指定端点",
              path[0] == [10, y] and path[-1] == [50, y])
        adj = all(abs(path[i + 1][0] - path[i][0]) +
                  abs(path[i + 1][1] - path[i][1]) == 1
                  for i in range(len(path) - 1))
        check("路径全部为上下左右相邻", adj)
        d, cy0, cx0 = PF.manhattan_black_distance(out, 4, y - 8, 56, y + 9)
        k = r["max_level"]
        check("路径每个像素距离 > 最高档（当档确为白点）",
              all(d[py - cy0, px - cx0] > k for px, py in path))
        reported = next(l["shortest_length"] for l in r["levels"]
                        if l["level"] == k)
        check("路径边数等于 BFS 最短长度", len(path) - 1 == reported)

    out, y = _stripe(6, gap=True)
    r = PF.analyze(out, 4, y - 8, 56, y + 9, [[10, y], [50, y]])
    check("有断口时零档已断", r["max_level"] is None
          and r["level0_connected"] is False
          and r["reason"] == "broken_at_zero")
    check("零档断时不返回通道线", r["path"] == [])


def test_endpoint_black_lenient() -> None:
    print("test_endpoint_black_lenient")
    out = np.full((20, 20), 255, np.uint8)
    out[5, 5] = 0
    r = PF.analyze(out, 0, 0, 20, 20, [[5, 5], [15, 15]])
    check("端点是黑点时不抛错，判为零档断并标记 endpoint_black",
          r["max_level"] is None and r["reason"] == "endpoint_black"
          and r["endpoints_white"] == [False, True])


def test_validation() -> None:
    print("test_validation")
    out = np.full((300, 300), 255, np.uint8)

    def must_fail(label, *args):
        try:
            PF.analyze(out, *args)
        except PF.PreflightError:
            check(label, True)
        else:
            check(label + "（应当报错）", False)

    must_fail("宽超过 256 被拒", 0, 0, 257, 10, [[1, 1], [2, 2]])
    must_fail("高超过 256 被拒", 0, 0, 10, 257, [[1, 1], [2, 2]])
    must_fail("空框被拒", 10, 10, 10, 20, [[11, 11], [12, 12]])
    must_fail("越界框被拒", 0, 0, 301, 10, [[1, 1], [2, 2]])
    must_fail("端点在框外被拒", 0, 0, 20, 20, [[1, 1], [25, 5]])
    must_fail("两端点相同被拒", 0, 0, 20, 20, [[5, 5], [5, 5]])


def test_performance() -> None:
    print("test_performance")
    rng = np.random.default_rng(7)
    out = np.where(rng.random((256, 256)) < 0.03, 0, 255).astype(np.uint8)
    t0 = time.perf_counter()
    r = PF.analyze(out, 0, 0, 256, 256, [[2, 128], [253, 128]])
    dt = time.perf_counter() - t0
    check(f"256x256 全 17 档试算在 1.0s 内（实测 {dt:.3f}s）", dt < 1.0)
    print(f"    最高档={r['max_level']}，路径像素={len(r['path'])}")


# ---------------------------------------------------------------- API
client = app.test_client()


def _upload(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    r = client.post("/api/upload",
                    data={"image": (io.BytesIO(buf.getvalue()), "g.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()["session_id"]


def test_api_preflight() -> None:
    print("test_api_preflight")
    # 60x60 黑底 + 横白道（厚 5）
    img = np.zeros((60, 60), np.uint8)
    img[28:33, :] = 255
    sid = _upload(img)
    rid0 = client.get(f"/api/session/{sid}").get_json()["current_revision_id"]

    r = client.post(f"/api/session/{sid}/preflight", json={
        "x0": 4, "y0": 20, "x1": 56, "y1": 40,
        "endpoints": [[10, 30], [50, 30]],
    })
    check("预检 200", r.status_code == 200)
    data = r.get_json()
    check("5px 道 API 最高档=2", data["max_level"] == 2)
    check("返回当前 revision_id", data["revision_id"] == rid0)
    check("返回最短通道坐标", len(data["path"]) > 0
          and data["path"][0] == [10, 30])

    # 参数错误 → 400
    r = client.post(f"/api/session/{sid}/preflight", json={
        "x0": 0, "y0": 0, "x1": 300, "y1": 10,
        "endpoints": [[1, 1], [2, 2]]})
    check("超宽框 API 400", r.status_code == 400)
    r = client.post(f"/api/session/{sid}/preflight", json={
        "x0": 0, "y0": 0, "x1": 20, "y1": 20,
        "endpoints": [[3, 3], [3, 3]]})
    check("相同端点 API 400", r.status_code == 400)
    r = client.post("/api/session/no-such/preflight", json={
        "x0": 0, "y0": 0, "x1": 2, "y1": 2,
        "endpoints": [[0, 0], [1, 1]]})
    check("未知会话 400", r.status_code == 400)


def test_api_revision_guard_support() -> None:
    print("test_api_revision_guard_support")
    # 用预检演示图的几何构造上传，验证：接受换稿 → revision_id 变；
    # 拒绝换稿 → current_revision_id 不变（前端据此做过期判定/沿用结论）。
    demo_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "sample", "preflight_demo.png")
    img = np.array(Image.open(demo_path).convert("L"))
    sid = _upload(img)
    box = {"x0": 24, "y0": 48, "x1": 236, "y1": 96}
    eps = [[44, 59], [214, 85]]

    d0 = client.post(f"/api/session/{sid}/preflight",
                     json={**box, "endpoints": eps}).get_json()
    check("演示 Z 线初始最高档=1", d0["max_level"] == 1)

    # 无条带时，细颈横切强制黑必被接受
    r = client.post(f"/api/session/{sid}/force", json={
        "x0": 116, "y0": 74, "x1": 121, "y1": 78, "mode": "black"})
    fd = r.get_json()
    check("细颈切口换稿被接受", fd["result"]["accepted"] is True)
    rid1 = fd["session"]["current_revision_id"]
    check("接受后 revision 变化", rid1 != d0["revision_id"])
    d1 = client.post(f"/api/session/{sid}/preflight",
                     json={**box, "endpoints": eps}).get_json()
    check("按原框/端点重测：零档已断", d1["max_level"] is None)
    check("重测响应带新 revision_id", d1["revision_id"] == rid1)

    # 新会话：登记覆盖切口传播区的条带，再切 → 拒绝，revision 不变
    sid2 = _upload(img)
    client.post(f"/api/session/{sid2}/strips",
                json={"top_line": 60, "bottom_line": 95, "label": "颈区条带"})
    before = client.get(f"/api/session/{sid2}").get_json()["current_revision_id"]
    r = client.post(f"/api/session/{sid2}/force", json={
        "x0": 116, "y0": 74, "x1": 121, "y1": 78, "mode": "black"})
    check("条带内切口被拒绝", r.get_json()["result"]["accepted"] is False)
    after = client.get(f"/api/session/{sid2}").get_json()["current_revision_id"]
    check("拒绝后 revision 不变（前端沿用当前预检结果）", before == after)
    d2 = client.post(f"/api/session/{sid2}/preflight",
                     json={**box, "endpoints": eps}).get_json()
    check("拒绝后预检仍基于旧稿，最高档=1", d2["max_level"] == 1
          and d2["revision_id"] == before)


if __name__ == "__main__":
    test_distance_exact()
    test_read_boundary()
    test_channel_levels_and_path()
    test_endpoint_black_lenient()
    test_validation()
    test_performance()
    test_api_preflight()
    test_api_revision_guard_support()
    print(f"\n预检测试完成：{PASS} 通过，{FAIL} 失败")
    sys.exit(1 if FAIL else 0)
