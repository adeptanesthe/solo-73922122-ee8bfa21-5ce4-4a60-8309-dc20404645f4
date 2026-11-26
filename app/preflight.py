"""反白细线扩墨预检（打样前）。

物理模型
--------
- 扩墨档 ``k`` 取 0..16 的整数像素。某像素到**当前成稿**中最近黑点的
  曼哈顿距离 ``d`` 满足 ``d <= k`` 时，该像素在当档被视为黑点（墨点扩开后
  被吃掉，不可通行）；否则仍为白点。
- 反白通道只能由检查框内上/下/左/右相邻的白点组成。
- 取数范围只需“检查框 + 四周 16 像素”：框外更远处的黑点到框内任意像素的
  曼哈顿距离必 > 16，不可能影响任何一档（k <= 16）的判定。
- 档位单调：k 越大，视为黑点的集合越大、可通行集合越小；因此从 0 档起
  逐档 BFS，遇到首个断档即可停止——最后一个连通档即“仍有通道的最高档”，
  该档 BFS 的父指针重建出一条边数最少的最短通道。

本模块不依赖 Flask / 会话对象，可独立单测。坐标约定：
- 对外（API/前端）一律用平面坐标 ``(x=列, y=行)``；
- 模块内部数组按 ``[行, 列]`` 索引。
"""

from __future__ import annotations

from collections import deque

import numpy as np

MAX_BOX = 256          # 检查框宽、高各至多 256 像素
MAX_SPREAD = 16        # 试算 0..16 整数扩墨档
MARGIN = MAX_SPREAD    # 向外多读 16 像素
_INF = 1 << 30


class PreflightError(ValueError):
    """预检参数非法（API 层映射为 400）。"""


# ---------------------------------------------------------------- 参数校验
def validate_request(
    height: int,
    width: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    endpoints,
) -> tuple[int, int, int, int, tuple[int, int], tuple[int, int]]:
    """校验半开区间框 ``[x0,x1) × [y0,y1)`` 与两个端点，返回规整后的值。"""

    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise PreflightError(
            f"检查框越界或为空：({x0},{y0})-({x1},{y1})，图像 {width}x{height}"
        )
    bw, bh = x1 - x0, y1 - y0
    if bw > MAX_BOX or bh > MAX_BOX:
        raise PreflightError(
            f"检查框宽高各至多 {MAX_BOX} 像素，当前为 {bw}x{bh}"
        )

    if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
        raise PreflightError("endpoints 必须给出两个端点 [[x,y],[x,y]]")
    pts: list[tuple[int, int]] = []
    for i, p in enumerate(endpoints):
        try:
            px, py = int(p[0]), int(p[1])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise PreflightError(f"第 {i + 1} 个端点坐标非法：{p!r}") from exc
        if not (x0 <= px < x1 and y0 <= py < y1):
            raise PreflightError(
                f"第 {i + 1} 个端点 ({px},{py}) 不在检查框 "
                f"[{x0},{x1})×[{y0},{y1}) 内"
            )
        pts.append((px, py))
    if pts[0] == pts[1]:
        raise PreflightError("两个端点必须是不同的像素")
    return x0, y0, x1, y1, pts[0], pts[1]


# ---------------------------------------------------------------- 距离变换
def _left_sweep(row: np.ndarray) -> np.ndarray:
    """一维左向精确曼哈顿传播：g[c] = min_{j<=c} row[j] + (c-j)。"""

    idx = np.arange(row.shape[0])
    return idx + np.minimum.accumulate(row - idx)


def _right_sweep(row: np.ndarray) -> np.ndarray:
    """一维右向精确曼哈顿传播：g[c] = min_{j>=c} row[j] + (j-c)。"""

    idx = np.arange(row.shape[0])
    return np.minimum.accumulate((row + idx)[::-1])[::-1] - idx


def manhattan_black_distance(
    out: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> tuple[np.ndarray, int, int]:
    """计算“框 + 向外 16px”裁剪区内每个像素到最近黑点的曼哈顿距离。

    返回 ``(dist, crop_y0, crop_x0)``：``dist`` 为裁剪区等大的 int64 数组，
    ``crop_y0/crop_x0`` 是裁剪区在成稿中的左上角。区内无黑点时距离为大数。

    两遍 4 邻域 chamfer（权重均为 1）对曼哈顿距离是精确的；逐行配合一维
    累积最小传播，避免在 Python 里逐列循环。
    """

    h, w = out.shape
    cx0 = max(0, x0 - MARGIN)
    cy0 = max(0, y0 - MARGIN)
    cx1 = min(w, x1 + MARGIN)
    cy1 = min(h, y1 + MARGIN)
    crop = out[cy0:cy1, cx0:cx1]
    dist = np.where(crop == 0, 0, _INF).astype(np.int64)
    ch, cw = dist.shape

    # 前向：先取上方行，再做行内左扫
    for r in range(ch):
        if r > 0:
            dist[r] = np.minimum(dist[r], dist[r - 1] + 1)
        dist[r] = _left_sweep(dist[r])

    # 后向：先取下方行，再做行内右扫
    for r in range(ch - 1, -1, -1):
        if r + 1 < ch:
            dist[r] = np.minimum(dist[r], dist[r + 1] + 1)
        dist[r] = _right_sweep(dist[r])

    return dist, cy0, cx0


# ---------------------------------------------------------------- BFS 通道
def _bfs_level(
    inner_dist: np.ndarray,
    level: int,
    start_rc: tuple[int, int],
    goal_rc: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """在框内按档 ``level`` 做 4 邻域 BFS，返回一条最短的 [行,列] 路径。

    用一圈“关闭”的 padding 边把网格包起来，于是展开下标后四邻域
    ``idx±1 / idx±W`` 无需任何边界判断（padding 格永不可通行，也不会发生
    行首尾横绕）。起终点在当档被视为黑点或不可达时返回 None。
    """

    bh, bw = inner_dist.shape
    W = bw + 2
    walk = (inner_dist > level)
    padded = np.zeros((bh + 2, W), dtype=np.uint8)
    padded[1:bh + 1, 1:bw + 1] = walk
    open_flat = bytearray(padded.ravel())

    sr, sc = start_rc
    gr, gc = goal_rc
    start = (sr + 1) * W + (sc + 1)
    goal = (gr + 1) * W + (gc + 1)
    if not open_flat[start] or not open_flat[goal]:
        return None

    n = len(open_flat)
    parent = [-1] * n
    parent[start] = start
    q = deque([start])
    while q:
        idx = q.popleft()
        if idx == goal:
            path: list[tuple[int, int]] = []
            cur = goal
            while cur != start:
                path.append((cur // W - 1, cur % W - 1))
                cur = parent[cur]
            path.append((sr, sc))
            path.reverse()
            return path
        for nxt in (idx - W, idx + W, idx - 1, idx + 1):  # 上、下、左、右
            if parent[nxt] == -1 and open_flat[nxt]:
                parent[nxt] = idx
                q.append(nxt)
    return None


# ---------------------------------------------------------------- 主入口
def analyze(
    out: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    endpoints,
    max_spread: int = MAX_SPREAD,
) -> dict:
    """对当前成稿执行完整预检，返回 JSON 可序列化的结果字典。

    调用前参数已经过 :func:`validate_request` 校验也可；本函数会再校验一次。
    """

    h, w = out.shape
    x0, y0, x1, y1, (ax, ay), (bx, by) = validate_request(
        h, w, int(x0), int(y0), int(x1), int(y1), endpoints
    )
    bh, bw = y1 - y0, x1 - x0
    start_rc = (ay - y0, ax - x0)
    goal_rc = (by - y0, bx - x0)

    dist, cy0, cx0 = manhattan_black_distance(out, x0, y0, x1, y1)
    oy, ox = y0 - cy0, x0 - cx0
    inner = dist[oy:oy + bh, ox:ox + bw]

    endpoints_white = [
        bool(out[ay, ax] == 255),
        bool(out[by, bx] == 255),
    ]

    levels: list[dict] = []
    best_level: int | None = None
    best_path_rc: list[tuple[int, int]] = []

    for level in range(max_spread + 1):
        rc_path = _bfs_level(inner, level, start_rc, goal_rc)
        if rc_path is None:
            # 首个断档：本档及更高档全部不连通（可通行集合单调收缩）
            levels.append({"level": level, "connected": False, "shortest_length": None})
            for higher in range(level + 1, max_spread + 1):
                levels.append(
                    {"level": higher, "connected": False, "shortest_length": None}
                )
            break

        edge_len = len(rc_path) - 1
        levels.append(
            {"level": level, "connected": True, "shortest_length": edge_len}
        )
        best_level = level
        best_path_rc = rc_path

    path_xy = [[x0 + c, y0 + r] for r, c in best_path_rc] if best_level is not None else []

    if best_level is None:
        if not all(endpoints_white):
            reason = "endpoint_black"
            message = "端点在当前成稿中已是黑点，零档即无通道。"
        else:
            reason = "broken_at_zero"
            message = "零档（未扩墨）反白通道已经断开。"
    else:
        reason = "connected"
        message = (
            f"仍有通道的最高扩墨档为 {best_level} 像素"
            + ("（0–16 档全部连通）" if best_level == max_spread else "")
            + f"，最短通道 {len(path_xy)} 个像素。"
        )

    return {
        "max_level": best_level,
        "level0_connected": best_level is not None and best_level >= 0,
        "reason": reason,
        "message": message,
        "levels": levels,
        "path": path_xy,
        "endpoints": [[ax, ay], [bx, by]],
        "endpoints_white": endpoints_white,
        "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "width": bw, "height": bh},
        "read_region": {
            "x0": cx0, "y0": cy0,
            "x1": cx0 + dist.shape[1], "y1": cy0 + dist.shape[0],
        },
        "max_spread": max_spread,
    }
