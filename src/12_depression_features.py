"""
Step 2: 窪地単位の TDA 特徴（H0 マージツリー → 窪地ごとの体積・入れ子構造）

考え方（実験計画.md §4.2 (b)）:
  sublevel set filtration の H0 は「標高を上げていったとき、窪地（連結成分）が底で生まれ、
  溢れ出し標高（サドル）で隣の窪地に合流して消える」過程を記録する。
  各 H0 ペア (birth, death) は1つの窪地に対応し、その領域は
  「death 標高未満で、birth セルとつながっているセル」である。

実装:
  GUDHI の cofaces_of_persistence_pairs() → flood fill の代わりに、
  H0 の計算そのもの（標高順に Union-Find で合流）を自前で行い、合流の瞬間に
  消える側の窪地の量を記録する。flood fill が不要で、全階層の窪地を一度に得られる。
    面積   = セル数 × セル面積
    体積   = (セル数 × death − 標高の合計) × セル面積     … death まで水を張ったときの貯留量
    入れ子 = 合流してきた（有意な）子窪地の段数 + 1
  合流規則は elder rule（底が低い方が生き残る）で、GUDHI の H0 と同じペアになる。
  隣接は 8 近傍（GUDHI の CubicalComplex で top-dimensional cells を与えた場合と同じ）。
  → --check で GUDHI と照合できる

パッチと境界（実験計画.md §4.3・§4.4）:
  メッシュ 500m の周囲に 500m のバッファを付けた 1.5km 四方で計算し、
  底（birth セル）が中心メッシュ内にある窪地だけを採用する。
  パッチの外周と NODATA に接するセルは「外へ排水される」として、仮想の排水口（最古の成分）に
  つなぐ。こうしないと、パッチの縁で水が止まる偽の窪地が出る（00_check_storage.py と同じ考え方）。

採用する窪地: persistence（death − birth）≥ PERS_MIN [m]。それ未満は DEM のノイズとして
  窪地として数えない（領域は親の窪地に含まれる）。

メッシュ単位の列（output/depression_features.parquet）:
  dep_count          窪地の個数（全階層）
  dep_pers_max/mean  persistence の最大・平均 [m]
  dep_volume_sum     最上位窪地（排水口に直接合流するもの）の体積の合計 [m3]
                     … 入れ子を二重に数えないための定義。fill 由来の sink_volume_sum に近くなるはず
  dep_volume_max     全階層での体積の最大 [m3]
  dep_area_sum       最上位窪地の面積の合計 [m2]
  dep_area_max       全階層での面積の最大 [m2]
  dep_meandepth_max  平均水深（体積 ÷ 面積）の最大 [m]
  dep_width_min      幅（面積 ÷ 長さ）の最小 [m]。長さ＝外接矩形の対角
  dep_storage_depth  dep_volume_sum ÷ メッシュ面積 × 1000 [mm]
  dep_nest_depth     入れ子の段数の最大（子窪地なし＝1、窪地なし＝0）   ← fill では出せない
  dep_n_nested       子窪地を持つ窪地の個数                             ← fill では出せない
  dep_volume_ratio   体積最大の窪地について、その中の最下層窪地（葉）の体積の最大 ÷ 自身の体積
                     （子なし＝1、窪地なし＝0）                          ← fill では出せない
                     「浸水し始めるまでに埋まる量」が全体のどれだけかの目安

実行:
  python -u src/12_depression_features.py --check     … GUDHI との照合と速度確認（数分）
  python -u src/12_depression_features.py             … 全メッシュ
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from numba import njit

BASE_DIR  = Path(__file__).parent.parent
DEM_PATH  = BASE_DIR / "output" / "dem_utm.tif"
MESH_PATH = BASE_DIR / "output" / "mesh_500m.gpkg"
OUT_PATH  = BASE_DIR / "output" / "depression_features.parquet"
OUT_DEPS  = BASE_DIR / "output" / "depressions.parquet"   # 窪地ごとの表（解析・逆解析用）

NODATA   = -9999.0
BUF_M    = 500.0     # パッチのバッファ [m]
PERS_MIN = 0.1       # 窪地として数える最小 persistence [m]

# 窪地レコードの列（numba 側の配列の並び）
REC_COLS = ["birth_idx", "birth", "death", "n", "sumz", "rmin", "rmax", "cmin", "cmax",
            "level", "leaf_vol", "top"]


# ------------------------------------------------------------------ #
# H0 マージツリー（numba）
# ------------------------------------------------------------------ #

@njit(cache=True)
def _find(parent, x):
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:          # 経路圧縮
        nx = parent[x]
        parent[x] = r
        x = nx
    return r


@njit(cache=True)
def merge_tree(z, valid, pers_min, use_drain):
    """標高順に Union-Find で合流させ、消えた窪地（H0 ペア）を記録する

    z, valid : 2次元配列（float64, bool）
    use_drain: True なら外周・NODATA 隣接セルを排水口につなぐ。False は GUDHI 照合用
    返り値   : (m, len(REC_COLS)) の float64 配列。persistence < pers_min の窪地は含まない
               （use_drain=False のときは pers_min=0 で全ペアを返す想定）
    """
    nr, nc = z.shape
    N = nr * nc
    D = N                                  # 排水口ノード
    parent = np.full(N + 1, -1, np.int64)
    birth = np.empty(N + 1)
    bidx = np.full(N + 1, -1, np.int64)
    cnt = np.zeros(N + 1)
    sumz = np.zeros(N + 1)
    rmin = np.zeros(N + 1, np.int64); rmax = np.zeros(N + 1, np.int64)
    cmin = np.zeros(N + 1, np.int64); cmax = np.zeros(N + 1, np.int64)
    clev = np.zeros(N + 1, np.int64)       # 有意な子窪地の段数の最大
    leafv = np.zeros(N + 1)                # 有意な葉窪地の体積の最大
    parent[D] = D
    birth[D] = -np.inf

    flat = z.ravel()
    vflat = valid.ravel()
    order = np.argsort(flat, kind="mergesort")
    out = np.empty((N, 12))
    m = 0

    for oi in range(N):
        c = order[oi]
        if not vflat[c]:
            continue
        h = flat[c]
        r0 = c // nc
        c0 = c % nc
        parent[c] = c
        birth[c] = h
        bidx[c] = c
        cnt[c] = 1.0
        sumz[c] = h
        rmin[c] = r0; rmax[c] = r0; cmin[c] = c0; cmax[c] = c0

        # 合流相手: 処理済みの8近傍 + （境界なら）排水口
        for k in range(9):
            if k == 8:
                if not use_drain:
                    continue
                edge = r0 == 0 or c0 == 0 or r0 == nr - 1 or c0 == nc - 1
                if not edge:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            rr = r0 + dr; cc = c0 + dc
                            if 0 <= rr < nr and 0 <= cc < nc and not vflat[rr * nc + cc]:
                                edge = True
                if not edge:
                    continue
                nb = D
            else:
                dr = k // 3 - 1
                dc = k % 3 - 1
                if k >= 4:                 # (0,0) を飛ばして 8 近傍を列挙
                    dr = (k + 1) // 3 - 1
                    dc = (k + 1) % 3 - 1
                rr = r0 + dr; cc = c0 + dc
                if rr < 0 or cc < 0 or rr >= nr or cc >= nc:
                    continue
                nb = rr * nc + cc
                if parent[nb] == -1:
                    continue
            ra = _find(parent, c)
            rb = _find(parent, nb)
            if ra == rb:
                continue
            # elder rule: 排水口が最古、次に底が低い方、同じなら番号が小さい方
            if rb == D or (ra != D and (birth[rb] < birth[ra] or
                                        (birth[rb] == birth[ra] and bidx[rb] < bidx[ra]))):
                old, yng = rb, ra
            else:
                old, yng = ra, rb
            pers = h - birth[yng]
            if pers >= pers_min and pers > 0.0:
                lev = clev[yng] + 1
                vol = cnt[yng] * h - sumz[yng]
                lv = leafv[yng] if clev[yng] > 0 else vol
                out[m, 0] = bidx[yng]; out[m, 1] = birth[yng]; out[m, 2] = h
                out[m, 3] = cnt[yng]; out[m, 4] = sumz[yng]
                out[m, 5] = rmin[yng]; out[m, 6] = rmax[yng]
                out[m, 7] = cmin[yng]; out[m, 8] = cmax[yng]
                out[m, 9] = lev; out[m, 10] = lv
                out[m, 11] = 1.0 if old == D else 0.0
                m += 1
                if lev > clev[old]:
                    clev[old] = lev
                if lv > leafv[old]:
                    leafv[old] = lv
            elif not use_drain:
                # 照合用：persistence 0 のペアも GUDHI は出さないので記録しない
                pass
            # 統計を生き残る側へ
            cnt[old] += cnt[yng]; sumz[old] += sumz[yng]
            if rmin[yng] < rmin[old]: rmin[old] = rmin[yng]
            if rmax[yng] > rmax[old]: rmax[old] = rmax[yng]
            if cmin[yng] < cmin[old]: cmin[old] = cmin[yng]
            if cmax[yng] > cmax[old]: cmax[old] = cmax[yng]
            parent[yng] = old
    return out[:m]


# ------------------------------------------------------------------ #
# パッチ
# ------------------------------------------------------------------ #

def mesh_windows(mesh: gpd.GeoDataFrame, transform, shape, cell: float):
    """各メッシュについて、バッファ付きパッチと中心メッシュの行・列範囲（DEM 全体座標）を返す"""
    inv = ~transform
    buf = int(round(BUF_M / cell))
    H, W = shape
    out = []
    for mid, g in zip(mesh["mesh_id"], mesh.geometry):
        x0, y0, x1, y1 = g.bounds
        c0, r0 = inv * (x0, y1)
        c1, r1 = inv * (x1, y0)
        r0, r1, c0, c1 = int(round(r0)), int(round(r1)), int(round(c0)), int(round(c1))
        pr0, pr1 = max(r0 - buf, 0), min(r1 + buf, H)
        pc0, pc1 = max(c0 - buf, 0), min(c1 + buf, W)
        out.append((mid, (pr0, pr1, pc0, pc1), (r0, r1, c0, c1)))
    return out


def depressions_in_patch(dem: np.ndarray, win, center, cell: float) -> pd.DataFrame:
    """パッチで窪地を計算し、底が中心メッシュ内にあるものを返す"""
    pr0, pr1, pc0, pc1 = win
    r0, r1, c0, c1 = center
    z = dem[pr0:pr1, pc0:pc1].astype(np.float64)
    valid = z != NODATA
    if valid.sum() < 50:
        return pd.DataFrame(columns=REC_COLS)
    rec = merge_tree(z, valid, PERS_MIN, True)
    d = pd.DataFrame(rec, columns=REC_COLS)
    nc = z.shape[1]
    br = d["birth_idx"].astype(np.int64) // nc + pr0
    bc = d["birth_idx"].astype(np.int64) % nc + pc0
    keep = (br >= r0) & (br < r1) & (bc >= c0) & (bc < c1)
    d = d[keep].copy()
    a = cell * cell
    d["pers"] = d["death"] - d["birth"]
    d["area"] = d["n"] * a
    d["volume"] = (d["n"] * d["death"] - d["sumz"]) * a
    d["mean_depth"] = d["volume"] / d["area"]
    d["length"] = np.hypot(d["rmax"] - d["rmin"] + 1, d["cmax"] - d["cmin"] + 1) * cell
    d["width"] = d["area"] / d["length"]
    d["leaf_vol"] = d["leaf_vol"] * a
    return d


def aggregate(d: pd.DataFrame, mesh_area: float) -> dict:
    if len(d) == 0:
        return {"dep_count": 0, "dep_pers_max": 0.0, "dep_pers_mean": 0.0,
                "dep_volume_sum": 0.0, "dep_volume_max": 0.0, "dep_area_sum": 0.0,
                "dep_area_max": 0.0, "dep_meandepth_max": 0.0, "dep_width_min": 0.0,
                "dep_storage_depth": 0.0, "dep_nest_depth": 0, "dep_n_nested": 0,
                "dep_volume_ratio": 0.0}
    top = d[d["top"] == 1]
    big = d.loc[d["volume"].idxmax()]
    vsum = float(top["volume"].sum())
    return {
        "dep_count": len(d),
        "dep_pers_max": float(d["pers"].max()),
        "dep_pers_mean": float(d["pers"].mean()),
        "dep_volume_sum": vsum,
        "dep_volume_max": float(d["volume"].max()),
        "dep_area_sum": float(top["area"].sum()),
        "dep_area_max": float(d["area"].max()),
        "dep_meandepth_max": float(d["mean_depth"].max()),
        "dep_width_min": float(d["width"].min()),
        "dep_storage_depth": vsum / mesh_area * 1000.0,
        "dep_nest_depth": int(d["level"].max()),
        "dep_n_nested": int((d["level"] >= 2).sum()),
        "dep_volume_ratio": float(big["leaf_vol"] / big["volume"]) if big["volume"] > 0 else 0.0,
    }


# ------------------------------------------------------------------ #
# GUDHI との照合
# ------------------------------------------------------------------ #

def check_against_gudhi(dem: np.ndarray, wins, n: int = 20, seed: int = 0) -> None:
    import gudhi
    rng = np.random.default_rng(seed)
    print(f"[照合] 排水口なし・全ペアで、ランダムな {n} パッチの H0 を GUDHI と比較")
    n_ok = 0
    for i in rng.choice(len(wins), n, replace=False):
        pr0, pr1, pc0, pc1 = wins[i][1]
        z = dem[pr0:pr1, pc0:pc1].astype(np.float64)
        if (z == NODATA).any():
            continue                       # 照合は NODATA のないパッチで行う
        # (birth, death) の組を多重集合として比べる
        mine = merge_tree(z, np.ones_like(z, bool), 0.0, False)[:, 1:3]
        cc = gudhi.CubicalComplex(top_dimensional_cells=z)
        cc.compute_persistence()
        g = np.array(cc.persistence_intervals_in_dimension(0))
        g = g[np.isfinite(g[:, 1]) & (g[:, 1] > g[:, 0])]
        key = lambda a: sorted(map(tuple, np.round(a, 4)))  # noqa: E731
        same = key(mine) == key(g)
        n_ok += same
        print(f"  mesh {wins[i][0]}: 自前 {len(mine)} ペア / GUDHI {len(g)} ペア → "
              f"{'一致' if same else '不一致'}")
    print(f"[照合] 一致 {n_ok} パッチ")


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="GUDHI との照合と速度確認のみ")
    args = ap.parse_args()

    t0 = time.time()
    print("DEM 読み込み中...", flush=True)
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1)
        transform, cell = src.transform, src.res[0]
    mesh = gpd.read_file(MESH_PATH)
    wins = mesh_windows(mesh, transform, dem.shape, cell)
    mesh_area = 500.0 * 500.0
    print(f"  {dem.shape}, セル {cell:.2f} m, メッシュ {len(wins)}（{time.time() - t0:.0f}秒）",
          flush=True)

    if args.check:
        check_against_gudhi(dem, wins)
        t1 = time.time()
        rng = np.random.default_rng(1)
        sample = rng.choice(len(wins), 50, replace=False)
        rows = [aggregate(depressions_in_patch(dem, wins[i][1], wins[i][2], cell), mesh_area)
                for i in sample]
        dt = (time.time() - t1) / len(sample)
        print(f"[速度] 1パッチ {dt * 1000:.0f} ms → 全 {len(wins)} メッシュで約 "
              f"{dt * len(wins) / 60:.0f} 分")
        print(pd.DataFrame(rows).describe().round(3).T.to_string())
        return

    rows, deps = [], []
    for k, (mid, win, center) in enumerate(wins):
        d = depressions_in_patch(dem, win, center, cell)
        rows.append({"mesh_id": mid, **aggregate(d, mesh_area)})
        if len(d):
            deps.append(d.assign(mesh_id=mid))
        if (k + 1) % 1000 == 0:
            el = time.time() - t0
            print(f"  {k + 1}/{len(wins)}  経過 {el / 60:.1f} 分", flush=True)

    feat = pd.DataFrame(rows)
    feat.to_parquet(OUT_PATH, index=False)
    pd.concat(deps, ignore_index=True).to_parquet(OUT_DEPS, index=False)
    print(feat.describe().round(3).T.to_string())
    print(f"保存: {OUT_PATH}\n保存: {OUT_DEPS}\n合計 {(time.time() - t0) / 60:.1f} 分")


if __name__ == "__main__":
    main()
