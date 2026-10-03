"""
Step 1: 従来地形特徴の拡張（Model A ベースライン強化）

背景:
  既存の Model A は15次元しかなく、洪水感受性予測で標準的に使われる
  HAND・曲率・河川距離・窪地指標が欠けている。
  この状態で「TDAで ROC-AUC +0.069」と主張しても、
  「ベースラインが薄かっただけ」と区別できない（実験計画.md 2節）。

算出する特徴量:
  hand_mean / hand_min          … 最近傍河道からの比高 [m]
  curv_plan_mean / _std         … 平面曲率（等高線の曲がり。凹地形=水が集まる）
  curv_prof_mean / _std         … 断面曲率（斜面方向の曲がり。加速・減速）
  dist_stream_mean / _min       … 河道までの距離 [m]
  ※ sink_* は 00_check_storage.py が output/sink_features.parquet に出力済み

前提:
  output/dem_utm.tif が存在すること（00_check_storage.py が生成・キャッシュ）
  無い場合は 00_check_storage.py を先に実行する

メモリ対策:
  1.58億セルのため、全量を float32 で持つと1配列632MB。
  曲率は行ブロック＋ハロー2行で逐次計算し、結果をラスタへ直接書き出す。

出力:
  output/terrain_extra.parquet … mesh_id + 上記特徴量
  output/hand.tif / curv_plan.tif / curv_prof.tif / dist_stream.tif（確認用）

実行:
  python -u src/06b_terrain_extra.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from pysheds.grid import Grid
from rasterstats import zonal_stats
from scipy import ndimage

BASE_DIR     = Path(__file__).parent.parent
DEM_UTM_PATH = BASE_DIR / "output" / "dem_utm.tif"
MESH_PATH    = BASE_DIR / "output" / "mesh_500m.gpkg"

OUT_HAND     = BASE_DIR / "output" / "hand.tif"
OUT_CPLAN    = BASE_DIR / "output" / "curv_plan.tif"
OUT_CPROF    = BASE_DIR / "output" / "curv_prof.tif"
OUT_DIST     = BASE_DIR / "output" / "dist_stream.tif"
OUT_PARQ     = BASE_DIR / "output" / "terrain_extra.parquet"

NODATA       = -9999.0

# 河道とみなす流域累積のしきい値 [m2]。1 km2 は日本の中小河川で標準的な値
STREAM_AREA_M2 = 1_000_000.0

BLOCK_ROWS   = 1024      # 曲率計算のブロック高
HALO         = 2         # 2階微分に必要なハロー行数


# ------------------------------------------------------------------ #
# 共通ユーティリティ
# ------------------------------------------------------------------ #

def save_like(arr: np.ndarray, path: Path, profile: dict) -> None:
    prof = profile.copy()
    prof.update({"dtype": "float32", "count": 1, "nodata": NODATA})
    out = np.where(np.isfinite(arr), arr, NODATA).astype(np.float32)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def zonal(raster_path: Path, mesh: gpd.GeoDataFrame,
          stats: list, prefix: str) -> pd.DataFrame:
    res = zonal_stats(mesh, str(raster_path), stats=stats,
                      nodata=NODATA, all_touched=False)
    df = pd.DataFrame(res, index=mesh.index)
    return df.rename(columns={s: f"{prefix}_{s}" for s in stats})


# ------------------------------------------------------------------ #
# 1. 流向・流域累積・河道マスク・HAND
# ------------------------------------------------------------------ #

def compute_hand(dem_path: Path, cellsize: float):
    """D8流向 → 流域累積 → 河道マスク → HAND を算出し保存"""
    print("[1/4] pysheds: 流向・流域累積・HAND ...", flush=True)
    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))

    pit_filled = grid.fill_pits(dem)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)
    fdir       = grid.flowdir(inflated)
    acc        = grid.accumulation(fdir)

    # 河道マスク：流域累積面積 > STREAM_AREA_M2
    acc_thresh = STREAM_AREA_M2 / (cellsize ** 2)
    stream     = np.asarray(acc) > acc_thresh
    print(f"      河道セル: {int(stream.sum()):,} "
          f"(しきい値 {acc_thresh:,.0f} セル = {STREAM_AREA_M2/1e6:.1f} km2)", flush=True)

    # HAND: 各セルから流下して最初に到達する河道セルとの標高差
    print("      HAND 計算中...", flush=True)
    hand = grid.compute_hand(fdir, inflated, stream)
    hand_arr = np.asarray(hand, dtype=np.float32)

    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        dem_arr = src.read(1)
    invalid = (dem_arr == NODATA)
    hand_arr[invalid] = np.nan

    valid = np.isfinite(hand_arr)
    print(f"      HAND range: {hand_arr[valid].min():.2f} ~ "
          f"{hand_arr[valid].max():.2f} m", flush=True)
    save_like(hand_arr, OUT_HAND, profile)
    del hand, hand_arr

    return stream, profile


# ------------------------------------------------------------------ #
# 2. 河道からの距離
# ------------------------------------------------------------------ #

def compute_dist_stream(stream: np.ndarray, cellsize: float,
                        profile: dict, dem_path: Path) -> None:
    print("[2/4] 河道からの距離（EDT）...", flush=True)
    # 河道セル以外について、最近傍の河道セルまでの距離
    dist = ndimage.distance_transform_edt(~stream, sampling=cellsize)
    dist = dist.astype(np.float32)

    with rasterio.open(dem_path) as src:
        dem_arr = src.read(1)
    dist[dem_arr == NODATA] = np.nan

    valid = np.isfinite(dist)
    print(f"      距離 range: {dist[valid].min():.0f} ~ {dist[valid].max():.0f} m",
          flush=True)
    save_like(dist, OUT_DIST, profile)


# ------------------------------------------------------------------ #
# 3. 曲率（ブロック処理）
# ------------------------------------------------------------------ #

def compute_curvature(dem_path: Path, cellsize: float) -> None:
    """
    Zevenbergen & Thorne (1987) の定義で平面曲率・断面曲率を算出。

      p = dz/dx, q = dz/dy, r = d2z/dx2, s = d2z/dxdy, t = d2z/dy2
      平面曲率 Kc = -(q^2 r - 2 p q s + p^2 t) / (p^2+q^2)^1.5
      断面曲率 Kp = -(p^2 r + 2 p q s + q^2 t) / ((p^2+q^2)(1+p^2+q^2)^1.5)

    負値 = 凹（水が集まる）、正値 = 凸（水が散る）。
    全量を一度に持つとメモリが足りないため行ブロック＋ハローで処理する。
    """
    print("[3/4] 曲率（平面・断面）をブロック処理で計算...", flush=True)
    with rasterio.open(dem_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

        prof = profile.copy()
        prof.update({"dtype": "float32", "count": 1, "nodata": NODATA})

        with rasterio.open(OUT_CPLAN, "w", **prof) as dst_plan, \
             rasterio.open(OUT_CPROF, "w", **prof) as dst_prof:

            n_block = (height + BLOCK_ROWS - 1) // BLOCK_ROWS
            for bi, row0 in enumerate(range(0, height, BLOCK_ROWS)):
                r_start = max(0, row0 - HALO)
                r_stop  = min(height, row0 + BLOCK_ROWS + HALO)
                win     = Window(0, r_start, width, r_stop - r_start)

                block = src.read(1, window=win).astype(np.float32)
                block[block == NODATA] = np.nan

                q, p = np.gradient(block, cellsize, cellsize)   # q=dz/dy, p=dz/dx
                _, r = np.gradient(p, cellsize, cellsize)       # r=d2z/dx2
                t, s = np.gradient(q, cellsize, cellsize)       # t=d2z/dy2, s=d2z/dydx

                p2, q2 = p * p, q * q
                denom_c = np.power(p2 + q2, 1.5)
                denom_p = (p2 + q2) * np.power(1.0 + p2 + q2, 1.5)

                with np.errstate(invalid="ignore", divide="ignore"):
                    k_plan = -(q2 * r - 2.0 * p * q * s + p2 * t) / denom_c
                    k_prof = -(p2 * r + 2.0 * p * q * s + q2 * t) / denom_p

                # 平坦地（p=q=0）は曲率0とする
                flat = (p2 + q2) < 1e-12
                k_plan[flat] = 0.0
                k_prof[flat] = 0.0

                # ハローを除いて書き出し
                off_top = row0 - r_start
                n_rows  = min(BLOCK_ROWS, height - row0)
                out_win = Window(0, row0, width, n_rows)
                for arr, dst in ((k_plan, dst_plan), (k_prof, dst_prof)):
                    chunk = arr[off_top:off_top + n_rows]
                    dst.write(np.where(np.isfinite(chunk), chunk,
                                       NODATA).astype(np.float32),
                              1, window=out_win)

                if (bi + 1) % 4 == 0 or bi + 1 == n_block:
                    print(f"      block {bi + 1}/{n_block}", flush=True)


# ------------------------------------------------------------------ #
# 4. メッシュ集計
# ------------------------------------------------------------------ #

def aggregate(mesh: gpd.GeoDataFrame) -> pd.DataFrame:
    print("[4/4] ゾーン統計 → 500mメッシュ...", flush=True)
    parts = [
        zonal(OUT_HAND,  mesh, ["mean", "min"],  "hand"),
        zonal(OUT_DIST,  mesh, ["mean", "min"],  "dist_stream"),
        zonal(OUT_CPLAN, mesh, ["mean", "std"],  "curv_plan"),
        zonal(OUT_CPROF, mesh, ["mean", "std"],  "curv_prof"),
    ]
    out = pd.concat(parts, axis=1)
    out.index.name = "mesh_id"
    print(out.describe().round(3).to_string(), flush=True)
    return out.reset_index()


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main() -> None:
    if not DEM_UTM_PATH.exists():
        raise SystemExit(
            f"再投影済みDEMが見つかりません: {DEM_UTM_PATH}\n"
            "先に src/00_check_storage.py を実行してください（数時間かかります）"
        )

    with rasterio.open(DEM_UTM_PATH) as src:
        cellsize = src.res[0]
        print(f"DEM: {src.width}x{src.height} px, res={cellsize:.2f} m "
              f"({src.width * src.height:,} セル)", flush=True)

    stream, profile = compute_hand(DEM_UTM_PATH, cellsize)
    compute_dist_stream(stream, cellsize, profile, DEM_UTM_PATH)
    del stream

    compute_curvature(DEM_UTM_PATH, cellsize)

    mesh = gpd.read_file(MESH_PATH).set_index("mesh_id")
    feats = aggregate(mesh)
    feats.to_parquet(OUT_PARQ, index=False)
    print(f"\n保存完了: {OUT_PARQ}", flush=True)


if __name__ == "__main__":
    main()
