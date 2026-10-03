"""
上流域積算雨量の算出（雨量特徴の物理的な作り直し）

背景:
  現行の rain_sum24h は「そのメッシュの真上に降った雨」だが、
  外水氾濫でメッシュXを浸水させる水は X の上流に降った雨である。
  実測では浸水メッシュの方が局所雨量が少ない（2020: 334mm vs 394mm）。
  雨は山に降り、洪水は平野で起きるため、局所雨量は「山か平野か」の
  指標として働いており、標高と冗長かつイベント間で不安定だった。

  そこで D8 流向に沿って雨量を積算し、
  「いまここに何 m3 の水が到達しつつあるか」を表す特徴量を作る。

算出する特徴量（イベント別）:
  catchment_area_m2     … 上流集水面積 [m2]
  rain_up_mean24h       … 上流域の平均24h雨量 [mm]
  rain_up_mean1h        … 上流域の平均最大1h雨量 [mm]
  inflow_volume_m3      … 上流積算流入量 [m3] = Σ(上流各セルの雨量 × セル面積)
  inflow_depth_mm       … 流入量をメッシュ面積で割った値 [mm]
  inflow_per_storage    … inflow_volume / 窪地貯留容量（無次元）
                          ← 「この地形にこの雨量で洪水が起きるか」に直接対応

前提:
  output/dem_utm.tif           （00_check_storage.py が生成）
  output/sink_features.parquet （同上）
  output/rainfall_features.parquet
  output/mesh_500m.gpkg

NODATA対処:
  00_check_storage.py と同じく NODATA に十分低い標高を与え、
  常に排水される外部として扱う（境界アーティファクト回避）。

実行:
  python -u src/07b_upstream_rain.py

出力:
  output/rain_upstream_features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features as rfeatures
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder

BASE_DIR     = Path(__file__).parent.parent
DEM_UTM_PATH = BASE_DIR / "output" / "dem_utm.tif"
# 流向計算用の粗いDEM。
# 5.58m・1.58億セルでは resolve_flats が float64 中間配列（1.2GB×複数）で
# メモリ不足になる。集水域の追跡は流域スケールの処理であり、
# HAND・流域解析は世界的にSRTM 30m級で行われているため約17mで十分。
# 微地形が必要な窪地・TDA特徴は 5.58m のまま別途計算する。
DEM_ROUTE_PATH = BASE_DIR / "output" / "dem_utm_route.tif"
ROUTE_FACTOR   = 3          # 5.58m × 3 ≒ 16.7m
MESH_PATH    = BASE_DIR / "output" / "mesh_500m.gpkg"
RAIN_PATH    = BASE_DIR / "output" / "rainfall_features.parquet"
SINK_PATH    = BASE_DIR / "output" / "sink_features.parquet"
OUT_PARQ     = BASE_DIR / "output" / "rain_upstream_features.parquet"

NODATA = -9999.0

# 上流積算する雨量列（多いと計算時間が伸びるので主要2つに絞る）
RAIN_COLS = ["rain_sum24h", "rain_max1h"]


# ------------------------------------------------------------------ #
# 1. 流向の計算（NODATA を外部排水口として扱う）
# ------------------------------------------------------------------ #

def make_routing_dem(src_path: Path, dst_path: Path, factor: int) -> None:
    """流向計算用に粗いDEMを作る（キャッシュあり）"""
    if dst_path.exists():
        with rasterio.open(dst_path) as r:
            print(f"      流向用DEMを再利用: {dst_path.name} "
                  f"({r.width}x{r.height} px, res={r.res[0]:.2f} m)", flush=True)
        return

    with rasterio.open(src_path) as src:
        h, w = src.height // factor, src.width // factor
        t = src.transform * Affine.scale(factor, factor)
        prof = src.profile.copy()
        prof.update({"height": h, "width": w, "transform": t,
                     "dtype": "float32", "nodata": NODATA})
        dst_arr = np.full((h, w), NODATA, dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dst_arr,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=t, dst_crs=src.crs,
                  src_nodata=NODATA, dst_nodata=NODATA,
                  resampling=Resampling.average)
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(dst_arr, 1)
    print(f"      流向用DEM作成: {w}x{h} px "
          f"({w * h:,} セル, 元の 1/{factor**2})", flush=True)


def compute_fdir(dem_path: Path):
    print("[1/5] 流向（D8）を計算中...", flush=True)
    grid = Grid.from_raster(str(dem_path))
    dem  = grid.read_raster(str(dem_path))

    dem_arr = np.asarray(dem, dtype=np.float32)
    valid   = dem_arr != NODATA

    sink_floor = float(dem_arr[valid].min()) - 1000.0
    work = np.where(valid, dem_arr, sink_floor).astype(np.float32)

    vf = dem.viewfinder
    work_vf = ViewFinder(affine=vf.affine, shape=vf.shape, crs=vf.crs,
                         nodata=np.float32(-1e9),
                         mask=np.ones(vf.shape, dtype=bool))
    work_raster = Raster(work, viewfinder=work_vf)
    del dem, dem_arr, work

    pit_filled = grid.fill_pits(work_raster)
    del work_raster
    flooded    = grid.fill_depressions(pit_filled)
    del pit_filled
    inflated   = grid.resolve_flats(flooded)
    del flooded
    fdir       = grid.flowdir(inflated)
    del inflated
    print("      流向計算完了", flush=True)

    return grid, fdir, work_vf, valid


# ------------------------------------------------------------------ #
# 2. メッシュ値をDEMグリッドへラスタ化
# ------------------------------------------------------------------ #

def rasterize_mesh_values(mesh: gpd.GeoDataFrame, values: pd.Series,
                          shape, transform) -> np.ndarray:
    """メッシュポリゴンの値をDEMグリッドに焼き込む"""
    shapes = ((geom, float(v)) for geom, v in zip(mesh.geometry, values))
    return rfeatures.rasterize(shapes, out_shape=shape, transform=transform,
                               fill=0.0, dtype="float32", all_touched=False)


def rasterize_mesh_index(mesh: gpd.GeoDataFrame, shape, transform) -> np.ndarray:
    """メッシュの行番号+1 を焼き込む（0=メッシュ外）。高速ゾーン統計用"""
    shapes = ((geom, i + 1) for i, geom in enumerate(mesh.geometry))
    return rfeatures.rasterize(shapes, out_shape=shape, transform=transform,
                               fill=0, dtype="int32", all_touched=False)


# ------------------------------------------------------------------ #
# 3. bincount による高速ゾーン統計
# ------------------------------------------------------------------ #

def zonal_mean_fast(idx: np.ndarray, arr: np.ndarray,
                    valid: np.ndarray, n_mesh: int) -> np.ndarray:
    """メッシュ別平均。rasterstats より2桁速い"""
    m = (idx > 0) & valid & np.isfinite(arr)
    flat_idx = idx[m].astype(np.int64)
    flat_val = arr[m].astype(np.float64)
    s = np.bincount(flat_idx, weights=flat_val, minlength=n_mesh + 1)[1:]
    c = np.bincount(flat_idx, minlength=n_mesh + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / c, np.nan)
    return out


def outlet_positions(idx: np.ndarray, acc: np.ndarray, n_mesh: int):
    """各メッシュの流出点（集水面積が最大のセル）の位置を返す

    メッシュ内のほとんどのセルは集水面積が小さく、河道セルだけが巨大な値を持つ。
    平均を取ると希釈されて消えるため（実測: 流域2,513km2に対しメッシュ平均の
    最大が9.9km2にしかならなかった）、最下流点の値を代表値として使う。
    """
    from scipy import ndimage
    labels = np.arange(1, n_mesh + 1)
    work = np.where(idx > 0, acc, -np.inf)
    pos = ndimage.maximum_position(work, labels=idx, index=labels)
    rows = np.array([p[0] if p is not None else 0 for p in pos], dtype=np.int64)
    cols = np.array([p[1] if p is not None else 0 for p in pos], dtype=np.int64)
    return rows, cols


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main() -> None:
    if not DEM_UTM_PATH.exists():
        raise SystemExit(f"{DEM_UTM_PATH} がありません。先に 00_check_storage.py を実行してください")

    print("[0/5] 流向計算用の粗いDEMを準備...", flush=True)
    make_routing_dem(DEM_UTM_PATH, DEM_ROUTE_PATH, ROUTE_FACTOR)

    with rasterio.open(DEM_ROUTE_PATH) as src:
        shape     = (src.height, src.width)
        transform = src.transform
        cellsize  = src.res[0]
    cell_area = cellsize ** 2
    print(f"DEM(routing): {shape[1]}x{shape[0]} px, res={cellsize:.2f} m", flush=True)

    grid, fdir, work_vf, valid = compute_fdir(DEM_ROUTE_PATH)

    print("[2/5] メッシュをラスタ化中...", flush=True)
    mesh = gpd.read_file(MESH_PATH)
    mesh_ids = mesh["mesh_id"].to_numpy()
    n_mesh   = len(mesh)
    idx = rasterize_mesh_index(mesh, shape, transform)
    print(f"      メッシュ {n_mesh:,} 個 / 被覆セル {int((idx > 0).sum()):,}", flush=True)

    # 集水面積（重みなしの流域累積）
    print("[3/5] 流域累積（集水面積）を計算中...", flush=True)
    acc_cells = np.asarray(grid.accumulation(fdir), dtype=np.float32)
    print(f"      最大集水セル数: {acc_cells.max():,.0f} "
          f"({acc_cells.max() * cell_area / 1e6:,.0f} km2)", flush=True)

    # 各メッシュの流出点（最下流セル）を特定する
    print("      流出点を特定中...", flush=True)
    orow, ocol = outlet_positions(idx, acc_cells, n_mesh)
    outlet_acc = acc_cells[orow, ocol]
    print(f"      集水面積: 最大 {outlet_acc.max() * cell_area / 1e6:,.0f} km2 / "
          f"中央 {np.median(outlet_acc) * cell_area / 1e6:.2f} km2", flush=True)

    rain = pd.read_parquet(RAIN_PATH)
    sink = pd.read_parquet(SINK_PATH)

    results = []
    for ev in sorted(rain["event"].unique()):
        print(f"[4/5] イベント {ev} の上流積算雨量を計算中...", flush=True)
        rv = rain[rain["event"] == ev].set_index("mesh_id").reindex(mesh_ids)

        row = pd.DataFrame({"mesh_id": mesh_ids, "event": ev})
        row["catchment_area_m2"] = outlet_acc * cell_area

        for col in RAIN_COLS:
            rain_ras = rasterize_mesh_values(mesh, rv[col].fillna(0.0), shape, transform)
            w = Raster(rain_ras, viewfinder=work_vf)
            acc_rain = np.asarray(grid.accumulation(fdir, weights=w), dtype=np.float32)

            suffix = "24h" if col.endswith("24h") else "1h"
            # 流出点における値を代表値とする
            out_rain = acc_rain[orow, ocol]
            # 流入量 [m3] = Σ(上流各セルの雨量[mm]/1000 × セル面積[m2])
            row[f"inflow_volume_m3_{suffix}"] = out_rain * cell_area / 1000.0
            # 上流域の平均雨量 [mm] = 積算雨量 / 上流セル数
            with np.errstate(invalid="ignore", divide="ignore"):
                row[f"rain_up_mean{suffix}"] = np.where(outlet_acc > 0,
                                                        out_rain / outlet_acc, np.nan)
            print(f"      {col}: 上流平均 "
                  f"{np.nanmedian(row[f'rain_up_mean{suffix}']):.0f} mm / "
                  f"流入量中央 {np.nanmedian(row[f'inflow_volume_m3_{suffix}']):,.0f} m3",
                  flush=True)
            del rain_ras, w, acc_rain, out_rain

        results.append(row)

    print("[5/5] 派生特徴量を算出・保存...", flush=True)
    out = pd.concat(results, ignore_index=True)
    out = out.merge(sink[["mesh_id", "sink_volume_sum"]], on="mesh_id", how="left")

    mesh_area = float(mesh.geometry.area.median())
    out["inflow_depth_mm"] = out["inflow_volume_m3_24h"] / mesh_area * 1000.0
    # 「上流から来る水が、この地形の貯留容量の何倍か」
    out["inflow_per_storage"] = out["inflow_volume_m3_24h"] / (out["sink_volume_sum"] + 1.0)
    out = out.drop(columns=["sink_volume_sum"])

    print(out.groupby("event").describe().T.round(2).to_string()[:2000], flush=True)
    out.to_parquet(OUT_PARQ, index=False)
    print(f"\n保存完了: {OUT_PARQ}", flush=True)


if __name__ == "__main__":
    main()
