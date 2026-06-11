"""
Sentinel-2 NDVI / MNDWI 取得 → 500mメッシュ平均化

データソース: Microsoft Planetary Computer (無料・認証不要)
使用バンド:
  B03 (Green, 10m) : MNDWI 分子
  B04 (Red,   10m) : NDVI 分母
  B08 (NIR,   10m) : NDVI / MNDWI
  B11 (SWIR,  20m) : MNDWI 分母

NDVI  = (B08 - B04) / (B08 + B04)  … 植生量 (高いほど植生豊富)
MNDWI = (B03 - B11) / (B03 + B11)  … 水域検出 (高いほど水域)

使用シーン（豪雨前の晴天日）:
  2020年: 2020-06-08 (cloud 12.3%, T52SFB)
  2023年: 2023-06-15 (cloud  8.7%, T52SFB)

出力:
  output/sentinel2_features.parquet
    列: mesh_id, event, ndvi_mean, ndvi_min, mndwi_mean, mndwi_max
"""

import os
# GDALのHTTPリトライ設定（ネットワークエラー対策）
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"

import numpy as np
import pandas as pd
import geopandas as gpd
import pystac_client
import planetary_computer
import stackstac
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from rasterstats import zonal_stats

BASE_DIR  = Path(__file__).parent.parent
MESH_PATH = BASE_DIR / "output" / "mesh_500m.gpkg"
OUT_PARQ  = BASE_DIR / "output" / "sentinel2_features.parquet"
OUT_DIR   = BASE_DIR / "output"

BBOX = [130.25, 32.92, 131.00, 33.50]   # [lon_min, lat_min, lon_max, lat_max]
DST_CRS = "EPSG:32653"                   # UTM53N（メッシュと同じ）

TARGET_SCENES = {
    "2020": {"date_range": "2020-05-01/2020-07-08", "max_cloud": 20},
    "2023": {"date_range": "2023-05-01/2023-07-09", "max_cloud": 20},
}


# ------------------------------------------------------------------ #
# 1. Planetary Computer でシーン検索
# ------------------------------------------------------------------ #

def search_best_scene(catalog, event_key: str) -> list:
    """研究エリアをカバーする最良日の全タイルを返す（複数タイルをモザイク対応）"""
    cfg = TARGET_SCENES[event_key]
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=BBOX,
        datetime=cfg["date_range"],
        query={"eo:cloud_cover": {"lt": cfg["max_cloud"]}},
        max_items=100,
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"{event_key}: 対象シーンが見つかりません")

    # 日付ごとにグループ化し、タイル合計雲量が最小の日を選ぶ
    from collections import defaultdict
    by_date = defaultdict(list)
    for it in items:
        by_date[it.datetime.date()].append(it)

    best_date = min(by_date, key=lambda d: sum(
        i.properties["eo:cloud_cover"] for i in by_date[d]))
    best_items = by_date[best_date]

    print(f"  選択日: {best_date}  タイル数: {len(best_items)}")
    for it in best_items:
        print(f"    {it.id[-30:]}  cloud={it.properties['eo:cloud_cover']:.1f}%")
    return best_items


# ------------------------------------------------------------------ #
# 2. stackstac でバンド読み込み → NDVI/MNDWI 計算
# ------------------------------------------------------------------ #

def compute_indices(items: list):
    # 複数タイルを一度に渡すと stackstac が自動モザイク
    stack = stackstac.stack(
        items,
        assets=["B03", "B04", "B08", "B11"],
        resolution=10,
        epsg=32653,
        bounds_latlon=BBOX,
        rescale=False,
    )

    print("    バンドデータをダウンロード中...")
    # 複数シーンの場合は time 軸を最大値でモザイク（雲マスクされた NaN を除去）
    da = stack.max(dim="time")
    arr = da.compute().values   # shape: (4, H, W)

    # x/y座標からAffine transformを計算
    x = da.x.values.astype(float)
    y = da.y.values.astype(float)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])   # 負値（上から下へ）
    # ピクセル左上コーナー
    west  = x[0]  - dx / 2
    north = y[0]  - dy / 2    # dy < 0 なので -dy/2 > 0
    transform = from_origin(west, north, dx, -dy)

    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "crs": DST_CRS, "transform": transform,
        "width": len(x), "height": len(y),
        "nodata": -9999.0,
    }

    bands = {name: arr[i].astype(np.float64)
             for i, name in enumerate(["B03", "B04", "B08", "B11"])}

    def safe_index(a, b):
        denom = a + b
        with np.errstate(invalid="ignore", divide="ignore"):
            idx = np.where(denom != 0, (a - b) / denom, np.nan)
        mask = (a <= 0) | (b <= 0) | (a > 10000) | (b > 10000)
        idx[mask] = np.nan
        return idx.astype(np.float32)

    ndvi  = safe_index(bands["B08"], bands["B04"])
    mndwi = safe_index(bands["B03"], bands["B11"])
    return ndvi, mndwi, profile


# ------------------------------------------------------------------ #
# 3. ラスタ保存 & ゾーン統計
# ------------------------------------------------------------------ #

def save_raster(arr: np.ndarray, path: Path, profile: dict) -> None:
    out = np.where(np.isnan(arr), profile["nodata"], arr).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def calc_zonal(raster_path: Path, mesh: gpd.GeoDataFrame,
               stats: list, prefix: str) -> pd.DataFrame:
    result = zonal_stats(mesh, str(raster_path), stats=stats,
                         nodata=-9999.0, all_touched=False)
    df = pd.DataFrame(result, index=mesh.index)
    # dict キー名で rename（位置指定だと rasterstats の返す順序にズレが生じる）
    df = df.rename(columns={s: f"{prefix}_{s}" for s in stats})
    return df


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    mesh = gpd.read_file(MESH_PATH).set_index("mesh_id")

    # メッシュのUTM範囲を確認
    bounds = mesh.total_bounds
    print(f"メッシュ範囲 (UTM53N): {bounds}")

    all_dfs = []
    for event_key in ["2020", "2023"]:
        print(f"\n=== {event_key}年イベント ===")
        items = search_best_scene(catalog, event_key)
        ndvi, mndwi, profile = compute_indices(items)

        # ラスタ範囲確認
        t = profile["transform"]
        raster_bounds = rasterio.transform.array_bounds(
            profile["height"], profile["width"], t)
        print(f"    ラスタ範囲 (UTM53N): {raster_bounds}")

        valid_ndvi  = ndvi[~np.isnan(ndvi)]
        valid_mndwi = mndwi[~np.isnan(mndwi)]
        print(f"    NDVI  range: {valid_ndvi.min():.3f} ~ {valid_ndvi.max():.3f}")
        print(f"    MNDWI range: {valid_mndwi.min():.3f} ~ {valid_mndwi.max():.3f}")

        ndvi_path  = OUT_DIR / f"ndvi_{event_key}.tif"
        mndwi_path = OUT_DIR / f"mndwi_{event_key}.tif"
        save_raster(ndvi,  ndvi_path,  profile)
        save_raster(mndwi, mndwi_path, profile)

        print("    ゾーン統計計算中...")
        ndvi_df  = calc_zonal(ndvi_path,  mesh, ["mean", "min"], "ndvi")
        mndwi_df = calc_zonal(mndwi_path, mesh, ["mean", "max"], "mndwi")

        df = pd.concat([ndvi_df, mndwi_df], axis=1).reset_index()
        df["event"] = event_key
        all_dfs.append(df)

        n_valid = df["ndvi_mean"].notna().sum()
        print(f"    有効メッシュ: {n_valid} / {len(df)}")

    result = pd.concat(all_dfs, ignore_index=True)
    result.to_parquet(OUT_PARQ, index=False)
    print(f"\n保存完了: {OUT_PARQ}")
    print(result.groupby("event")[["ndvi_mean", "mndwi_mean"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
