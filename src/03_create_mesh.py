"""
500m×500mメッシュを作成し、各セルのDEM統計量を計算する
入力: ../output/dem_merged.tif
出力: ../output/mesh_500m.gpkg  (GeoPackage)
      ../output/mesh_dem_stats.parquet  (Parquet, 以降の処理で使用)

各セルの特徴量（この段階）:
  - elev_mean, elev_min, elev_max, elev_std  : 標高統計
  - valid_ratio  : 有効ピクセル割合（海・欠損除外）
  - mesh_row, mesh_col  : メッシュインデックス
  - center_lat, center_lon  : セル中心座標
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
from pyproj import Transformer

BASE_DIR = Path(__file__).parent.parent
DEM_PATH = BASE_DIR / "output" / "dem_merged.tif"
OUT_DIR = BASE_DIR / "output"

MESH_SIZE_M = 500       # メッシュ1辺 [m]
MIN_VALID_RATIO = 0.3   # 有効ピクセル率がこれ以下のセルは除外
NODATA = -9999.0

# 投影座標系: WGS84 UTM Zone 53N (九州全域をカバー)
UTM53N = "EPSG:32653"
GEO_CRS = "EPSG:6668"  # JGD2011 地理座標系


def build_mesh_grid(bounds_utm, mesh_size=MESH_SIZE_M):
    """UTM座標のboundsに500mグリッドを生成"""
    x_min, y_min, x_max, y_max = bounds_utm
    xs = np.arange(x_min, x_max, mesh_size)
    ys = np.arange(y_min, y_max, mesh_size)
    cells = []
    for row_i, y in enumerate(ys):
        for col_i, x in enumerate(xs):
            geom = box(x, y, x + mesh_size, y + mesh_size)
            cells.append({"mesh_row": row_i, "mesh_col": col_i, "geometry": geom})
    gdf = gpd.GeoDataFrame(cells, crs=UTM53N)
    gdf["mesh_id"] = gdf.index
    return gdf


def compute_dem_stats(gdf_mesh, dem_path):
    """各メッシュセルのDEM統計量を計算"""
    gdf_geo = gdf_mesh.to_crs(GEO_CRS)  # DEMと同じCRSに変換

    records = []
    with rasterio.open(dem_path) as src:
        total = len(gdf_geo)
        for i, (idx, row) in enumerate(gdf_geo.iterrows()):
            if i % 500 == 0:
                print(f"  {i}/{total} セル処理中...")
            try:
                data, _ = rio_mask(src, [row.geometry], crop=True, nodata=NODATA)
                vals = data[0].flatten()
                valid = vals[vals != NODATA]
                valid_ratio = len(valid) / len(vals) if len(vals) > 0 else 0.0

                if valid_ratio < MIN_VALID_RATIO or len(valid) == 0:
                    records.append(
                        {"mesh_id": idx, "valid_ratio": valid_ratio,
                         "elev_mean": np.nan, "elev_min": np.nan,
                         "elev_max": np.nan, "elev_std": np.nan}
                    )
                else:
                    records.append(
                        {"mesh_id": idx,
                         "valid_ratio": valid_ratio,
                         "elev_mean": float(np.mean(valid)),
                         "elev_min":  float(np.min(valid)),
                         "elev_max":  float(np.max(valid)),
                         "elev_std":  float(np.std(valid))}
                    )
            except Exception:
                records.append(
                    {"mesh_id": idx, "valid_ratio": 0.0,
                     "elev_mean": np.nan, "elev_min": np.nan,
                     "elev_max": np.nan, "elev_std": np.nan}
                )

    return pd.DataFrame(records).set_index("mesh_id")


def add_center_coords(gdf_mesh):
    """メッシュ中心の緯経度を付加（UTMで重心計算してから変換）"""
    centroids_utm = gdf_mesh.geometry.centroid  # UTM座標で正確に重心計算
    gdf_pts = gpd.GeoDataFrame(geometry=centroids_utm, crs=UTM53N).to_crs(GEO_CRS)
    gdf_mesh["center_lon"] = gdf_pts.geometry.x
    gdf_mesh["center_lat"] = gdf_pts.geometry.y
    return gdf_mesh


def main():
    if not DEM_PATH.exists():
        print("dem_merged.tif が見つかりません。先に 02_merge_dem.py を実行してください。")
        return

    print("DEMのCRS・範囲を確認...")
    with rasterio.open(DEM_PATH) as src:
        bounds_geo = src.bounds
        print(f"  CRS: {src.crs}")
        print(f"  範囲 (地理座標): {bounds_geo}")

    # UTM座標に変換して500mグリッドを作成
    print("\n500mメッシュグリッドを作成...")
    transformer = Transformer.from_crs(GEO_CRS, UTM53N, always_xy=True)
    x_min, y_min = transformer.transform(bounds_geo.left, bounds_geo.bottom)
    x_max, y_max = transformer.transform(bounds_geo.right, bounds_geo.top)
    bounds_utm = (x_min, y_min, x_max, y_max)

    gdf_mesh = build_mesh_grid(bounds_utm)
    print(f"  生成メッシュ数: {len(gdf_mesh)}")

    # 中心座標付加
    gdf_mesh = add_center_coords(gdf_mesh)

    # DEM統計量計算
    print("\nDEM統計量を計算中（時間がかかります）...")
    stats_df = compute_dem_stats(gdf_mesh, DEM_PATH)
    gdf_mesh = gdf_mesh.join(stats_df)

    # 有効セルのみ保持
    valid_mask = gdf_mesh["valid_ratio"] >= MIN_VALID_RATIO
    gdf_valid = gdf_mesh[valid_mask].copy()
    print(f"\n有効メッシュ数: {len(gdf_valid)} / {len(gdf_mesh)}")

    # 保存
    gpkg_path = OUT_DIR / "mesh_500m.gpkg"
    gdf_valid.to_file(gpkg_path, driver="GPKG")
    print(f"GeoPackage保存: {gpkg_path}")

    parquet_path = OUT_DIR / "mesh_dem_stats.parquet"
    df_out = gdf_valid.drop(columns="geometry")
    df_out.to_parquet(parquet_path)
    print(f"Parquet保存: {parquet_path}")

    print("\n--- サンプル（先頭5行）---")
    print(df_out[["mesh_row", "mesh_col", "center_lat", "center_lon",
                   "elev_mean", "elev_min", "elev_max"]].head())


if __name__ == "__main__":
    main()
