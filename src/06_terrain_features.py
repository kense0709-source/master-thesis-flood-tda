"""
TWI・傾斜・流向の計算（pysheds）→ 500mメッシュ統計

処理フロー:
  1. DEM（EPSG:6668）をUTM53N（EPSG:32653）へ再投影
  2. pysheds で pit fill → resolve flats → D8 流向 → 流域累積面積
  3. 傾斜（slope）を numpy gradient で計算
  4. TWI = ln( acc * cellsize / tan(slope) ) を計算
  5. rasterstats でゾーン統計 → 500mメッシュに割り当て

出力:
  output/terrain_features.parquet
    列: mesh_id, twi_mean, slope_mean, slope_max, flowacc_mean
  output/twi.tif  （TWIラスタ、確認用）
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from pysheds.grid import Grid
from rasterstats import zonal_stats
import pandas as pd
from pathlib import Path
import tempfile, os

BASE_DIR   = Path(__file__).parent.parent
DEM_PATH   = BASE_DIR / "output" / "dem_merged.tif"
MESH_PATH  = BASE_DIR / "output" / "mesh_500m.gpkg"
OUT_PARQ   = BASE_DIR / "output" / "terrain_features.parquet"
OUT_TWI    = BASE_DIR / "output" / "twi.tif"
OUT_SLOPE  = BASE_DIR / "output" / "slope.tif"

DST_CRS    = "EPSG:32653"   # UTM Zone 53N
NODATA_DEM = -9999.0
NODATA_OUT = np.nan


# ------------------------------------------------------------------ #
# 1. DEM を UTM53N へ再投影
# ------------------------------------------------------------------ #

def reproject_dem(src_path: Path, dst_path: Path) -> None:
    print("DEM を UTM53N に再投影中...")
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": DST_CRS, "transform": transform,
                       "width": width, "height": height,
                       "nodata": NODATA_DEM, "dtype": "float32"})
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=DST_CRS,
                resampling=Resampling.bilinear,
            )
    with rasterio.open(dst_path) as r:
        print(f"  再投影済み DEM: {r.width}×{r.height} px, res={r.res[0]:.2f}m")


# ------------------------------------------------------------------ #
# 2. pysheds でフロー解析
# ------------------------------------------------------------------ #

def compute_flow(dem_utm_path: Path):
    """流向・流域累積を計算。flow_acc (格子数) を返す"""
    print("pysheds: pit fill → resolve flats → D8 flow direction...")
    grid = Grid.from_raster(str(dem_utm_path))
    dem  = grid.read_raster(str(dem_utm_path))

    # pit fill
    pit_filled = grid.fill_pits(dem)
    # flat 解消
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)

    # D8 流向
    fdir = grid.flowdir(inflated)

    # 流域累積（格子数）
    print("  流域累積計算中...")
    acc = grid.accumulation(fdir)

    return grid, fdir, acc, inflated


# ------------------------------------------------------------------ #
# 3. 傾斜・TWI 計算
# ------------------------------------------------------------------ #

def compute_slope_twi(dem_utm_path: Path, acc: np.ndarray) -> tuple:
    """
    slope [rad] と TWI を計算。
    acc は pysheds の Raster オブジェクト or ndarray。
    """
    with rasterio.open(dem_utm_path) as src:
        dem_arr = src.read(1).astype(np.float32)
        cellsize = src.res[0]          # UTM なので単位 [m]
        profile  = src.profile.copy()
        nodata   = src.nodata

    dem_arr[dem_arr == nodata] = np.nan

    # 勾配（x, y 方向）→ 傾斜角 [rad]
    dz_dy, dz_dx = np.gradient(dem_arr, cellsize, cellsize)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_rad = np.clip(slope_rad, 1e-6, None)   # tan(0) 回避

    # TWI = ln( acc * cellsize / tan(slope) )
    acc_arr = np.array(acc, dtype=np.float32)
    specific_area = (acc_arr + 1) * cellsize      # +1 でゼロ流量防止
    twi = np.log(specific_area / np.tan(slope_rad))
    twi[np.isnan(dem_arr)] = np.nan

    return slope_rad, twi, profile, cellsize


def save_raster(arr: np.ndarray, path: Path, profile: dict) -> None:
    prof = profile.copy()
    prof.update({"dtype": "float32", "count": 1, "nodata": -9999.0})
    out = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


# ------------------------------------------------------------------ #
# 4. ゾーン統計 → メッシュへ割り当て
# ------------------------------------------------------------------ #

def zonal_mean(raster_path: Path, mesh: gpd.GeoDataFrame,
               stats: list, prefix: str) -> pd.DataFrame:
    result = zonal_stats(mesh, str(raster_path), stats=stats,
                         nodata=-9999.0, all_touched=False)
    df = pd.DataFrame(result, index=mesh.index)
    df.columns = [f"{prefix}_{s}" for s in stats]
    return df


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        dem_utm_path = Path(f.name)

    try:
        # 1. 再投影
        reproject_dem(DEM_PATH, dem_utm_path)

        # 2. フロー解析
        grid, fdir, acc, _ = compute_flow(dem_utm_path)

        # 3. 傾斜・TWI
        print("傾斜・TWI 計算中...")
        slope_rad, twi, profile, cellsize = compute_slope_twi(dem_utm_path, acc)
        print(f"  cellsize: {cellsize:.2f} m")
        valid = ~np.isnan(twi)
        print(f"  TWI  range: {twi[valid].min():.2f} ~ {twi[valid].max():.2f}")
        print(f"  Slope range: {np.degrees(slope_rad[valid]).min():.2f} ~ "
              f"{np.degrees(slope_rad[valid]).max():.2f} deg")

        # 4. ラスタ保存
        print("ラスタ保存中...")
        save_raster(twi,                 OUT_TWI,   profile)
        save_raster(np.degrees(slope_rad), OUT_SLOPE, profile)

        # 5. メッシュ読み込み（UTM53N）
        mesh = gpd.read_file(MESH_PATH).set_index("mesh_id")

        # 6. ゾーン統計
        print("ゾーン統計計算中...")
        twi_df   = zonal_mean(OUT_TWI,   mesh, ["mean"],         "twi")
        slope_df = zonal_mean(OUT_SLOPE, mesh, ["mean", "max"],  "slope")

        features = pd.concat([twi_df, slope_df], axis=1)
        features.index.name = "mesh_id"
        features = features.reset_index()

        print(f"\n有効メッシュ数: {features['twi_mean'].notna().sum()} / {len(features)}")
        print(features[["twi_mean", "slope_mean", "slope_max"]].describe().round(2).to_string())

        features.to_parquet(OUT_PARQ, index=False)
        print(f"\n保存完了: {OUT_PARQ}")

    finally:
        os.unlink(dem_utm_path)


if __name__ == "__main__":
    main()
