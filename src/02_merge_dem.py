"""
GeoTIFFタイルを1枚のDEMモザイクに結合する
入力: ../output/dem_tiles/*.tif
出力: ../output/dem_merged.tif
"""

from pathlib import Path
import rasterio
from rasterio.merge import merge
from rasterio.crs import CRS

BASE_DIR = Path(__file__).parent.parent
TILES_DIR = BASE_DIR / "output" / "dem_tiles"
OUT_PATH = BASE_DIR / "output" / "dem_merged.tif"

NODATA = -9999.0


def main():
    tif_files = sorted(TILES_DIR.glob("*.tif"))
    print(f"タイル数: {len(tif_files)}")
    if not tif_files:
        print("タイルが見つかりません。先に 01_gml_to_geotiff.py を実行してください。")
        return

    datasets = [rasterio.open(f) for f in tif_files]
    mosaic, transform = merge(datasets, nodata=NODATA)

    meta = datasets[0].meta.copy()
    meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "nodata": NODATA,
            "compress": "lzw",
            "bigtiff": "IF_SAFER",
        }
    )

    with rasterio.open(OUT_PATH, "w", **meta) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    print(f"出力: {OUT_PATH}")
    print(f"サイズ: {mosaic.shape[2]} cols × {mosaic.shape[1]} rows")

    # 範囲確認
    with rasterio.open(OUT_PATH) as ds:
        b = ds.bounds
        print(f"範囲: 経度 {b.left:.4f}〜{b.right:.4f}, 緯度 {b.bottom:.4f}〜{b.top:.4f}")


if __name__ == "__main__":
    main()
