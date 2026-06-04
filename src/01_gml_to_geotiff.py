"""
国土地理院 DEM5A GML(XML) → GeoTIFF 変換
各XMLファイルを1つのGeoTIFFタイルに変換する
出力先: ../output/dem_tiles/
"""

import xml.etree.ElementTree as ET
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent.parent
DEM_DIR = BASE_DIR / "DEM"
OUT_DIR = BASE_DIR / "output" / "dem_tiles"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GML_NS = {
    "gml": "http://www.opengis.net/gml/3.2",
    "fgd": "http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema",
}
NODATA = -9999.0


def parse_gml(filepath: Path):
    tree = ET.parse(filepath)
    root = tree.getroot()

    lower = root.find(".//gml:lowerCorner", GML_NS).text.split()
    upper = root.find(".//gml:upperCorner", GML_NS).text.split()
    # JGD2024.bl は (緯度, 経度) 順
    lat_min, lon_min = float(lower[0]), float(lower[1])
    lat_max, lon_max = float(upper[0]), float(upper[1])

    high = root.find(".//gml:high", GML_NS).text.split()
    cols = int(high[0]) + 1  # 0-indexed max → size
    rows = int(high[1]) + 1

    raw = root.find(".//gml:tupleList", GML_NS).text.strip().splitlines()
    elevations = np.array(
        [float(line.split(",")[1]) for line in raw if "," in line],
        dtype=np.float32,
    )

    expected = rows * cols
    if len(elevations) != expected:
        # エッジタイル等でデータが少ない・多い場合: nodata でパディングまたはトリミング
        if len(elevations) < expected:
            elevations = np.concatenate(
                [elevations, np.full(expected - len(elevations), NODATA, dtype=np.float32)]
            )
        else:
            elevations = elevations[:expected]

    data = elevations.reshape(rows, cols)
    return data, (lon_min, lat_min, lon_max, lat_max)


def gml_to_geotiff(xml_path: Path) -> Path:
    out_path = OUT_DIR / (xml_path.stem + ".tif")
    if out_path.exists():
        return out_path  # スキップ（再実行時）

    data, bounds = parse_gml(xml_path)
    rows, cols = data.shape
    lon_min, lat_min, lon_max, lat_max = bounds

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, cols, rows)
    crs = CRS.from_epsg(6668)  # JGD2011 geographic (JGD2024と同等扱い)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=NODATA,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    return out_path


def main():
    xml_files = sorted(DEM_DIR.rglob("*.xml"))
    # fmdid (メタデータ) XMLを除外
    xml_files = [f for f in xml_files if "FG-GML" in f.name]
    print(f"対象XMLファイル数: {len(xml_files)}")

    t0 = time.time()
    ok, err = 0, 0
    for i, f in enumerate(xml_files, 1):
        try:
            out = gml_to_geotiff(f)
            ok += 1
            if i % 20 == 0 or i == len(xml_files):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(xml_files)}] {elapsed:.1f}s 経過")
        except Exception as e:
            print(f"  エラー: {f.name} → {e}")
            err += 1

    print(f"\n完了: 成功 {ok} / エラー {err}")
    print(f"出力先: {OUT_DIR}")


if __name__ == "__main__":
    main()
