"""
Step 8 の準備: 国土地理院「地形分類」ベクトルタイル（GeoJSON）を流域の範囲だけ取得する

データ: https://github.com/gsi-cyberjapan/experimental_landformclassification
  experimental_landformclassification1 … 地形分類（自然地形）。土地条件図・治水地形分類図（更新版）等の合成
  experimental_landformclassification2 … 地形分類（人工地形）。盛土地・干拓地・切土地など
  ZL14 が実データ（人工地形は ZL14〜16 のみ実データ。ZL13 以下は範囲図）。
  属性は図式コード `code`（7桁）。名称は style.js の codeList（14_landuse.py に転記）。

水害地形分類図（ecom-plat、紙地図のスキャン）の代わりに使う理由は method_choices.md ステップ9。

出力: 地形分類/tiles/{layer}/14_{x}_{y}.geojson（取得済みはスキップ。データのないタイルは空ファイル）
"""

from __future__ import annotations

import math
import time
import urllib.error
import urllib.request
from pathlib import Path

import geopandas as gpd

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output"
TILE_DIR = BASE_DIR / "地形分類" / "tiles"

LAYERS = ["experimental_landformclassification1", "experimental_landformclassification2"]
Z = 14
URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.geojson"


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    r = math.radians(lat)
    y = int((1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n)
    return x, y


def main() -> None:
    mesh = gpd.read_file(OUT_DIR / "mesh_500m.gpkg").to_crs(4326)
    lon0, lat0, lon1, lat1 = mesh.total_bounds
    x0, y1 = lonlat_to_tile(lon0, lat0, Z)
    x1, y0 = lonlat_to_tile(lon1, lat1, Z)
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    print(f"ZL{Z} タイル {x1 - x0 + 1} × {y1 - y0 + 1} = {len(tiles)} 枚 × {len(LAYERS)} レイヤ")

    for layer in LAYERS:
        d = TILE_DIR / layer
        d.mkdir(parents=True, exist_ok=True)
        n_get = n_empty = 0
        for i, (x, y) in enumerate(tiles):
            f = d / f"{Z}_{x}_{y}.geojson"
            if f.exists():
                continue
            try:
                with urllib.request.urlopen(URL.format(layer=layer, z=Z, x=x, y=y), timeout=60) as r:
                    f.write_bytes(r.read())
                n_get += 1
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                f.write_bytes(b"")          # データなし（404）。次回の再取得を避ける
                n_empty += 1
            time.sleep(0.1)                 # サーバ負荷への配慮
            if (i + 1) % 100 == 0:
                print(f"  {layer[-1]}: {i + 1}/{len(tiles)}", flush=True)
        print(f"{layer}: 取得 {n_get}、データなし {n_empty}")


if __name__ == "__main__":
    main()
