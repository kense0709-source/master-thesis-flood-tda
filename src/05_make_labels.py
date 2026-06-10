"""
浸水推定図GeoJSON → 500mメッシュ ラベル化（0/1）

処理フロー:
  1. output/mesh_500m.gpkg を読み込む（UTM53N, 10,025セル）
  2. 各イベントの浸水推定図GeoJSONを読み込む
  3. 空間結合（intersects）でメッシュセルと浸水ポリゴンを照合
  4. 1つでも浸水ポリゴンと交差するセル → label=1, それ以外 → label=0
  5. output/mesh_labels.parquet に保存

出力:
  output/mesh_labels.parquet
    列: mesh_id, event, label (0 or 1)
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

GEOJSON = {
    "2020": next((BASE_DIR / "浸水推定図" / "2020_第2報").rglob("*.geojson")),
    "2023": next((BASE_DIR / "浸水推定図" / "2023").rglob("*.geojson")),
}
MESH_PATH  = BASE_DIR / "output" / "mesh_500m.gpkg"
OUT_PATH   = BASE_DIR / "output" / "mesh_labels.parquet"


def make_labels(mesh: gpd.GeoDataFrame, flood: gpd.GeoDataFrame) -> pd.Series:
    """mesh（UTM53N）と flood（任意CRS）を空間結合してラベルを返す"""
    flood_proj = flood.to_crs(mesh.crs)[["geometry"]]
    flood_proj = flood_proj[~flood_proj.is_empty & flood_proj.is_valid]

    joined = gpd.sjoin(mesh[["geometry"]], flood_proj, how="left", predicate="intersects")
    flooded_ids = set(joined.loc[joined["index_right"].notna()].index)

    labels = pd.Series(
        [1 if idx in flooded_ids else 0 for idx in mesh.index],
        index=mesh.index,
        name="label",
        dtype="int8",
    )
    return labels


def main():
    print("メッシュ読み込み中...")
    mesh = gpd.read_file(MESH_PATH)
    mesh = mesh.set_index("mesh_id")
    print(f"  メッシュ数: {len(mesh)}")

    records = []
    for event, geojson_path in GEOJSON.items():
        print(f"\n=== {event}年イベント ===")
        flood = gpd.read_file(geojson_path)
        print(f"  浸水ポリゴン数: {len(flood)}, CRS: {flood.crs}")

        labels = make_labels(mesh, flood)
        n1 = labels.sum()
        n0 = len(labels) - n1
        ratio = n1 / len(labels) * 100
        print(f"  浸水あり (label=1): {n1:,} セル  ({ratio:.1f}%)")
        print(f"  浸水なし (label=0): {n0:,} セル  ({100-ratio:.1f}%)")

        df = labels.reset_index()
        df.columns = ["mesh_id", "label"]
        df["event"] = event
        records.append(df)

    result = pd.concat(records, ignore_index=True)[["mesh_id", "event", "label"]]
    result.to_parquet(OUT_PATH, index=False)
    print(f"\n保存完了: {OUT_PATH}")
    print(result.groupby(["event", "label"]).size().rename("count").to_string())


if __name__ == "__main__":
    main()
