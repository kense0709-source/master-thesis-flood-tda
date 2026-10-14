"""
Step 8: 土地利用・地形分類を 500m メッシュの面積率に集計する（Model D の特徴量）

入力
  土地利用/L03-b-16_*-jgd_GML/*.shp
      国土数値情報 土地利用細分メッシュ 平成28年度（2016）。約100m セル。2020・2023 のどちらの水害より前の時点
  地形分類/tiles/experimental_landformclassification{1,2}/14_*.geojson（14a_download_landform.py）
      国土地理院 地形分類（自然地形・人工地形）。自然地形は2つの出典が合成されている:
        5桁コード … 土地条件図など
        7桁コード … 治水地形分類図（更新版）
      重なる場所は 7桁（治水地形分類図。水害向けに作られた図）を優先する

方法
  メッシュ格子（EPSG:32653、500m）にそろえた 25m 格子に各ポリゴンのクラスを焼き付け（rasterize）、
  メッシュごとに各クラスの画素の割合をとる。1 メッシュ = 20 × 20 画素
  地形分類はデータの整備範囲が限られる（主に平野部）ので、整備範囲の割合 lf_covered も出す。
  整備範囲外は各クラス 0 になるため、「その地形がない」と「図がない」を lf_covered で区別できる

出力: output/landuse_features.parquet（mesh_id ＋ lu_* 7列 ＋ lf_* 13列）
      output/landuse_check.png（主なクラスの地図）
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import shape

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output"
CRS      = "EPSG:32653"
plt.rcParams["font.family"] = "Yu Gothic"
RES      = 25.0          # 焼き付けの画素 [m]（500m の 1/20）

# ---- 土地利用（L03-b の土地利用種コード → 列）----
LU_CLASSES = {
    "lu_paddy":     ["0100"],                      # 田
    "lu_farm":      ["0200"],                      # その他の農用地
    "lu_forest":    ["0500"],                      # 森林
    "lu_building":  ["0700"],                      # 建物用地
    "lu_transport": ["0901", "0902"],              # 道路・鉄道
    "lu_water":     ["1100"],                      # 河川地及び湖沼
    "lu_other":     ["0600", "1000", "1400", "1500", "1600"],  # 荒地・その他の用地・海浜・海水域・ゴルフ場
}

# ---- 地形分類（図式コード → 列）。名称は style.js の codeList ----
LF_NAMES = {
    "lf_mountain":   ["山地斜面等", "崖･段丘崖", "地すべり地形"],
    "lf_terrace":    ["台地･段丘"],
    "lf_fan":        ["山麓堆積地形", "扇状地"],
    "lf_levee":      ["自然堤防", "砂州・砂丘", "天井川等"],          # 低地の微高地
    "lf_valley":     ["凹地・浅い谷"],
    "lf_floodplain": ["氾濫平野・海岸平野"],
    "lf_backmarsh":  ["後背低地･湿地"],
    "lf_oldchannel": ["旧河道", "落堀"],
    "lf_riverbed":   ["河川敷･浜"],
    "lf_water":      ["水部", "旧水部"],
    # 人工地形（レイヤ2）
    "lf_fill":       ["盛土地･埋立地", "改変工事中"],
    "lf_cut":        ["切土地", "農耕平坦化地"],
    "lf_reclaim":    ["干拓地"],
}
CODE_NAME = {
    # 自然地形
    "10101": "山地斜面等", "11201": "山地斜面等", "11202": "山地斜面等", "11203": "山地斜面等",
    "11204": "山地斜面等", "1010101": "山地斜面等",
    "10202": "崖･段丘崖", "10204": "崖･段丘崖", "2010201": "崖･段丘崖",
    "10205": "地すべり地形", "10206": "地すべり地形",
    **{c: "台地･段丘" for c in ["10301", "10302", "10303", "10304", "10305", "10306", "10307",
                                 "10308", "10310", "10312", "10314", "10508", "2010101"]},
    **{c: "山麓堆積地形" for c in ["10401", "10402", "10403", "10404", "10406", "10407", "3010101"]},
    "10501": "扇状地", "10502": "扇状地", "3020101": "扇状地",
    "10503": "自然堤防", "3040101": "自然堤防",
    "10506": "天井川等", "10507": "天井川等", "10801": "天井川等",
    "10504": "砂州・砂丘", "10505": "砂州・砂丘", "10512": "砂州・砂丘", "3050101": "砂州・砂丘",
    "10601": "凹地・浅い谷", "2010301": "凹地・浅い谷",
    "3030101": "氾濫平野・海岸平野", "10701": "氾濫平野・海岸平野", "10702": "氾濫平野・海岸平野",
    "10705": "氾濫平野・海岸平野",
    "10703": "後背低地･湿地", "10804": "後背低地･湿地", "3030201": "後背低地･湿地",
    "10704": "旧河道", "3040201": "旧河道", "3040202": "旧河道", "3040301": "落堀",
    "10802": "河川敷･浜", "10803": "河川敷･浜", "10807": "河川敷･浜", "10808": "河川敷･浜",
    "10805": "水部", "10806": "水部", "10901": "水部", "10903": "水部", "5010201": "水部",
    "10904": "旧水部", "5010301": "旧水部",
    # 人工地形
    "11001": "切土地", "11003": "切土地", "11009": "切土地", "11011": "切土地", "4010301": "切土地",
    "11002": "農耕平坦化地", "11008": "干拓地", "4010101": "干拓地",
    **{c: "盛土地･埋立地" for c in ["11004", "11005", "11006", "11007", "11014", "4010201"]},
    "11010": "改変工事中",
}


def grid(mesh: gpd.GeoDataFrame):
    x0, y0, x1, y1 = mesh.total_bounds
    w, h = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    return from_origin(x0, y1, RES, RES), (h, w)


def burn(geoms, values, transform, shape_):
    """values は 1 始まりのクラス番号。0 = データなし"""
    return rasterize(zip(geoms, values), out_shape=shape_, transform=transform,
                     fill=0, dtype="int16")


def fractions(cls_r: np.ndarray, mesh_r: np.ndarray, n_cls: int, n_mesh: int) -> np.ndarray:
    """メッシュ × クラス（0 = データなし を含む）の画素数 → 割合"""
    ok = mesh_r >= 0
    idx = mesh_r[ok].astype(np.int64) * (n_cls + 1) + cls_r[ok]
    cnt = np.bincount(idx, minlength=n_mesh * (n_cls + 1)).reshape(n_mesh, n_cls + 1)
    return cnt / cnt.sum(axis=1, keepdims=True)


def load_landuse(bounds_ll) -> gpd.GeoDataFrame:
    parts = []
    for f in sorted(glob.glob(str(BASE_DIR / "土地利用" / "L03-b-16_*-jgd_GML" / "*.shp"))):
        g = gpd.read_file(f, encoding="cp932", bbox=tuple(bounds_ll))
        g = g.rename(columns={g.columns[1]: "code"})[["code", "geometry"]]
        parts.append(g)
    return pd.concat(parts, ignore_index=True).set_crs(4612, allow_override=True)


def load_landform(layer: str) -> gpd.GeoDataFrame:
    rows = []
    for f in glob.glob(str(BASE_DIR / "地形分類" / "tiles" / layer / "*.geojson")):
        if os.path.getsize(f) == 0:
            continue
        with open(f, encoding="utf-8") as fh:
            for ft in json.load(fh)["features"]:
                rows.append((str(ft["properties"]["code"]), shape(ft["geometry"])))
    return gpd.GeoDataFrame({"code": [r[0] for r in rows]},
                            geometry=[r[1] for r in rows], crs=4326)


def main() -> None:
    t0 = time.time()
    mesh = gpd.read_file(OUT_DIR / "mesh_500m.gpkg")
    transform, shape_ = grid(mesh)
    n_mesh = len(mesh)
    mesh_r = burn(mesh.geometry, np.arange(n_mesh) + 1, transform, shape_).astype(np.int32) - 1
    # 1 メッシュ = 20×20 画素になっているか（格子がずれていると割合が狂う）
    cnt = np.bincount(mesh_r[mesh_r >= 0], minlength=n_mesh)
    assert (cnt == 400).all(), f"メッシュの画素数が 400 でない: {np.unique(cnt)}"
    bounds_ll = mesh.to_crs(4326).total_bounds + np.array([-0.01, -0.01, 0.01, 0.01])
    out = pd.DataFrame({"mesh_id": mesh["mesh_id"].values})

    # ---- 土地利用 ----
    lu = load_landuse(bounds_ll).to_crs(CRS)
    code2cls = {c: i + 1 for i, cs in enumerate(LU_CLASSES.values()) for c in cs}
    unknown = set(lu["code"]) - set(code2cls)
    assert not unknown, f"未対応の土地利用コード: {unknown}"
    r = burn(lu.geometry, lu["code"].map(code2cls).values, transform, shape_)
    fr = fractions(r, mesh_r, len(LU_CLASSES), n_mesh)
    for i, name in enumerate(LU_CLASSES):
        out[name] = fr[:, i + 1]
    print(f"土地利用: {len(lu):,} セル、データなしの画素 {fr[:, 0].mean():.4f}（{time.time() - t0:.0f}s）")

    # ---- 地形分類 ----
    name2cls = {n: i + 1 for i, ns in enumerate(LF_NAMES.values()) for n in ns}
    nat = load_landform("experimental_landformclassification1")
    art = load_landform("experimental_landformclassification2")
    for g, lab in [(nat, "自然"), (art, "人工")]:
        unknown = set(g["code"]) - set(CODE_NAME)
        assert not unknown, f"未対応の地形分類コード（{lab}）: {unknown}"
        g["cls"] = g["code"].map(CODE_NAME).map(name2cls)
    nat = nat.to_crs(CRS)
    art = art.to_crs(CRS)
    # 自然地形: 5桁（土地条件図等）を先に、7桁（治水地形分類図）で上書き
    old = nat[nat["code"].str.len() == 5]
    new = nat[nat["code"].str.len() == 7]
    r_old = burn(old.geometry, old["cls"].values, transform, shape_)
    r_new = burn(new.geometry, new["cls"].values, transform, shape_)
    both = (r_old > 0) & (r_new > 0)
    agree = (r_old[both] == r_new[both]).mean() if both.any() else np.nan
    print(f"自然地形: 5桁 {len(old):,}・7桁 {len(new):,} ポリゴン。両方ある画素 {both.sum():,}"
          f"（クラス一致率 {agree:.3f}）→ 7桁を優先")
    r_nat = np.where(r_new > 0, r_new, r_old)
    # 人工地形は自然地形の上に重ねる（盛土地・干拓地などは人工改変が浸水に直結するため）
    r_art = burn(art.geometry, art["cls"].values, transform, shape_)
    r_lf = np.where(r_art > 0, r_art, r_nat)
    fr = fractions(r_lf, mesh_r, len(LF_NAMES), n_mesh)
    for i, name in enumerate(LF_NAMES):
        out[name] = fr[:, i + 1]
    out["lf_covered"] = 1 - fractions(r_nat, mesh_r, len(LF_NAMES), n_mesh)[:, 0]
    print(f"地形分類: 整備範囲の割合 平均 {out['lf_covered'].mean():.3f}、"
          f"全面整備のメッシュ {(out['lf_covered'] > 0.99).mean():.3f}")

    out.to_parquet(OUT_DIR / "landuse_features.parquet", index=False)

    # ---- 確認：浸水との関係（単独 ROC-AUC）と地図 ----
    from sklearn.metrics import roc_auc_score
    lab = pd.read_parquet(OUT_DIR / "mesh_labels.parquet")
    y = lab.groupby("mesh_id")["label"].max() if "event" in lab else lab.set_index("mesh_id")["label"]
    z = out.set_index("mesh_id").join(y.rename("y"), how="inner")
    print("\n列              平均    浸水率（>0.5のメッシュ）  単独ROC-AUC")
    for c in out.columns[1:]:
        hi = z[c] > 0.5
        rate = z.loc[hi, "y"].mean() if hi.any() else np.nan
        print(f"{c:15s} {z[c].mean():.3f}   {rate:.3f}（{hi.sum():5d}）          "
              f"{roc_auc_score(z['y'], z[c]):.3f}")

    show = ["lu_paddy", "lu_building", "lf_floodplain", "lf_backmarsh",
            "lf_oldchannel", "lf_levee", "lf_reclaim", "lf_covered"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    nr, nc = mesh["mesh_row"].max() + 1, mesh["mesh_col"].max() + 1
    for ax, c in zip(axes.ravel(), show):
        img = np.full((nr, nc), np.nan)
        img[mesh["mesh_row"], mesh["mesh_col"]] = out[c].values
        ax.imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=1)
        flood = np.full((nr, nc), np.nan)
        yy = z["y"].reindex(mesh["mesh_id"]).values
        flood[mesh["mesh_row"], mesh["mesh_col"]] = np.where(yy == 1, 1, np.nan)
        ax.contour(np.nan_to_num(flood), levels=[0.5], colors="red", linewidths=0.4,
                   origin="lower")
        ax.set_title(c)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("土地利用・地形分類の面積率（赤線＝2020・2023 のいずれかで浸水）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "landuse_check.png", dpi=120)
    print(f"\n完了 {time.time() - t0:.0f}s → output/landuse_features.parquet")


if __name__ == "__main__":
    main()
