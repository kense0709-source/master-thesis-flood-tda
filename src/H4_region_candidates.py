"""Phase H 手順3：地域（流域）の候補を選ぶための集計

H3 の 500m マス（福岡・佐賀・長崎・熊本）に、
  ・水系コード（国土数値情報 W07 流域メッシュ、100m）… マスの中心が入る 100m メッシュの値
  ・平均標高・平均傾斜（国土数値情報 G04-a 標高・傾斜度 3次メッシュ、1km）
を付け、水系ごとに使えるマス数・1 の数・地形の性質を集計する。
出力：output/hazard/H4_basins.csv、H4_cells_basin.parquet、fig_H4_basins.png
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import zipfile


def shp_in(f, suffix):
    n = [i for i in zipfile.ZipFile(f).namelist() if i.endswith(suffix)][0]
    return f"zip://{f}!{n}"

ROOT = Path(__file__).resolve().parents[1]
HZ = ROOT / "ハザードマップ"
OUT = ROOT / "output/hazard"
LAB = "label_2km"
plt.rcParams.update({"font.family": "Yu Gothic", "font.size": 10})

cells = pd.read_parquet(OUT / "H3_cells.parquet")
pts = gpd.GeoSeries(gpd.points_from_xy(cells["x"], cells["y"]), crs=6670).to_crs(6668)
lat, lon = pts.y.values, pts.x.values

# 地域メッシュコード（1次 4 桁＋2次 2 桁＋3次 2 桁＋1/10 細分 2 桁）
a = lat * 1.5
p, u = np.floor(a).astype(int), np.floor(lon - 100).astype(int)
fa, fo = (a - p) * 8, (lon - 100 - u) * 8
q, v = np.floor(fa).astype(int), np.floor(fo).astype(int)
ga, go = (fa - q) * 10, (fo - v) * 10
r, w = np.floor(ga).astype(int), np.floor(go).astype(int)
s, x = np.floor((ga - r) * 10).astype(int), np.floor((go - w) * 10).astype(int)
m1 = [f"{pp}{uu:02d}" for pp, uu in zip(p, u)]
m3 = [f"{a_}{qq}{vv}{rr}{ww}" for a_, qq, vv, rr, ww in zip(m1, q, v, r, w)]
m100 = [f"{c}{ss}{xx}" for c, ss, xx in zip(m3, s, x)]
cells["mesh1"], cells["mesh3"], cells["mesh100"] = m1, m3, m100

# 水系（W07）
need = set(m100)
basin = []
for f in sorted((HZ / "raw_w07").glob("W07-09_*-jgd_GML.zip")):
    stem = f.stem.replace("_GML", "")
    df = pyogrio.read_dataframe(shp_in(f, "_ValleyMesh.shp"), read_geometry=False,
                                columns=["W07_001", "W07_002", "W07_004"], encoding="cp932")
    basin.append(df[df["W07_001"].isin(need)])
basin = pd.concat(basin).rename(columns={"W07_001": "mesh100", "W07_002": "suikei", "W07_004": "suikei_name"})
cells = cells.merge(basin.drop_duplicates("mesh100"), on="mesh100", how="left")

# 標高・傾斜（G04-a）
topo = []
for f in sorted((HZ / "raw_g04").glob("G04-a-11_*-jgd_GML.zip")):
    stem = f.stem.replace("_GML", "")
    df = pyogrio.read_dataframe(shp_in(f, "TertiaryMesh.shp"), read_geometry=False,
                                columns=["G04a_001", "G04a_002", "G04a_010"])
    topo.append(df)
topo = pd.concat(topo).rename(columns={"G04a_001": "mesh3", "G04a_002": "elev", "G04a_010": "slope"})
for c in ["elev", "slope"]:
    topo[c] = pd.to_numeric(topo[c], errors="coerce").where(lambda s_: s_ > -999)
cells = cells.merge(topo.drop_duplicates("mesh3"), on="mesh3", how="left")
cells.to_parquet(OUT / "H4_cells_basin.parquet", index=False)

c = cells[cells["suikei"].notna() & (cells["suikei"] != "unknown")]
use = c[c[LAB] != "使わない"]
g = c.groupby(["suikei", "suikei_name"])
t = pd.DataFrame({
    "県": g["pref"].agg(lambda s_: "・".join(s_.value_counts().index[:3])),
    "マス": g.size(),
    "使える": use.groupby(["suikei", "suikei_name"]).size(),
    "1": use.groupby(["suikei", "suikei_name"])[LAB].apply(lambda s_: (s_ == "1").sum()),
    "平均標高m": g["elev"].mean().round(0),
    "標高20m未満の割合": g["elev"].apply(lambda s_: (s_ < 20).mean()).round(2),
    "平均傾斜度": g["slope"].mean().round(1),
    "傾斜3度未満の割合": g["slope"].apply(lambda s_: (s_ < 3).mean()).round(2),
}).fillna(0)
t["0"] = t["使える"] - t["1"]
t["使える割合"] = (t["使える"] / t["マス"]).round(2)
t["1の割合"] = (t["1"] / t["使える"].where(t["使える"] > 0)).round(3)
t = t.reset_index().sort_values("1", ascending=False)
cols = ["suikei", "suikei_name", "県", "マス", "使える", "1", "0", "使える割合", "1の割合",
        "平均標高m", "標高20m未満の割合", "平均傾斜度", "傾斜3度未満の割合"]
t[cols].to_csv(OUT / "H4_basins.csv", index=False, encoding="utf-8-sig")
print(t[cols].head(30).to_string(index=False))
print("水系なし（海岸の小流域など）のマス", int((cells["suikei"].isna() | (cells["suikei"] == "unknown")).sum()))

# 地図：1 が 50 マス以上の水系を色分け
big = t[t["1"] >= 50]["suikei"].tolist()
fig, ax = plt.subplots(figsize=(9, 10.5), facecolor="#fcfcfb")
ax.scatter(cells["x"], cells["y"], s=0.6, c="#e6e4dc", marker="s", linewidths=0)
cmap = plt.get_cmap("tab20")
for i, sk in enumerate(big):
    d = c[c["suikei"] == sk]
    ax.scatter(d["x"], d["y"], s=0.6, color=cmap(i % 20), marker="s", linewidths=0)
    ax.text(d["x"].median(), d["y"].median(), d["suikei_name"].iloc[0], fontsize=8.5, fontweight="bold",
            ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.85))
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_title("1 が 50 マス以上ある水系（4県内の部分）", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig_H4_basins.png", dpi=130, facecolor="#fcfcfb")
