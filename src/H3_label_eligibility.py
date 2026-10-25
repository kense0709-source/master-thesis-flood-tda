"""Phase H 手順2：500m マスごとに 1 / 0 / 使わない を判定する（福岡・佐賀・長崎・熊本）

ルール（method_choices.md ステップ10、2026-10-25 決定）
  1       ：L2（想定最大規模）の区域がマスの 0.5 以上
  0       ：それ以外で、マスから D km 以内に「L2 未計算の一級・二級河川」が無い
  使わない：それ以外（近くに未計算の一級・二級河川がある）
  D は 1・2・5 km。基本は 2 km

入力：ハザードマップ/raw/A31a-25_{89,40..44}_*_GEOJSON.zip（L2 のファイルだけ使う）
      ハザードマップ/raw_w05/W05-07_{40..43}_GML.zip（川の中心線と河川コード）
      ハザードマップ/raw_n03/N03-20250101_{40..43}_GML.zip（県の境界。海を除く）
      ハザードマップ/A31_R7_datalist.xlsx（L2 掲載河川の一覧）
出力：output/hazard/H3_cells.parquet、H3_summary.txt、fig_H3_eligibility.png
座標：EPSG:6670（平面直角 II 系）。地図と集計用のマス。学習用のマスは地域ごとの UTM で作り直す
"""
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from rasterio.features import rasterize
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
HZ = ROOT / "ハザードマップ"
OUT = ROOT / "output/hazard"
CRS = 6670
CELL, SUB = 500.0, 20                 # 500m マス、25m 画素
PIX = CELL / SUB
PREFS = {"40": "福岡", "41": "佐賀", "42": "長崎", "43": "熊本"}
HZ_ZIPS = ["89_10", "40_10", "40_20", "41_10", "42_10", "43_10", "43_20", "44_10", "44_20"]
DISTS = [1, 2, 5]
LEGAL = {1, 2, 3, 5, 6, 7}             # W05_003：一級直轄・一級指定・二級（対象区間を兼ねるものを含む）

plt.rcParams.update({"font.family": "Yu Gothic", "font.size": 10})


def log(*a):
    print(*a, flush=True)


# --- 県の境界 ---
prefs = []
for code, name in PREFS.items():
    g = gpd.read_file(f"zip://{HZ}/raw_n03/N03-20250101_{code}_GML.zip!N03-20250101_{code}.shp")
    prefs.append(gpd.GeoDataFrame({"code": [code], "name": [name]},
                                  geometry=[g.to_crs(CRS).geometry.union_all()], crs=CRS))
prefs = pd.concat(prefs, ignore_index=True)
xmin, ymin, xmax, ymax = prefs.total_bounds
x0, y0 = np.floor(xmin / CELL) * CELL, np.floor(ymin / CELL) * CELL
NC, NR = int(np.ceil((xmax - x0) / CELL)), int(np.ceil((ymax - y0) / CELL))
T500 = from_origin(x0, y0 + NR * CELL, CELL, CELL)       # 行 0 が北
T25 = from_origin(x0, y0 + NR * CELL, PIX, PIX)
log("grid", NR, NC)

# マスの中心がどの県か（0＝海・県外）
pref_id = rasterize(((geom, int(c)) for geom, c in zip(prefs.geometry, prefs.code)),
                    out_shape=(NR, NC), transform=T500, fill=0, dtype="uint8")

# --- L2 の区域を 25m 画素に焼き、マスの割合にする ---
hz = np.zeros((NR * SUB, NC * SUB), dtype="uint8")
n_poly = 0
for z in HZ_ZIPS:
    with zipfile.ZipFile(HZ / f"raw/A31a-25_{z}_GEOJSON.zip") as zf:
        names = [i for i in zf.infolist() if i.filename.replace("\\", "/").split("/")[-1].startswith("A31a-20-")]
        geoms = []
        for i in names:
            g = gpd.read_file(io.BytesIO(zf.read(i)))
            geoms.append(g.to_crs(CRS).geometry)
    if geoms:
        s = pd.concat(geoms)
        n_poly += len(s)
        rasterize(((geom, 1) for geom in s), out=hz, transform=T25)
    log(z, len(names), "files")
frac = hz.reshape(NR, SUB, NC, SUB).sum(axis=(1, 3), dtype=np.uint32) / (SUB * SUB)
del hz
log("polygons", n_poly)

# --- 一級・二級河川で L2 未計算のもの ---
lst = pd.read_excel(HZ / "A31_R7_datalist.xlsx", sheet_name="想定最大規模", dtype=str)
lst.columns = [str(c).replace("\n", "") for c in lst.columns]
lst = lst[lst[lst.columns[13]].notna()]
done = set(lst["河川コード"].str.strip().str.zfill(10))
riv = []
for code in PREFS:
    g = gpd.read_file(f"zip://{HZ}/raw_w05/W05-07_{code}_GML.zip!W05-07_{code}-g_Stream.shp", encoding="cp932")
    riv.append(g.set_crs(4612).to_crs(CRS).assign(pref=code))
riv = pd.concat(riv, ignore_index=True)
riv["rc"] = riv["W05_002"].astype(str).str.strip().str.zfill(10)
riv["kubun"] = pd.to_numeric(riv["W05_003"], errors="coerce").fillna(0).astype(int)
riv["legal"] = riv["kubun"].isin(LEGAL) & ~riv["rc"].str.endswith("0000")
riv["done"] = riv["rc"].isin(done)
riv["km"] = riv.length / 1000
undone = riv[riv["legal"] & ~riv["done"]]
log("legal rivers", riv.loc[riv.legal, "rc"].nunique(), "undone", undone["rc"].nunique())

near = {}
for d in DISTS:
    buf = undone.geometry.buffer(d * 1000)
    near[d] = rasterize(((geom, 1) for geom in buf), out_shape=(NR, NC), transform=T500,
                        fill=0, all_touched=True, dtype="uint8").astype(bool)

# --- マスの表 ---
rr, cc = np.nonzero(pref_id)
cells = pd.DataFrame({"row": rr, "col": cc,
                      "x": x0 + (cc + 0.5) * CELL, "y": y0 + (NR - rr - 0.5) * CELL,
                      "pref": pd.Series(pref_id[rr, cc]).astype(str).map(PREFS).values,
                      "frac_l2": frac[rr, cc]})
for d in DISTS:
    cells[f"near_undone_{d}km"] = near[d][rr, cc]
    lab = np.where(cells["frac_l2"] >= 0.5, "1", np.where(cells[f"near_undone_{d}km"], "使わない", "0"))
    cells[f"label_{d}km"] = lab
cells.to_parquet(OUT / "H3_cells.parquet", index=False)

with open(OUT / "H3_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"マス {CELL:.0f}m、EPSG:{CRS}。陸のマス {len(cells)}。L2 ポリゴン {n_poly}\n")
    f.write(f"一級・二級河川（個別コード）{riv.loc[riv.legal, 'rc'].nunique()} 本のうち L2 未計算 {undone['rc'].nunique()} 本\n")
    for d in DISTS:
        t = cells.groupby("pref")[f"label_{d}km"].value_counts().unstack(fill_value=0)
        t = t[["1", "0", "使わない"]]
        t["使える割合"] = ((t["1"] + t["0"]) / t.sum(axis=1)).round(3)
        t["使える中の1の割合"] = (t["1"] / (t["1"] + t["0"])).round(3)
        f.write(f"\n=== D = {d} km ===\n{t.to_string()}\n")
    u = undone.groupby(["pref", "rc"]).agg(name=("W05_004", "first"), kubun=("kubun", "first"),
                                          km=("km", "sum"))
    f.write("\n=== L2 未計算の一級・二級河川（長い順、県ごと上位 10）===\n")
    for p_, g_ in u.reset_index().groupby("pref"):
        f.write(f"{PREFS[p_]}\n{g_.sort_values('km', ascending=False).head(10)[['rc', 'name', 'kubun', 'km']].round(1).to_string(index=False)}\n")
log(open(OUT / "H3_summary.txt", encoding="utf-8").read())

# --- 地図（D = 2km）---
img = np.full((NR, NC), np.nan)
code = {"1": 2, "0": 1, "使わない": 0}
img[cells["row"], cells["col"]] = cells["label_2km"].map(code).values
cols = ["#a39e93", "#e6e4dc", "#2a78d6"]
fig, ax = plt.subplots(figsize=(9, 10.5), facecolor="#fcfcfb")
ext = (x0, x0 + NC * CELL, y0, y0 + NR * CELL)
ax.imshow(img, extent=ext, cmap=ListedColormap(cols), vmin=-0.5, vmax=2.5, interpolation="nearest")
undone.plot(ax=ax, color="#d03b3b", linewidth=0.6)
prefs.boundary.plot(ax=ax, color="#52514e", linewidth=0.5)
for _, r in prefs.iterrows():
    c = r.geometry.representative_point()
    ax.text(c.x, c.y, r["name"], fontsize=12, fontweight="bold", color="#0b0b0b",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.8))
ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.legend(handles=[Patch(color=cols[2], label="1（L2 がマスの半分以上）"),
                   Patch(color=cols[1], label="0（近くの一級・二級河川はすべて計算済み）"),
                   Patch(color=cols[0], label="使わない（2km 以内に未計算の川）"),
                   Line2D([], [], color="#d03b3b", lw=1.5, label="L2 未計算の一級・二級河川")],
          loc="lower left", frameon=True, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig_H3_eligibility.png", dpi=130, facecolor="#fcfcfb")
log("done")
