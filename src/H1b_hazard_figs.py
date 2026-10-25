"""Phase H 手順1 の説明用の図：筑後川（国管理）の洪水浸水想定区域を 500m マスに重ねる

入力：ハザードマップ/extract_89/（H1_inspect_hazard.py が取り出した GeoJSON）、output/mesh_500m.gpkg、
      output/mesh_labels.parquet（2020 浸水推定図）
出力：output/hazard/fig_*.png、output/hazard/H1b_summary.json、output/hazard/mesh_hazard_chikugo_national.parquet

ポリゴンを 25m 格子（マスの 1/20）に焼き直し、マスごとに 400 画素の平均で「浸水域の割合」を出す。
画素の中心で判定するので、H1_inspect_hazard.py の厳密な重ね合わせとは少しずれる。
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from rasterio.features import rasterize
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "ハザードマップ/extract_89"
OUT = ROOT / "output/hazard"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "Yu Gothic", "font.size": 10})
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASIN = "#ecebe6"
# 浸水深ランク 1〜5（一色の濃淡。浅い→深い）
RANK_COLORS = ["#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
RANK_LABELS = ["0〜0.5m", "0.5〜3m", "3〜5m", "5〜10m", "10〜20m"]

CELL = 500.0
SUB = 20                      # 1 マス = 20×20 画素（25m）
PIX = CELL / SUB

mesh = gpd.read_file(ROOT / "output/mesh_500m.gpkg")
x0 = mesh.total_bounds[0] - mesh["mesh_col"].min() * CELL   # mesh_col=0 の西端
y0 = mesh.total_bounds[1] - mesh["mesh_row"].min() * CELL   # mesh_row=0 の南端
NR, NC = int(mesh["mesh_row"].max()) + 1, int(mesh["mesh_col"].max()) + 1
H, W = NR * SUB, NC * SUB
transform = from_origin(x0, y0 + NR * CELL, PIX, PIX)       # 画像の上＝北
extent = (x0, x0 + NC * CELL, y0, y0 + NR * CELL)

basin = np.zeros((NR, NC), bool)
basin[mesh["mesh_row"].astype(int), mesh["mesh_col"].astype(int)] = True


def rank_raster(fname, col):
    g = gpd.read_file(EX / fname).to_crs(mesh.crs)
    shapes = ((geom, int(r)) for geom, r in zip(g.geometry, g[col]))
    return rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype="uint8"), g


def mesh_frac(r):
    # 画像は上が北。マスの行は南から数えるので上下を反転してから集計
    a = (r[::-1] > 0).reshape(NR, SUB, NC, SUB).mean(axis=(1, 3))
    return np.where(basin, a, np.nan)


r1, _ = rank_raster("A31a-10-25_89_8909060001_10.geojson", "A31a_105")
r2, g2 = rank_raster("A31a-20-25_89_8909060001_10.geojson", "A31a_205")
f1, f2 = mesh_frac(r1), mesh_frac(r2)

rows, cols = mesh["mesh_row"].astype(int).values, mesh["mesh_col"].astype(int).values
tab = pd.DataFrame({"mesh_id": mesh["mesh_id"], "frac_l1": f1[rows, cols], "frac_l2": f2[rows, cols]})
tab.to_parquet(OUT / "mesh_hazard_chikugo_national.parquet", index=False)

lab = pd.read_parquet(ROOT / "output/mesh_labels.parquet")
lab20 = lab[lab["event"].astype(str) == "2020"].set_index("mesh_id")["label"]
tab["y2020"] = tab["mesh_id"].map(lab20).fillna(0).astype(int)
act = np.full((NR, NC), np.nan)
act[rows, cols] = tab["y2020"].values

cmap_rank = ListedColormap(RANK_COLORS)
norm_rank = BoundaryNorm(np.arange(0.5, 6.5), 5)


def base_ax(ax, title):
    ax.imshow(np.where(basin, 1, np.nan), extent=extent, origin="lower",
              cmap=ListedColormap([BASIN]), interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def scalebar(ax):
    xa, ya = extent[0] + 3000, extent[2] + 3000
    ax.add_patch(Rectangle((xa, ya), 10000, 500, color=INK2))
    ax.text(xa + 5000, ya + 1200, "10 km", ha="center", fontsize=9, color=INK2)


# --- 図1：L1 と L2 の浸水深ランク（25m 画素のまま） ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURF)
for ax, r, t in [(axes[0], r1, "計画規模（L1）"), (axes[1], r2, "想定最大規模（L2）")]:
    base_ax(ax, t)
    ax.imshow(np.where(r > 0, r, np.nan), extent=extent, cmap=cmap_rank, norm=norm_rank,
              interpolation="nearest")
    scalebar(ax)
handles = [Rectangle((0, 0), 1, 1, color=c) for c in RANK_COLORS] + [Rectangle((0, 0), 1, 1, color=BASIN)]
fig.legend(handles, RANK_LABELS + ["流域（想定区域の外）"], loc="lower center", ncol=6, frameon=False,
           fontsize=9.5, labelcolor=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(OUT / "fig_l1_l2_map.png", dpi=130, facecolor=SURF)
plt.close(fig)

# --- 図2：拡大。25m の階段状の境界と 500m マスの割合 ---
mixed = ((f2 > 0.1) & (f2 < 0.9)).astype(float)
k = 6
best, bi, bj = -1, 0, 0
for i in range(NR - k):
    for j in range(NC - k):
        s = mixed[i:i + k, j:j + k].sum() + 0.3 * np.nan_to_num(f2[i:i + k, j:j + k]).sum()
        if s > best and basin[i:i + k, j:j + k].all():
            best, bi, bj = s, i, j
zx0, zy0 = x0 + bj * CELL, y0 + bi * CELL
fig, ax = plt.subplots(figsize=(6.4, 6.4), facecolor=SURF)
ax.set_facecolor(BASIN)
clip = g2.cx[zx0:zx0 + k * CELL, zy0:zy0 + k * CELL]
for rk in range(1, 6):
    sub = clip[clip["A31a_205"] == rk]
    if len(sub):
        sub.plot(ax=ax, color=RANK_COLORS[rk - 1], linewidth=0)
for t in range(k + 1):
    ax.axvline(zx0 + t * CELL, color=INK, lw=1.2)
    ax.axhline(zy0 + t * CELL, color=INK, lw=1.2)
for a in range(k):
    for b in range(k):
        v = f2[bi + a, bj + b]
        ax.text(zx0 + (b + 0.5) * CELL, zy0 + (a + 0.5) * CELL, f"{v:.2f}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
ax.set_xlim(zx0, zx0 + k * CELL); ax.set_ylim(zy0, zy0 + k * CELL)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT / "fig_zoom.png", dpi=130, facecolor=SURF)
plt.close(fig)
zc = gpd.GeoSeries(gpd.points_from_xy([zx0 + k * CELL / 2], [zy0 + k * CELL / 2]), crs=mesh.crs).to_crs(4326)

# --- 図3：マスの割合 → ラベル、2020 の実際の浸水と比べる ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=SURF)
base_ax(axes[0], "L2 がマスに占める割合")
cmf = ListedColormap(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
im = axes[0].imshow(np.where(f2 > 0, f2, np.nan), extent=extent, origin="lower", cmap=cmf,
                    norm=BoundaryNorm([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001], 6), interpolation="nearest")
cb = fig.colorbar(im, cax=axes[0].inset_axes([0.45, 0.06, 0.5, 0.035]), orientation="horizontal",
                  ticks=[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
cb.ax.tick_params(labelsize=8, colors=INK2); cb.outline.set_visible(False)
base_ax(axes[1], "ラベル案：割合 ≥ 0.5 を 1")
axes[1].imshow(np.where(f2 >= 0.5, 1, np.nan), extent=extent, origin="lower",
               cmap=ListedColormap(["#2a78d6"]), interpolation="nearest")
base_ax(axes[2], "参考：2020年の実際の浸水（浸水推定図）")
axes[2].imshow(np.where(act == 1, 1, np.nan), extent=extent, origin="lower",
               cmap=ListedColormap(["#eb6834"]), interpolation="nearest")
for ax in axes:
    scalebar(ax)
fig.tight_layout()
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "fig_frac_label.png", dpi=120, facecolor=SURF)
plt.close(fig)

# --- 数値（ページのグラフ用） ---
v2 = tab.loc[tab["frac_l2"] > 0, "frac_l2"].values
edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999, 1.0001]
hist = np.histogram(v2, bins=edges)[0].tolist()
n = len(tab)
y2 = tab["frac_l2"] >= 0.5
summary = {
    "n_mesh": n,
    "zoom_center_lonlat": [round(zc.x[0], 4), round(zc.y[0], 4)],
    "hist_edges": edges, "hist_counts_l2_touched": hist,
    "pos_rate": {
        "l1_any": float((tab["frac_l1"] > 0).mean()), "l1_half": float((tab["frac_l1"] >= 0.5).mean()),
        "l2_any": float((tab["frac_l2"] > 0).mean()), "l2_half": float(y2.mean()),
        "y2020": float(tab["y2020"].mean()),
    },
    "y2020_in_l2_half": float(y2[tab["y2020"] == 1].mean()),
    "y2020_in_l2_any": float((tab.loc[tab["y2020"] == 1, "frac_l2"] > 0).mean()),
    "l2_rank_km2": {int(k_): float(v) for k_, v in
                    g2.assign(a=g2.area / 1e6).groupby("A31a_205")["a"].sum().items()},
}
(OUT / "H1b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1))
