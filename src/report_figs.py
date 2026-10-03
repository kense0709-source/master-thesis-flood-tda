"""
進捗報告用の図（2026-10 報告）

数値の出典はすべて output/step0_results.md。
地図（図2・図5）は mesh_labels / mesh_500m / dem_utm_route / flowacc_km2 から再計算する。

出力: output/report_figs/fig1〜fig5_*.png（16:9、スライド貼り付け用）
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 体裁（dataviz 参照パレット）----
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
GRAY, INK, INK2, GRID = "#b4b2ab", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "Yu Gothic",
    "font.size": 12,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "figure.dpi": 100,
})
FIGSIZE = (12.8, 7.2)
DPI = 150


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def hbar(ax, labels, values, colors, fmt="{:.3f}"):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6, edgecolor="white", linewidth=2)
    ax.set_yticks(y, labels)
    for yi, v in zip(y, values):
        ax.text(v - 0.01, yi, fmt.format(v), va="center", ha="right", color="white",
                fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", length=0, labelsize=13)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)


# =====================================================================
# 図1 コピーベースライン（step0_results.md §5）
# =====================================================================
def fig1():
    labels = ["旧 Model A（以前の報告）", "旧 Model B（TDAあり・以前の報告）",
              "組み直したモデル（地形12＋TDA）", "コピー（2020の浸水域をそのまま使う）"]
    f1 = [0.249, 0.340, 0.630, 0.625]
    colors = [GRAY, GRAY, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    hbar(ax, labels, f1, colors)
    ax.axvline(0.625, color=ORANGE, ls="--", lw=1.5)
    ax.set_xlim(0, 0.8)
    ax.set_xlabel("F1（2020で学習 → 2023を予測）")
    ax.set_title("何も学習しない「コピー」に、以前のモデルは大差で負けていた")
    fig.text(0.01, -0.02,
             "コピー: 2020年に浸水したメッシュを2023年も浸水と予測するだけ。2023浸水の64.5%が2020でも浸水していたため強い。\n"
             "→ 以前報告した Model A 0.859 / Model B 0.928（ROC-AUC）は取り下げ、今後はコピーを比較対象として必ず併記する。",
             color=INK2, fontsize=11)
    save(fig, "fig1_copy_baseline.png")


# =====================================================================
# 図2 2020と2023の浸水域の重なり（地図）
# =====================================================================
def hillshade(dem, res, az=315, alt=45):
    dy, dx = np.gradient(dem, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    a, z = np.radians(az), np.radians(90 - alt)
    hs = np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)
    return np.clip(hs, 0, 1)


def load_dem(step=3):
    with rasterio.open(ROOT / "output" / "dem_utm_route.tif") as r:
        dem = r.read(1)[::step, ::step].astype("float32")
        b, res = r.bounds, r.res[0] * step
    valid = dem != -9999
    filled = np.where(valid, dem, np.nan)
    hs = hillshade(np.where(valid, dem, np.nanmean(filled)), res)
    hs = np.where(valid, hs, np.nan)
    extent = (b.left, b.right, b.bottom, b.top)
    return dem, valid, hs, extent


def km_axes(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig2():
    mesh = gpd.read_file(ROOT / "output" / "mesh_500m.gpkg")
    lab = pd.read_parquet(ROOT / "output" / "mesh_labels.parquet")
    w = lab.pivot(index="mesh_id", columns="event", values="label")
    n20, n23 = int(w["2020"].sum()), int(w["2023"].sum())
    both = int(((w["2020"] == 1) & (w["2023"] == 1)).sum())
    copy_f1 = 2 * both / (n20 + n23)
    print(f"2020={n20} 2023={n23} both={both} ({both/n23:.1%}) copyF1={copy_f1:.3f}")

    cat = pd.Series("none", index=w.index)
    cat[(w["2020"] == 1) & (w["2023"] == 0)] = "2020のみ"
    cat[(w["2020"] == 0) & (w["2023"] == 1)] = "2023のみ"
    cat[(w["2020"] == 1) & (w["2023"] == 1)] = "両年とも"
    mesh = mesh.merge(cat.rename("cat"), left_on="mesh_id", right_index=True)

    _, _, hs, extent = load_dem()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(hs, extent=extent, cmap="Greys_r", vmin=-0.3, vmax=1.3, alpha=0.6)
    spec = [("両年とも", AQUA, both), ("2020のみ", BLUE, n20 - both), ("2023のみ", ORANGE, n23 - both)]
    for name, col, _ in spec:
        sub = mesh[mesh["cat"] == name]
        sub.plot(ax=ax, color=col, edgecolor="white", linewidth=0.3)
    fb = mesh[mesh["cat"] != "none"].total_bounds
    pad = 4000
    km_axes(ax, (fb[0] - pad, fb[2] + pad, fb[1] - pad, fb[3] + pad))
    ax.legend(handles=[Patch(color=c, label=f"{n}（{k}メッシュ）") for n, c, k in spec],
              loc="lower right", frameon=True, fontsize=12)
    ax.set_title(f"2023年の浸水メッシュの{both/n23:.1%}は、2020年にも浸水していた")
    fig.text(0.01, 0.0,
             f"500mメッシュ単位。2020: {n20}メッシュ / 2023: {n23}メッシュ / 両年: {both}メッシュ"
             f"（コピーで F1 = {copy_f1:.3f}）。浸水域周辺を拡大、背景は陰影起伏図。",
             color=INK2, fontsize=11)
    save(fig, "fig2_overlap_map.png")


# =====================================================================
# 図3 特徴量グループを足したときの PR-AUC（step0_results.md §4.3）
# =====================================================================
def fig3():
    base = 0.508
    labels = ["＋ 標高統計", "＋ TWI・傾斜", "＋ 雨量（解析雨量）", "＋ 衛星（NDVI・MNDWI）"]
    vals = [0.577, 0.490, 0.284, 0.166]
    colors = [BLUE if v > base else RED for v in vals]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    hbar(ax, labels, vals, colors)
    ax.axvline(base, color=INK, lw=1.5)
    ax.text(base, 3.55, f" 窪地特徴5つだけ: {base:.3f}", color=INK, fontsize=12, va="bottom")
    for yi, v in zip(np.arange(4)[::-1], vals):
        d = v - base
        ax.text(0.72, yi, f"{d:+.3f}", va="center", ha="right",
                color=BLUE if d > 0 else RED, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlabel("PR-AUC（2020で学習 → 2023を予測）")
    ax.set_title("雨量・衛星データを足すと、モデルが大きく悪化する")
    fig.text(0.01, -0.02,
             "原因: 雨量・衛星はイベントごとに分布が違う（例: MNDWI平均 2020年 0.151 / 2023年 −0.011）。z標準化・上流積算雨量でも改善せず。\n"
             "2イベントだけでは「雨の降り方」と「地形」の影響を切り分けられない。",
             color=INK2, fontsize=11)
    save(fig, "fig3_feature_groups.png")


# =====================================================================
# 図4 強いベースライン上での TDA の上乗せ（step0_results.md §4.4）
# =====================================================================
def fig4():
    metrics = ["ROC-AUC", "PR-AUC", "F1"]
    data = {  # (地形12, +TDA812)
        "2020→2023": [(0.965, 0.966), (0.552, 0.587), (0.630, 0.630)],
        "2023→2020": [(0.961, 0.963), (0.548, 0.549), (0.633, 0.636)],
    }
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True)
    x = np.arange(2)
    wdt = 0.36
    for i, (ax, m) in enumerate(zip(axes, metrics)):
        a = [data[k][i][0] for k in data]
        b = [data[k][i][1] for k in data]
        ax.bar(x - wdt / 2, a, wdt, color=GRAY, edgecolor="white", linewidth=2, label="地形のみ（12次元）")
        ax.bar(x + wdt / 2, b, wdt, color=BLUE, edgecolor="white", linewidth=2, label="地形 ＋ TDA（812次元）")
        for xi, (va, vb) in enumerate(zip(a, b)):
            ax.text(xi - wdt / 2, va + 0.015, f"{va:.3f}", ha="center", fontsize=10, color=INK2)
            ax.text(xi + wdt / 2, vb + 0.015, f"{vb:.3f}", ha="center", fontsize=10, color=INK)
            ax.text(xi, 1.08, f"{vb - va:+.3f}", ha="center", fontsize=13, fontweight="bold", color=INK)
        ax.set_xticks(x, list(data))
        ax.set_ylim(0, 1.15)
        ax.set_title(m, loc="center", fontsize=13)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    axes[0].legend(loc="lower left", fontsize=11, frameon=True)
    fig.suptitle("ベースラインを強くすると、TDA（Persistence Image 800次元）を足してもほとんど変わらない",
                 x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.01, -0.03,
             "太字は TDA を足したときの差。PR-AUC が上がったのは 2020→2023 の一方向だけ（+0.035）で、逆方向では +0.001。\n"
             "未実装: 窪地ごとの体積・入れ子の段数（階層情報）、パッチ境界の修正 → これらを入れてから最終判断する。",
             color=INK2, fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_tda_gain.png")


# =====================================================================
# 図5 DEM が流域を覆っていない
# =====================================================================
def fig5():
    dem, valid, hs, extent = load_dem()
    with rasterio.open(ROOT / "output" / "flowacc_km2.tif") as r:
        acc = r.read(1)
        tr = r.transform
    acc = np.where(acc == -9999, np.nan, acc)
    iy, ix = np.unravel_index(np.nanargmax(acc), acc.shape)
    px, py = rasterio.transform.xy(tr, iy, ix)
    amax = float(np.nanmax(acc))
    s = acc >= 20  # 集水面積20km²以上を河道として表示（縮小時に途切れないようブロック最大）
    h, w_ = (s.shape[0] // 3) * 3, (s.shape[1] // 3) * 3
    stream = s[:h, :w_].reshape(h // 3, 3, w_ // 3, 3).any(axis=(1, 3))
    stream = np.maximum.reduce([stream, np.roll(stream, 1, 0), np.roll(stream, 1, 1)])

    t = Transformer.from_crs("EPSG:6668", "EPSG:32653", always_xy=True)
    lat = np.linspace(32.9167, 33.5, 50)
    ex, ey = t.transform(np.full_like(lat, 131.0), lat)

    fig, (ax, axb) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [2.2, 1]})
    ax.imshow(np.where(valid, 1.0, np.nan), extent=extent, cmap="Greys", vmin=0, vmax=8)
    ax.imshow(hs, extent=extent, cmap="Greys_r", vmin=-0.3, vmax=1.3, alpha=0.7)
    sext = (extent[0], extent[0] + stream.shape[1] * (extent[1] - extent[0]) / dem.shape[1],
            extent[3] - stream.shape[0] * (extent[3] - extent[2]) / dem.shape[0], extent[3])
    ax.imshow(np.where(stream, 1.0, np.nan), extent=sext, cmap="Blues", vmin=0, vmax=1.0,
              interpolation="nearest")
    ax.plot(ex, ey, color=RED, lw=3)
    ax.text(ex[-1] - 1500, ey[-1] - 3000, "DEMの東端（東経131.00°）\nここで切れている",
            color=RED, ha="right", va="top", fontsize=12, fontweight="bold")
    ax.annotate("上流（日田・玖珠方面）は\nこの外側", xy=(ex[25] + 500, ey[25]),
                xytext=(ex[25] - 26000, ey[25] - 24000), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.plot(px, py, "o", ms=10, color=ORANGE, mec="white", mew=2)
    ax.text(px + 1500, py + 1500, f"集水面積が最大の地点\n{amax:,.0f} km²", fontsize=11, color=INK)
    km_axes(ax, extent)
    ax.set_title("DEMが流域界ではなく矩形で切れている")

    names = ["DEM内で1点に\n集まる最大面積", "DEMの有効範囲", "筑後川流域"]
    vals = [amax, 2513, 2860]
    cols = [ORANGE, GRAY, BLUE]
    axb.bar(range(3), vals, color=cols, width=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        axb.text(i, v + 50, f"{v:,.0f}", ha="center", fontsize=13, color=INK)
    axb.set_xticks(range(3), names, fontsize=11)
    axb.set_ylabel("面積 [km²]")
    axb.set_ylim(0, 3300)
    axb.tick_params(axis="x", length=0)
    axb.grid(axis="y", color=GRID); axb.set_axisbelow(True)
    axb.set_title(f"流域の約{amax/2860:.0%}しか\n見えていない", fontsize=13)

    fig.text(0.01, -0.02,
             "2020・2023の浸水を起こした水は DEM の外（上流）に降った雨 → これまでの雨量特徴の評価はそもそも成立していなかった。\n"
             "対応案: 国土数値情報W07で流域界を確定し、DEM10Bで流域全体を整備して流向・集水面積を再計算する。",
             color=INK2, fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_dem_coverage.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
