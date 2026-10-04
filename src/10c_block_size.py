"""
Step 0c: 空間ブロックの大きさを、空間自己相関が消える距離から決める

考え方（Roberts et al. 2017, Ecography 40: 913-929）:
  空間ブロック CV のブロックは、モデルの残差の空間自己相関が消える距離より大きくする。
  そうしないと、テストメッシュの隣（相関の強いメッシュ）が訓練に入り、点数が過大になる。
  2.5km と 5km で spatial の PR-AUC が 0.58 と 0.43 と大きく違ったのはこの効果と考えられる。

測るもの（距離ごとの自己相関＝コレログラム）:
  1. 浸水ラベル（いずれかで浸水）そのもの
  2. Model A（地形12）の残差  y − p   … 10b_robustness の 5km 版 out-of-fold 予測
  3. Model A+PI の残差
  4. 主な説明変数（elev_mean, twi_mean, sink_area_frac）… blockCV パッケージ流の参考値

自己相関の定義:
  標準化した値 z について、距離ビンごとに z_i·z_j の平均（Moran's I 型の相関）。
  500m 格子なので、行・列のずれ（オフセット）ごとに配列をずらして全ペアを計算する。

「消える距離」:
  相関が初めて 0.1 を下回る距離（主）、0.05 を下回る距離（参考）。

出力:
  output/block_size_correlogram.png
  output/block_size_results.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import eval_common as ec  # noqa: E402

OUT_PNG = ec.OUT_DIR / "block_size_correlogram.png"
OUT_TXT = ec.OUT_DIR / "block_size_results.txt"

MAX_KM = 25.0
BIN_KM = 0.5
CELL_KM = 0.5

# 配色（00_check_storage.py と同じ検証済み categorical の並び）
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_SUB, GRID, SURF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"


def load_grid_values() -> tuple[pd.DataFrame, dict]:
    """メッシュの行・列と、自己相関を測る値（列）を1つの表にまとめる"""
    mesh = gpd.read_file(ec.OUT_DIR / "mesh_500m.gpkg",
                         columns=["mesh_id", "mesh_row", "mesh_col"],
                         ignore_geometry=True)
    d = ec.load_dataset(5.0)
    st = ec.static_table(d)[["mesh_id", "label", "elev_mean", "twi_mean",
                             "sink_area_frac"]]
    df = mesh.merge(st, on="mesh_id", how="inner")

    # 5km・上限600本の spatial out-of-fold 予測（シード平均）から残差を作る
    pr = pd.read_parquet(ec.OUT_DIR / "robust_b5_preds.parquet")
    pr = pr[pr["setting"] == "spatial"]
    for model, col in [("地形12", "resid_A"), ("地形12+TDA", "resid_API")]:
        m = pr[pr["model"] == model].groupby("mesh_id").agg(p=("p", "mean"),
                                                             y=("y", "first"))
        df = df.merge((m["y"] - m["p"]).rename(col).reset_index(),
                      on="mesh_id", how="left")

    groups = {
        "主": {"label": "浸水ラベル", "resid_A": "残差 Model A",
               "resid_API": "残差 Model A+PI"},
        "参考": {"elev_mean": "標高平均", "twi_mean": "TWI平均",
                 "sink_area_frac": "窪地面積率"},
    }
    return df, groups


def correlogram(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """オフセットごとにずらして z_i·z_j を集計し、距離ビンごとの平均を返す"""
    r0, c0 = df["mesh_row"].min(), df["mesh_col"].min()
    nr = df["mesh_row"].max() - r0 + 1
    nc = df["mesh_col"].max() - c0 + 1
    v = df[col].values.astype(float)
    ok = ~np.isnan(v)
    z = (v - v[ok].mean()) / v[ok].std()

    G = np.full((nr, nc), np.nan)
    G[df["mesh_row"].values[ok] - r0, df["mesh_col"].values[ok] - c0] = z[ok]

    kmax = int(MAX_KM / CELL_KM)
    nb = int(MAX_KM / BIN_KM)
    s = np.zeros(nb)
    n = np.zeros(nb)
    # (dr, dc) と (−dr, −dc) は同じペアなので半平面だけ数える
    for dr in range(0, kmax + 1):
        for dc in range(-kmax, kmax + 1):
            if dr == 0 and dc <= 0:
                continue
            dist = CELL_KM * np.hypot(dr, dc)
            if dist > MAX_KM:
                continue
            # a[i, j] と b[i, j] が (dr, dc) だけ離れたセルの組になるよう切り出す
            if dc >= 0:
                a, b = G[:nr - dr, :nc - dc], G[dr:, dc:]
            else:
                a, b = G[:nr - dr, -dc:], G[dr:, :nc + dc]
            prod = a * b
            m = ~np.isnan(prod)
            k = min(int(dist / BIN_KM), nb - 1)
            s[k] += prod[m].sum()
            n[k] += m.sum()
    centers = (np.arange(nb) + 0.5) * BIN_KM
    cg = pd.DataFrame({"dist_km": centers, "corr": s / np.maximum(n, 1),
                       "n_pairs": n.astype(int)})
    # 最初のビン（0〜0.5km）は格子上にペアが存在しないので除く
    return cg[cg["n_pairs"] > 0].reset_index(drop=True)


def first_below(cg: pd.DataFrame, th: float) -> float:
    below = cg[(cg["n_pairs"] > 0) & (cg["corr"] < th)]
    return float(below["dist_km"].iloc[0]) if len(below) else np.nan


def make_figure(curves: dict, groups: dict) -> None:
    for font in ("Meiryo", "Yu Gothic", "MS Gothic"):
        if font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, (gname, cols) in zip(axes, groups.items()):
        ax.axhline(0, color=INK_SUB, linewidth=1)
        ax.axhline(0.1, color=GRID, linewidth=1.5, linestyle="--")
        for x in (2.5, 5, 10):
            ax.axvline(x, color=GRID, linewidth=1, zorder=0)
            ax.text(x, 1.02, f"{x:g}km", color=INK_SUB, fontsize=8,
                    ha="center", va="bottom", transform=ax.get_xaxis_transform())
        for i, (col, jp) in enumerate(cols.items()):
            cg = curves[col]
            ax.plot(cg["dist_km"], cg["corr"], color=SERIES_COLORS[i],
                    linewidth=2, label=jp)
        ax.set_xlim(0, MAX_KM)
        ax.set_xlabel("距離 [km]", color=INK_SUB)
        ax.set_title(f"({'a' if gname == '主' else 'b'}) "
                     f"{'浸水ラベルとモデル残差' if gname == '主' else '説明変数（参考）'}",
                     color=INK, loc="left", fontsize=11, pad=16)
        ax.legend(frameon=False, labelcolor=INK_SUB, fontsize=9, loc="upper right")
        ax.set_facecolor(SURF)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SUB, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.6)
    axes[0].set_ylabel("自己相関（破線 0.1）", color=INK_SUB)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURF)


def main() -> None:
    df, groups = load_grid_values()
    curves, rows = {}, []
    for gname, cols in groups.items():
        for col, jp in cols.items():
            cg = correlogram(df, col)
            curves[col] = cg
            rows.append({"区分": gname, "変数": jp,
                         "0.5km": round(cg["corr"].iloc[0], 3),
                         "2.5km": round(cg.loc[cg["dist_km"].sub(2.75).abs().idxmin(), "corr"], 3),
                         "5km": round(cg.loc[cg["dist_km"].sub(5.25).abs().idxmin(), "corr"], 3),
                         "10km": round(cg.loc[cg["dist_km"].sub(10.25).abs().idxmin(), "corr"], 3),
                         "<0.1 の距離[km]": first_below(cg, 0.1),
                         "<0.05 の距離[km]": first_below(cg, 0.05)})
    tab = pd.DataFrame(rows)
    make_figure(curves, groups)

    lines = ["=== Step 0c: 空間自己相関の距離 ===",
             f"距離ビン {BIN_KM} km、最大 {MAX_KM} km。残差は 5km・spatial の out-of-fold 予測（5シード平均）",
             "", tab.to_string(index=False), "",
             "距離ビン別（主の3変数）"]
    wide = pd.DataFrame({"dist_km": curves["label"]["dist_km"]})
    for col in groups["主"]:
        wide[col] = curves[col]["corr"].round(3)
    lines.append(wide.head(30).to_string(index=False))
    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n保存: {OUT_PNG}\n保存: {OUT_TXT}")


if __name__ == "__main__":
    main()
