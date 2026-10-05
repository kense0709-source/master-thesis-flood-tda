"""
Step 7: Persistence Image の逆解析（Obayashi, Hiraoka & Kimura 2018）

問い: PI のどの領域（どの標高で生まれ、どれだけ深い窪地／ループか）が、
      従来の地形指標で説明できない浸水の情報を持っているか。

方法:
  1. 線形モデル（L2 ロジスティック回帰）に 従来地形12（標準化）＋ PI を入れる
     → PI の重みは「地形12 では説明できない分」を表す
     PI は全画素共通の1つの値で割るだけにし、画素ごとの標準化はしない
     （重みを画素ごとに比べられる＝重みベクトルをそのまま birth×persistence 平面の「双対 PD」として読める）
  2. 正則化の強さ C は 10km 空間ブロック CV の PR-AUC で選ぶ（内部は 5km ブロック）
  3. 外側5分割 × シード5 = 25 回当てはめ、重みの平均と符号の一致率を出す
  4. 重みを 20×20 の画像（横=birth 標高、縦=persistence）に戻して図示
  5. メッシュごとの PI の寄与（Σ 重み×PI）を地図にする

PI の座標:
  buffered（既定）… 08b_tda_buffered.py が保存した範囲（output/tda_buffered_imager.json）
  old             … 08 と同じ手順（最初の300メッシュで fit）で Imager を作り直す。
                    08 の実行が途中再開されていた場合は一致しない可能性がある（注意書きを出す）
  persim の画像は [birth のビン, persistence のビン] の並び（確認済み）。
  old は 08 で 20×20 に引き伸ばしている。buffered は [0,1] に正規化して 20×20 にそのまま収めている。

実行:
  python -u src/13_inverse_analysis.py [--pi buffered|old]

出力:
  output/inverse_{pi}_weights.png / _map.png / .txt / _weights.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import eval_common as ec  # noqa: E402

IMG = 20
C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
INK, INK_SUB, GRID, SURF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"
# 発散配色：青（浸水しにくい側）― 灰 ― 橙（浸水しやすい側）
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])


def imager_axes(pi: str) -> dict:
    """H0/H1 の birth・persistence の範囲を返す"""
    if pi == "buffered":
        # 08b が保存した範囲。persistence の最上段には上限超えのペアも入っている
        ax = json.loads((ec.OUT_DIR / "tda_buffered_imager.json").read_text(encoding="utf-8"))
        return {h: (a["b0"], a["b1"], 0.0, a["p1"]) for h, a in ax.items()}
    # old: 08 と同じ手順で作り直す
    import importlib.util
    import geopandas as gpd
    import rasterio
    spec = importlib.util.spec_from_file_location("tda08", Path(__file__).parent / "08_tda_features.py")
    t8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t8)
    mesh = gpd.read_file(t8.MESH_PATH)
    h0s, h1s = [], []
    with rasterio.open(t8.DEM_PATH) as src:
        for _, row in mesh.iloc[:t8.SAMPLE_N].iterrows():
            z = t8.preprocess_patch(t8.read_patch(src, row.geometry, mesh.crs.to_string(),
                                                  src.crs.to_string()))
            if z is None:
                continue
            h0, h1 = t8.compute_diagrams(z)
            if len(h0): h0s.append(h0)
            if len(h1): h1s.append(h1)
    out = {}
    for h, ds in (("H0", h0s), ("H1", h1s)):
        im = t8.fit_imager(ds, IMG)
        out[h] = (*im.birth_range, *im.pers_range)
    print("[注意] old の座標は Imager を作り直したもの。08 が途中再開されていた場合は一致しない")
    return out


def prepare(pi: str):
    d = ec.load_dataset(10.0, 5.0)
    st = ec.static_table(d)
    cols = ec.TDA_B if pi == "buffered" else ec.TDA
    A = st[ec.TERRAIN].fillna(0.0)
    A = (A - A.mean()) / A.std()
    P = st[cols].fillna(0.0)
    P = P / P.values.std()
    X = np.hstack([A.values, P.values])
    return st, X, len(ec.TERRAIN), cols


def fit_lr(X, y, C):
    return LogisticRegression(penalty="l2", C=C, class_weight="balanced",
                              max_iter=2000).fit(X, y)


def spatial_cv(st, X, y, C, seed, n_pi_off=None):
    """10km 外側5分割の OOF 予測（n_pi_off を与えると PI 列を除いた地形のみ）"""
    oof = np.zeros(len(y))
    W = []
    for tr, te in ec.spatial_folds(st["block"].values, y, seed, 5):
        Xt = X[:, :n_pi_off] if n_pi_off else X
        m = fit_lr(Xt[tr], y[tr], C)
        oof[te] = m.decision_function(Xt[te])
        W.append(m.coef_[0])
    return oof, W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", choices=["buffered", "old"], default="buffered")
    args = ap.parse_args()

    st, X, nA, cols = prepare(args.pi)
    y = st["label"].values
    lines = [f"===== PI の逆解析（{args.pi}）=====",
             "線形モデル: L2 ロジスティック回帰（class_weight=balanced）、入力 = 地形12（標準化）＋ PI",
             "評価: 10km 空間ブロック5分割（spatial、ラベル=いずれかで浸水）", ""]

    # C の選択（シード0の空間CV、地形12+PI の PR-AUC）
    lines.append("C の選択（PR-AUC、シード0）")
    best = None
    for C in C_GRID:
        oof, _ = spatial_cv(st, X, y, C, 0)
        pr = average_precision_score(y, oof)
        lines.append(f"  C={C:<6} PR {pr:.3f}")
        if best is None or pr > best[1]:
            best = (C, pr)
    C = best[0]
    lines += [f"→ C = {C}", ""]

    # 線形モデルでも PI が効くか（地形のみ vs 地形+PI）、25回の重み
    rows, W = [], []
    for seed in range(5):
        o_full, w = spatial_cv(st, X, y, C, seed)
        o_base, _ = spatial_cv(st, X, y, C, seed, n_pi_off=nA)
        W += w
        rows.append({"seed": seed,
                     "PR 地形12": average_precision_score(y, o_base),
                     "PR 地形12+PI": average_precision_score(y, o_full),
                     "ROC 地形12": roc_auc_score(y, o_base),
                     "ROC 地形12+PI": roc_auc_score(y, o_full)})
    r = pd.DataFrame(rows)
    lines += ["線形モデルでの比較（5シード平均、10km spatial）",
              r.drop(columns="seed").mean().round(3).to_string(), ""]

    W = np.array(W)                       # (25, nA + 800)
    wpi = W[:, nA:]
    mean = wpi.mean(0)
    agree = np.maximum((wpi > 0).mean(0), (wpi < 0).mean(0))   # 符号の一致率

    axes = imager_axes(args.pi)
    recs = []
    for h, off in (("H0", 0), ("H1", 400)):
        b0, b1, p0, p1 = axes[h]
        for k in range(400):
            ib, ip = divmod(k, IMG)
            recs.append({
                "dim": h, "ib": ib, "ip": ip,
                "birth_lo": b0 + ib * (b1 - b0) / IMG, "birth_hi": b0 + (ib + 1) * (b1 - b0) / IMG,
                "pers_lo": p0 + ip * (p1 - p0) / IMG, "pers_hi": p0 + (ip + 1) * (p1 - p0) / IMG,
                "weight": mean[off + k], "sign_agree": agree[off + k],
                "pi_mean_flood": st.loc[y == 1, cols[off + k]].mean(),
                "pi_mean_dry": st.loc[y == 0, cols[off + k]].mean(),
            })
    wt = pd.DataFrame(recs)
    wt.to_parquet(ec.OUT_DIR / f"inverse_{args.pi}_weights.parquet", index=False)

    # 地形12 の重み（参考）
    wa = pd.Series(W[:, :nA].mean(0), index=ec.TERRAIN).sort_values()
    lines += ["地形12 の重み（標準化後、25回平均）", wa.round(3).to_string(), ""]

    # 寄与の大きい画素（符号一致率 0.9 以上）
    stable = wt[wt["sign_agree"] >= 0.9].copy()
    show = ["dim", "birth_lo", "birth_hi", "pers_lo", "pers_hi", "weight", "sign_agree",
            "pi_mean_flood", "pi_mean_dry"]
    for title, part in (("浸水しやすい側（重み正）上位10", stable.nlargest(10, "weight")),
                        ("浸水しにくい側（重み負）上位10", stable.nsmallest(10, "weight"))):
        lines += [title, part[show].round(3).to_string(index=False), ""]
    lines.append(f"符号一致率 0.9 以上の画素: H0 {(stable.dim == 'H0').sum()} / 400、"
                 f"H1 {(stable.dim == 'H1').sum()} / 400")

    # H0・H1 それぞれの寄与の大きさ
    contrib = X[:, nA:] * mean
    c0, c1 = contrib[:, :400].sum(1), contrib[:, 400:].sum(1)
    lines += ["",
              f"PI の寄与（Σ重み×PI）の標準偏差: H0 {c0.std():.3f} / H1 {c1.std():.3f}",
              f"寄与と浸水の相関: H0 {np.corrcoef(c0, y)[0, 1]:.3f} / H1 {np.corrcoef(c1, y)[0, 1]:.3f}",
              f"寄与と地形12の主な指標の相関: "
              + ", ".join(f"{c} {np.corrcoef(c0 + c1, st[c].fillna(0))[0, 1]:+.2f}"
                          for c in ["elev_mean", "slope_mean", "twi_mean", "sink_area_frac"])]

    plot_weights(wt, axes, args.pi)
    plot_map(st, c0 + c1, y, args.pi)

    text = "\n".join(lines)
    (ec.OUT_DIR / f"inverse_{args.pi}.txt").write_text(text, encoding="utf-8")
    print(text)


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK_SUB, labelsize=9)


def _font():
    for font in ("Meiryo", "Yu Gothic", "MS Gothic"):
        if font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_weights(wt, axes, pi):
    _font()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(SURF)
    lim = np.abs(wt["weight"]).max()
    for ax, h in zip(axs, ("H0", "H1")):
        b0, b1, p0, p1 = axes[h]
        g = wt[wt.dim == h]
        img = np.zeros((IMG, IMG))
        msk = np.zeros((IMG, IMG), bool)
        img[g.ip, g.ib] = g.weight              # 縦=persistence、横=birth
        msk[g.ip, g.ib] = g.sign_agree < 0.9
        im = ax.imshow(img, origin="lower", cmap=DIVERGING,
                       norm=TwoSlopeNorm(0, -lim, lim),
                       extent=[b0, b1, p0, p1], aspect="auto")
        # 符号が安定しない画素は斜線で示す
        ys, xs = np.nonzero(msk)
        bw, pw = (b1 - b0) / IMG, (p1 - p0) / IMG
        for yy, xx in zip(ys, xs):
            ax.add_patch(plt.Rectangle((b0 + xx * bw, p0 + yy * pw), bw, pw, fill=False,
                                       hatch="///", edgecolor="#b8b7b2", linewidth=0))
        ax.set_xlabel("birth（生まれた標高）[m]", color=INK_SUB)
        ax.set_ylabel("persistence（深さ・顕著さ）[m]", color=INK_SUB)
        ax.set_title(f"{h} の重み", color=INK, loc="left", fontsize=11)
        _style(ax)
    cb = fig.colorbar(im, ax=axs, shrink=0.8)
    cb.ax.tick_params(colors=INK_SUB, labelsize=8)
    cb.set_label("ロジスティック回帰の重み（25回平均）", color=INK_SUB)
    fig.suptitle("PI の重み：橙＝浸水しやすい側、青＝しにくい側、斜線＝25回で符号が不安定（一致率<0.9）",
                 color=INK_SUB, fontsize=10, x=0.02, ha="left", y=1.0)
    fig.savefig(ec.OUT_DIR / f"inverse_{pi}_weights.png", dpi=150, facecolor=SURF,
                bbox_inches="tight")


def plot_map(st, contrib, y, pi):
    import geopandas as gpd
    _font()
    mesh = gpd.read_file(ec.OUT_DIR / "mesh_500m.gpkg", columns=["mesh_id", "mesh_row", "mesh_col"],
                         ignore_geometry=True)
    m = mesh.merge(pd.DataFrame({"mesh_id": st.mesh_id, "c": contrib, "y": y}), on="mesh_id")
    r0, c0 = m.mesh_row.min(), m.mesh_col.min()
    H, W = m.mesh_row.max() - r0 + 1, m.mesh_col.max() - c0 + 1
    C = np.full((H, W), np.nan)
    Y = np.full((H, W), np.nan)
    C[m.mesh_row - r0, m.mesh_col - c0] = m.c
    Y[m.mesh_row - r0, m.mesh_col - c0] = m.y
    # mesh_row は北ほど大きい（重心の y と相関 1.0 を確認）→ origin="lower" で北が上
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(SURF)
    lim = np.nanpercentile(np.abs(C), 98)
    im = axs[0].imshow(C, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest",
                       origin="lower")
    axs[0].contour((np.nan_to_num(Y) > 0.5).astype(float), levels=[0.5], colors=INK,
                   linewidths=0.6, origin="lower")
    axs[0].set_title("PI の寄与（橙=浸水側に押す、黒線=浸水メッシュの輪郭）", color=INK,
                     loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=axs[0], shrink=0.8)
    cb.ax.tick_params(colors=INK_SUB, labelsize=8)
    axs[1].imshow(np.where(np.isnan(Y), np.nan, Y), cmap=LinearSegmentedColormap.from_list(
        "lab", ["#f0efec", "#eb6834"]), interpolation="nearest", origin="lower")
    axs[1].set_title("浸水ラベル（2020・2023 のいずれか）", color=INK, loc="left", fontsize=10)
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        _style(ax)
    fig.savefig(ec.OUT_DIR / f"inverse_{pi}_map.png", dpi=150, facecolor=SURF, bbox_inches="tight")


if __name__ == "__main__":
    main()
