"""
結果の章（論文/第5章_結果_下書き.md）の図と表

既存の評価結果（10km 外側ブロック・内部CV 5km、5シード）の out-of-fold 予測を読み直し、
ブロック・ブートストラップでモデル間の差を再計算する。モデルの再学習はしない。
A（・D0）の予測は全ファイルで同一（シード・分割が同じ）なので、ファイルをまたいで比べられる。
ブートストラップは1組あたり数分かかるため、結果を results_figs/boot_cache.parquet に保存し、
計算済みの組は再利用する（予測ファイルを作り直したらキャッシュを消すこと）。

出力: output/results_figs/
  fig_r1_scores.png   … モデル別 PR-AUC・F1（spatial）。シード間 SD のエラーバー
  fig_r2_diff.png     … TDA モデルの対 A の差と 95% CI（spatial）
  fig_r3_modelD.png   … 土地利用・地形分類（D0 − A）と、D0 への TDA の上乗せ（spatial）
  results_table.txt   … 上の数値（spatial と sp_event）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import eval_common as ec  # noqa: E402

OUT = ec.OUT_DIR / "results_figs"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "boot_cache.parquet"

# (予測ファイル, ファイル内のモデル名 → 論文での名前)。基準モデル A・D0 は最初に出てくるファイルから取る
SOURCES = [
    ("robust_b10_in5_preds", {"地形12": "A", "地形12+TDA": "A+PI"}),
    ("compare_A_Cp_b10_in5_preds", {"C'": "C'"}),
    ("compare_A_C_B_b10_in5_preds", {"C": "C", "B": "B"}),
    ("compare_A_APIb_b10_in5_preds", {"A+PIb": "A+PIb"}),
    ("compare_A_APIl_b10_in5_preds", {"A+PIl": "A+PIl"}),
    ("compare_A_D0_Dp_b10_in5_preds", {"D0": "D0", "D'": "D'"}),
    ("compare_D0_D_b10_in5_preds", {"D": "D"}),
]
# 基準モデルの予測が一致することを確かめるファイル（モデル名, ファイル）
SAME_BASE = [("A", "compare_A_C_B_b10_in5_preds"), ("D0", "compare_D0_D_b10_in5_preds")]

TDA_MODELS = ["A+PI", "A+PIb", "A+PIl", "C'", "C", "B"]
ORDER = ["A"] + TDA_MODELS + ["D0", "D'", "D"]
LABEL = {
    "A": "A　従来地形12",
    "A+PI": "A+PI　旧PI（500m単独）",
    "A+PIb": "A+PIb　バッファ付きPI",
    "A+PIl": "A+PIl　同（birth=log標高）",
    "C'": "C'　A＋窪地特徴",
    "C": "C　A＋窪地＋旧PI",
    "B": "B　窪地＋旧PI（従来指標なし）",
    "D0": "D0　A＋土地利用・地形分類",
    "D'": "D'　D0＋窪地特徴",
    "D": "D　D0＋バッファ付きPI",
}
# 直接比較（基準, 比較）
D_PAIRS = [("A", "D0"), ("D0", "D'"), ("D0", "D")]
PAIRS = ([("A", m) for m in TDA_MODELS]
         + [("A+PI", "A+PIb"), ("A+PI", "C"), ("C'", "C")] + D_PAIRS)

BLUE, ORANGE, GRAY, INK2 = "#2a78d6", "#eb6834", "#b4b2ab", "#52514e"
plt.rcParams.update({
    "font.family": "Yu Gothic", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})


def load_preds() -> pd.DataFrame:
    parts, seen = [], set()
    for f, rename in SOURCES:
        p = pd.read_parquet(ec.OUT_DIR / f"{f}.parquet")
        p["model"] = p["model"].replace(rename)
        keep = (set(rename.values()) | {"A", "D0"}) - seen
        p = p[p["model"].isin(keep)]
        seen |= set(p["model"].unique())
        parts.append(p)
    preds = pd.concat(parts, ignore_index=True)
    # 基準モデルの予測がファイル間で一致することの確認（一致しなければ比較の前提が崩れる）
    key = ["setting", "direction", "seed", "mesh_id"]
    for m, f in SAME_BASE:
        a = pd.read_parquet(ec.OUT_DIR / f"{f}.parquet")
        a = a[a["model"] == m].sort_values(key)
        b = preds[preds["model"] == m].sort_values(key)
        assert np.allclose(a["p"].values, b["p"].values), f"{m} の予測がファイル間で一致しない"
    return preds


def scores(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (s, dr, m, sd), g in preds.groupby(["setting", "direction", "model", "seed"]):
        rows.append({"setting": s, "direction": dr, "model": m, "seed": sd,
                     "pr_auc": ec.average_precision_score(g["y"], g["p"]),
                     "f1": ec.f1_score(g["y"], g["pred"], zero_division=0)})
    r = pd.DataFrame(rows)
    return (r.groupby(["setting", "direction", "model"])[["pr_auc", "f1"]]
             .agg(["mean", "std"]).reset_index())


def bootstrap_all(preds: pd.DataFrame) -> pd.DataFrame:
    cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["pair"])
    new = []
    for base, plus in PAIRS:
        pair = f"{plus} − {base}"
        if pair in set(cache["pair"]):
            continue
        bb = ec.block_bootstrap(preds, base, plus)
        bb.insert(0, "pair", pair)
        new.append(bb)
        print(f"{pair} 完了", flush=True)
    if new:
        cache = pd.concat([cache] + new, ignore_index=True)
        cache.to_parquet(CACHE, index=False)
    order = {f"{p} − {b}": i for i, (b, p) in enumerate(PAIRS)}
    out = cache[cache["pair"].isin(order)].copy()
    return out.sort_values("pair", key=lambda s: s.map(order), kind="stable")


def forest(diffs: pd.DataFrame, pairs: list[tuple[str, str]], labels: list[str],
           xlabel: str, title: str, fname: str) -> None:
    """spatial の差と 95% CI。区間が 0 を含まないものを橙で示す"""
    d = diffs[diffs["setting"] == "spatial"]
    names = [f"{p} − {b}" for b, p in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(pairs) + 1.4), sharey=True)
    y = np.arange(len(pairs))[::-1]
    for ax, metric in zip(axes, ["PR-AUC", "F1"]):
        t = d[d["metric"] == metric].set_index("pair").loc[names]
        # block_bootstrap は表示用の文字列で返す（"+0.055"、"[+0.004, +0.096]"）
        ci = t["95%CI"].str.strip("[]").str.split(",", expand=True).astype(float)
        lo, hi = ci[0].values, ci[1].values
        v = t["diff"].astype(float).values
        for yi, vi, l, h in zip(y, v, lo, hi):
            c = ORANGE if (l > 0 or h < 0) else BLUE
            ax.plot([l, h], [yi, yi], color=c, lw=2)
            ax.plot(vi, yi, "o", color=c, ms=6)
            ax.text(h + 0.004, yi, f"{vi:+.3f}", va="center", fontsize=9, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"{metric} の{xlabel}")
    axes[0].set_yticks(y, labels)
    fig.suptitle(title + "　橙＝区間が0を含まない", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    preds = load_preds()
    sc = scores(preds)
    diffs = bootstrap_all(preds)

    with open(OUT / "results_table.txt", "w", encoding="utf-8") as f:
        f.write("10km 外側ブロック・内部CV 5km、5シード、ブロック・ブートストラップ1000回\n\n")
        f.write("== 指標（平均±シード間SD）==\n")
        f.write(sc.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n\n")
        f.write("== 差と 95% CI ==\n")
        f.write(diffs.to_string(index=False) + "\n")
    print(diffs.to_string(index=False))

    # ---- 図1：spatial のモデル別スコア ----
    s = sc[sc["setting"] == "spatial"].set_index("model").loc[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, (col, name) in zip(axes, [("pr_auc", "PR-AUC"), ("f1", "F1")]):
        colors = [GRAY if m in ("A", "D0") else BLUE for m in ORDER]
        ax.barh(y, s[(col, "mean")], xerr=s[(col, "std")], color=colors,
                error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
        ax.axvline(s.loc["A", (col, "mean")], color=INK2, lw=0.8, ls="--")
        for yi, v in zip(y, s[(col, "mean")]):
            ax.text(v + 0.005, yi + 0.25, f"{v:.3f}", fontsize=9, color=INK2)
        ax.set_xlabel(name)
        ax.set_xlim(0, max(s[(col, "mean")]) * 1.25)
    axes[0].set_yticks(y, [LABEL[m] for m in ORDER])
    fig.suptitle("モデル別の性能（空間ブロック CV 10km、5シード平均±SD。灰＝基準モデル）",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r1_scores.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 図2：TDA モデルの対 A の差 ----
    forest(diffs, [("A", m) for m in TDA_MODELS], [LABEL[m] for m in TDA_MODELS],
           "差（対 A）", "従来地形（A）への上乗せ と 95% 信頼区間（ブロック・ブートストラップ）",
           "fig_r2_diff.png")

    # ---- 図3：Model D ----
    forest(diffs, D_PAIRS,
           ["D0 − A　土地利用・地形分類", "D' − D0　＋窪地特徴", "D − D0　＋バッファ付きPI"],
           "差", "土地利用・地形分類の効果と、その後の TDA の上乗せ（95% 信頼区間）",
           "fig_r3_modelD.png")
    print("saved", OUT)


if __name__ == "__main__":
    main()
