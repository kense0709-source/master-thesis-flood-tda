"""
結果の章（論文/第5章_結果.md）の図と表

既存の評価結果（10km 外側ブロック・内部CV 5km、5シード）の out-of-fold 予測を読み直し、
ブロック・ブートストラップで「対 A の差」を再計算する。モデルの再学習はしない（数分）。
A の予測は全ファイルで同一（シード・分割が同じ）なので、ファイルをまたいで比べられる。

出力: output/results_figs/
  fig_r1_scores.png   … モデル別 PR-AUC・F1（spatial）。シード間 SD のエラーバー
  fig_r2_diff.png     … 対 A の差と 95% CI（spatial、PR-AUC・F1）
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

# (予測ファイル, ファイル内のモデル名 → 論文での名前)
SOURCES = [
    ("robust_b10_in5_preds", {"地形12": "A", "地形12+TDA": "A+PI"}),
    ("compare_A_Cp_b10_in5_preds", {"C'": "C'"}),
    ("compare_A_C_B_b10_in5_preds", {"C": "C", "B": "B"}),
    ("compare_A_APIb_b10_in5_preds", {"A+PIb": "A+PIb"}),
    ("compare_A_APIl_b10_in5_preds", {"A+PIl": "A+PIl"}),
]
ORDER = ["A", "A+PI", "A+PIb", "A+PIl", "C'", "C", "B"]
LABEL = {
    "A": "A　従来地形12",
    "A+PI": "A+PI　旧PI（500m単独）",
    "A+PIb": "A+PIb　バッファ付きPI",
    "A+PIl": "A+PIl　同（birth=log標高）",
    "C'": "C'　A＋窪地特徴",
    "C": "C　A＋窪地＋旧PI",
    "B": "B　窪地＋旧PI（従来指標なし）",
}
# 直接比較（基準, 比較）
PAIRS = [("A", m) for m in ORDER[1:]] + [("A+PI", "A+PIb"), ("A+PI", "C"), ("C'", "C")]

BLUE, ORANGE, GRAY, INK2 = "#2a78d6", "#eb6834", "#b4b2ab", "#52514e"
plt.rcParams.update({
    "font.family": "Yu Gothic", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
})


def load_preds() -> pd.DataFrame:
    parts = []
    for f, rename in SOURCES:
        p = pd.read_parquet(ec.OUT_DIR / f"{f}.parquet")
        p["model"] = p["model"].replace(rename)
        keep = set(rename.values()) | ({"A"} if f == SOURCES[0][0] else set())
        parts.append(p[p["model"].isin(keep)])
    preds = pd.concat(parts, ignore_index=True)
    # A の予測がファイル間で一致することの確認（一致しなければ比較の前提が崩れる）
    a = pd.read_parquet(ec.OUT_DIR / "compare_A_C_B_b10_in5_preds.parquet")
    a = a[a["model"] == "A"].sort_values(["setting", "direction", "seed", "mesh_id"])
    b = preds[preds["model"] == "A"].sort_values(["setting", "direction", "seed", "mesh_id"])
    assert np.allclose(a["p"].values, b["p"].values), "A の予測がファイル間で一致しない"
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


def main() -> None:
    preds = load_preds()
    sc = scores(preds)

    diffs = []
    for base, plus in PAIRS:
        bb = ec.block_bootstrap(preds, base, plus)
        bb.insert(0, "pair", f"{plus} − {base}")
        diffs.append(bb)
        print(f"{plus} − {base} 完了", flush=True)
    diffs = pd.concat(diffs, ignore_index=True)

    with open(OUT / "results_table.txt", "w", encoding="utf-8") as f:
        f.write("10km 外側ブロック・内部CV 5km、5シード、ブロック・ブートストラップ1000回\n\n")
        f.write("== 指標（平均±シード間SD）==\n")
        f.write(sc.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n\n")
        f.write("== 差と 95% CI ==\n")
        f.write(diffs.to_string(index=False) + "\n")
    print(diffs.to_string(index=False))

    # ---- 図1：spatial のモデル別スコア ----
    s = sc[sc["setting"] == "spatial"].set_index("model").loc[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for ax, (col, name) in zip(axes, [("pr_auc", "PR-AUC"), ("f1", "F1")]):
        colors = [GRAY if m == "A" else BLUE for m in ORDER]
        ax.barh(y, s[(col, "mean")], xerr=s[(col, "std")], color=colors,
                error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
        ax.axvline(s.loc["A", (col, "mean")], color=INK2, lw=0.8, ls="--")
        for yi, v in zip(y, s[(col, "mean")]):
            ax.text(v + 0.005, yi + 0.25, f"{v:.3f}", fontsize=9, color=INK2)
        ax.set_xlabel(name)
        ax.set_xlim(0, max(s[(col, "mean")]) * 1.25)
    axes[0].set_yticks(y, [LABEL[m] for m in ORDER])
    fig.suptitle("モデル別の性能（空間ブロック CV 10km、5シード平均±SD）", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r1_scores.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 図2：対 A の差と 95% CI（spatial）----
    d = diffs[(diffs["setting"] == "spatial")]
    pairs = [f"{m} − A" for m in ORDER[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    y = np.arange(len(pairs))[::-1]
    for ax, metric in zip(axes, ["PR-AUC", "F1"]):
        t = d[d["metric"] == metric].set_index("pair").loc[pairs]
        # block_bootstrap は表示用の文字列で返す（"+0.055"、"[+0.004, +0.096]"）
        ci = t["95%CI"].str.strip("[]").str.split(",", expand=True).astype(float)
        lo, hi = ci[0].values, ci[1].values
        v = t["diff"].astype(float).values
        sig = lo > 0
        for yi, vi, l, h, sg in zip(y, v, lo, hi, sig):
            c = ORANGE if sg else BLUE
            ax.plot([l, h], [yi, yi], color=c, lw=2)
            ax.plot(vi, yi, "o", color=c, ms=6)
            ax.text(h + 0.004, yi, f"{vi:+.3f}", va="center", fontsize=9, color=INK2)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(f"{metric} の差（対 A）")
    axes[0].set_yticks(y, [LABEL[p.split(" − ")[0]] for p in pairs])
    fig.suptitle("従来地形（A）への上乗せ と 95% 信頼区間（ブロック・ブートストラップ）"
                 "　橙＝区間が0を含まない", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_r2_diff.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
