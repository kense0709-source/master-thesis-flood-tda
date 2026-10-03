"""
地形12 vs 地形12+TDA の頑健性確認（10_model.py v2 の結果を確かめる）

確認すること:
  1. 木の上限：300本 → 600本に上げても結論が変わらないか（v2 で採用本数が上限近くに張り付いた設定あり）
  2. 空間ブロックの大きさ：5km 以外（2.5km / 10km）でも TDA の差の向きが同じか
  3. 差の確かさ：ブロック単位のブートストラップで TDA の差の 95% 信頼区間を出す
     （シード間 SD は乱数の揺れしか表さないため。どのブロックがテストに来るかの偶然性を評価する）

評価設定は spatial（本筋）と sp_event のみ（event は場所が重複する旧設定のため省略）。
雨量・衛星は使わない（2026-10-02 方針決定）。

ブートストラップ:
  テスト予測（全メッシュの out-of-fold 予測）を固定し、空間ブロックを復元抽出して
  PR-AUC・F1 の差（+TDA − 地形12）を再計算する。差はシードごとに計算してシード平均をとる。
  モデルの再学習はしない（評価集合の偶然性のみを見る）。

実行（夜間、約7時間）:
  python -u src/10b_robustness.py [--blocks 5 2.5 10] [--max-trees 600] [--seeds 5] [--n-boot 1000]

出力（ブロックの大きさごとに、終わり次第保存）:
  output/robust_b{km}.parquet       … シード別の指標
  output/robust_b{km}_preds.parquet … メッシュ別の予測
  output/robust_results.txt         … まとめ（ブロックごとに追記）
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
import eval_common as ec  # noqa: E402

warnings.filterwarnings("ignore")

BASE, PLUS = "地形12", "地形12+TDA"
OUT_TXT = ec.OUT_DIR / "robust_results.txt"


def block_bootstrap(preds: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """(setting, direction) ごとに、ブロック復元抽出で TDA の差の分布を出す"""
    rng = np.random.default_rng(seed)
    rows = []
    for (s, dr), g in preds.groupby(["setting", "direction"], sort=False):
        seeds = sorted(g["seed"].unique())
        # シードごとに base / plus の予測を同じメッシュ順で並べる
        ref = g[(g["model"] == BASE) & (g["seed"] == seeds[0])].sort_values("mesh_id")
        mesh = ref["mesh_id"].values
        y = ref["y"].values
        blocks = ref["block"].values
        P = {}
        for sd in seeds:
            for m in (BASE, PLUS):
                x = g[(g["model"] == m) & (g["seed"] == sd)].set_index("mesh_id").loc[mesh]
                P[(sd, m)] = (x["p"].values, x["pred"].values)

        ub, inv = np.unique(blocks, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(ub))]

        def diffs(idx):
            dpr, df1 = [], []
            for sd in seeds:
                (pb, cb), (pp, cp) = P[(sd, BASE)], P[(sd, PLUS)]
                dpr.append(average_precision_score(y[idx], pp[idx])
                           - average_precision_score(y[idx], pb[idx]))
                df1.append(f1_score(y[idx], cp[idx], zero_division=0)
                           - f1_score(y[idx], cb[idx], zero_division=0))
            return np.mean(dpr), np.mean(df1)

        obs_pr, obs_f1 = diffs(np.arange(len(y)))
        boot = np.empty((n_boot, 2))
        for b in range(n_boot):
            pick = rng.integers(0, len(ub), len(ub))
            idx = np.concatenate([members[k] for k in pick])
            if y[idx].sum() == 0:
                boot[b] = np.nan
                continue
            boot[b] = diffs(idx)
        boot = boot[~np.isnan(boot[:, 0])]
        for j, (name, obs) in enumerate([("PR-AUC", obs_pr), ("F1", obs_f1)]):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            rows.append({"setting": s, "direction": dr, "metric": name,
                         "diff": f"{obs:+.3f}", "95%CI": f"[{lo:+.3f}, {hi:+.3f}]",
                         "P(差≤0)": f"{(boot[:, j] <= 0).mean():.3f}",
                         "n_blocks": len(ub)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=float, nargs="+", default=[5.0, 2.5, 10.0])
    ap.add_argument("--max-trees", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    ec.MAX_TREES = args.max_trees
    feature_sets = {BASE: ec.FEATURE_SETS[BASE], PLUS: ec.FEATURE_SETS[PLUS]}

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("===== 頑健性確認：地形12 vs 地形12+TDA =====\n")
        f.write(f"木の上限 {args.max_trees} 本 / シード {args.seeds} / "
                f"ブートストラップ {args.n_boot} 回 / 設定 spatial・sp_event\n")
        f.write("比較用 v2（5km・上限300本）: output/eval_results.txt\n\n")

    for km in args.blocks:
        t0 = time.time()
        print(f"=== ブロック {km} km ===", flush=True)
        d = ec.load_dataset(km)
        res, preds = ec.run_all(d, feature_sets, seeds=range(args.seeds),
                                settings=("spatial", "sp_event"),
                                log=lambda s: print(s, flush=True), return_preds=True)
        tag = f"{km:g}".replace(".", "p")
        res.to_parquet(ec.OUT_DIR / f"robust_b{tag}.parquet", index=False)
        preds.to_parquet(ec.OUT_DIR / f"robust_b{tag}_preds.parquet", index=False)

        boot = block_bootstrap(preds[preds["model"].isin([BASE, PLUS])], args.n_boot)
        txt = [f"--- ブロック {km} km（{d['block'].nunique()} ブロック、"
               f"{(time.time() - t0) / 60:.0f} 分） ---",
               ec.summarize(res).to_string(index=False), "",
               "TDA の差（シード平均）とブロック・ブートストラップの 95% 信頼区間",
               boot.to_string(index=False), "", ""]
        with open(OUT_TXT, "a", encoding="utf-8") as f:
            f.write("\n".join(txt))
        print("\n".join(txt), flush=True)

    print(f"保存: {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
