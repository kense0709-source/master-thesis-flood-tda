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
  --inner-km を付けた場合は、ファイル名の末尾に _in{km} が付く（例: robust_results_in5.txt）
  （内部CVだけ小さいブロックにする。10km 外側では内部CVが不安定になったため。eval_common.add_blocks 参照）
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import eval_common as ec  # noqa: E402

warnings.filterwarnings("ignore")

BASE, PLUS = "地形12", "地形12+TDA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=float, nargs="+", default=[5.0, 2.5, 10.0])
    ap.add_argument("--max-trees", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--inner-km", type=float, default=None,
                    help="内部CVのブロックの大きさ。省略時は外側と同じ")
    args = ap.parse_args()

    ec.MAX_TREES = args.max_trees
    feature_sets = {BASE: ec.FEATURE_SETS[BASE], PLUS: ec.FEATURE_SETS[PLUS]}
    # 内側を分けた実行は別ファイルに出す（既存の結果を上書きしない）
    sfx = f"_in{args.inner_km:g}".replace(".", "p") if args.inner_km else ""
    out_txt = ec.OUT_DIR / f"robust_results{sfx}.txt"

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("===== 頑健性確認：地形12 vs 地形12+TDA =====\n")
        f.write(f"内部CVのブロック: {args.inner_km or '外側と同じ'} km\n")
        f.write(f"木の上限 {args.max_trees} 本 / シード {args.seeds} / "
                f"ブートストラップ {args.n_boot} 回 / 設定 spatial・sp_event\n")
        f.write("比較用 v2（5km・上限300本）: output/eval_results.txt\n\n")

    for km in args.blocks:
        t0 = time.time()
        print(f"=== ブロック {km} km ===", flush=True)
        d = ec.load_dataset(km, args.inner_km)
        res, preds = ec.run_all(d, feature_sets, seeds=range(args.seeds),
                                settings=("spatial", "sp_event"),
                                log=lambda s: print(s, flush=True), return_preds=True)
        tag = f"{km:g}".replace(".", "p") + sfx
        res.to_parquet(ec.OUT_DIR / f"robust_b{tag}.parquet", index=False)
        preds.to_parquet(ec.OUT_DIR / f"robust_b{tag}_preds.parquet", index=False)

        boot = ec.block_bootstrap(preds, BASE, PLUS, args.n_boot)
        txt = [f"--- ブロック {km} km（{d['block'].nunique()} ブロック、"
               f"{(time.time() - t0) / 60:.0f} 分） ---",
               ec.summarize(res).to_string(index=False), "",
               "TDA の差（シード平均）とブロック・ブートストラップの 95% 信頼区間",
               boot.to_string(index=False), "", ""]
        with open(out_txt, "a", encoding="utf-8") as f:
            f.write("\n".join(txt))
        print("\n".join(txt), flush=True)

    print(f"保存: {out_txt}", flush=True)


if __name__ == "__main__":
    main()
