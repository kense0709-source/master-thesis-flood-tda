"""
Model A / A+PI / B / C' / C の比較（実験計画.md §5）

評価は eval_common（spatial と sp_event、内部CVで木の本数・閾値、シード、ブロック・ブートストラップ）。
各モデルについて、基準モデル（既定 A）との差の 95% 信頼区間を出す。

実行例:
  # PI を含まない軽いモデルだけ（数分）
  python -u src/10d_model_compare.py --models A "C'" --blocks 5 10 --inner-km 5
  # 全モデル（PI を含むので重い。夜間）
  python -u src/10d_model_compare.py --models A A+PI B "C'" C --blocks 10 --inner-km 5

出力:
  output/compare_{タグ}.txt / .parquet / _preds.parquet
  タグ = モデル名とブロックの大きさ（例: A_Cp_b10_in5）
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import eval_common as ec  # noqa: E402

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["A", "C'"],
                    help=f"比べるモデル（{', '.join(ec.MODEL_SETS)}）")
    ap.add_argument("--base", default="A")
    ap.add_argument("--blocks", type=float, nargs="+", default=[10.0])
    ap.add_argument("--inner-km", type=float, default=5.0)
    ap.add_argument("--max-trees", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    if args.base not in args.models:
        args.models = [args.base] + args.models
    ec.MAX_TREES = args.max_trees
    sets = {m: ec.MODEL_SETS[m] for m in args.models}
    mtag = "_".join(m.replace("'", "p").replace("+", "") for m in args.models)

    for km in args.blocks:
        t0 = time.time()
        tag = f"{mtag}_b{km:g}".replace(".", "p")
        if args.inner_km:
            tag += f"_in{args.inner_km:g}".replace(".", "p")
        print(f"=== ブロック {km} km（内部CV {args.inner_km} km）: {args.models} ===", flush=True)

        d = ec.load_dataset(km, args.inner_km)
        missing = [c for f in sets.values() for c in f if c not in d.columns]
        if missing:
            raise SystemExit(f"列がありません: {sorted(set(missing))[:5]} … "
                             "12_depression_features.py を先に実行してください")
        res, preds = ec.run_all(d, sets, seeds=range(args.seeds),
                                settings=("spatial", "sp_event"),
                                log=lambda s: print(s, flush=True), return_preds=True)
        res.to_parquet(ec.OUT_DIR / f"compare_{tag}.parquet", index=False)
        preds.to_parquet(ec.OUT_DIR / f"compare_{tag}_preds.parquet", index=False)

        lines = [f"===== モデル比較: {', '.join(args.models)}（基準 {args.base}） =====",
                 f"外側ブロック {km} km（{d['block'].nunique()} 個）/ 内部CV {args.inner_km} km / "
                 f"木の上限 {args.max_trees} / シード {args.seeds} / "
                 f"ブートストラップ {args.n_boot} 回 / {(time.time() - t0) / 60:.0f} 分",
                 "", ec.summarize(res).to_string(index=False), ""]
        for m in args.models:
            if m == args.base:
                continue
            boot = ec.block_bootstrap(preds, args.base, m, args.n_boot)
            lines += [f"--- {m} − {args.base}（95% 信頼区間） ---",
                      boot.to_string(index=False), ""]
        text = "\n".join(lines)
        (ec.OUT_DIR / f"compare_{tag}.txt").write_text(text, encoding="utf-8")
        print(text, flush=True)


if __name__ == "__main__":
    main()
