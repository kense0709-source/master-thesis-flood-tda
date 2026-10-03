"""
Phase 3: モデル比較（地形のみ vs 地形＋TDA）

評価は src/eval_common.py に集約（早期打ち切り・閾値は訓練側の空間ブロック検証で決定）。
評価設定:
  event    … イベント間（2020→2023 / 2023→2020）。場所は訓練・テストで重複
  spatial  … 空間ブロック CV。ラベル = いずれかのイベントで浸水（静的感受性）
  sp_event … 空間ブロック × イベント間（別イベントかつ未見の場所）
コピーベースライン（訓練イベントの浸水域をそのまま予測）を event / sp_event に併記。
各設定をシード5本で回し、平均±標準偏差と、TDA 追加による差（同一シードでの対応差）を出す。

旧版（2026-07-03、Model A 0.859 / Model B 0.928）は評価に不備があり使用しない。
その出力 output/model_results.txt は記録として残している。

実行:
  python -u src/10_model.py [--block-km 5] [--seeds 5]

出力:
  output/eval_results.parquet … シード別の全指標
  output/eval_results.txt     … 平均±標準偏差・TDA の対応差
"""

import argparse
import sys
from pathlib import Path
import warnings

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_common import (OUT_DIR, FEATURE_SETS, load_dataset, run_all,  # noqa: E402
                         summarize)

warnings.filterwarnings("ignore")


def tda_gain(res: pd.DataFrame, base: str = "地形12", plus: str = "地形12+TDA",
             cols=("pr_auc", "roc_auc", "f1")) -> pd.DataFrame:
    """同一シード・同一設定での (plus − base) の平均・標準偏差・正になった回数"""
    key = ["setting", "direction", "seed"]
    a = res[res["model"] == base].set_index(key)[list(cols)]
    b = res[res["model"] == plus].set_index(key)[list(cols)]
    diff = (b - a).reset_index()
    rows = []
    for (s, dr), g in diff.groupby(["setting", "direction"], sort=False):
        row = {"setting": s, "direction": dr}
        for c in cols:
            row[c] = f"{g[c].mean():+.3f}±{g[c].std():.3f} ({(g[c] > 0).sum()}/{len(g)})"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-km", type=float, default=5.0)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    lines = []

    def out(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    d = load_dataset(args.block_km)
    out("===== モデル比較（評価コード修正版） =====")
    out(f"空間ブロック: {args.block_km} km 四方（{d['block'].nunique()} ブロック）, "
        f"シード数: {args.seeds}")
    out(f"浸水メッシュ: {d.groupby('event')['label'].sum().to_dict()}")
    out("木の本数・F1 閾値: 訓練側を空間ブロックで5分割した内部CVで決定（テストは不使用）")
    out()

    res = run_all(d, FEATURE_SETS, seeds=range(args.seeds), log=print)
    res.to_parquet(OUT_DIR / "eval_results.parquet", index=False)

    out("--- 指標（平均±標準偏差、シード間） ---")
    out(summarize(res).to_string(index=False))
    out()
    out("--- TDA 追加による差（地形12+TDA − 地形12、括弧内は差が正だったシード数） ---")
    out(tda_gain(res).to_string(index=False))

    path = OUT_DIR / "eval_results.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n保存: {path}")


if __name__ == "__main__":
    main()
