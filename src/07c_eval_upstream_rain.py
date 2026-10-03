"""
上流積算雨量の効果検証

比較する問い:
  1. 上流積算雨量は、局所雨量より浸水を説明できるか（単独AUC）
  2. 地形特徴に足したとき、局所雨量は有害・上流雨量は有益か（モデル比較）
  3. inflow_per_storage（流入量 ÷ 貯留容量）は効くか

前提:
  output/rain_upstream_features.parquet （07b_upstream_rain.py が生成）
  output/sink_features.parquet
  output/master_features.parquet

実行:
  python -u src/07c_eval_upstream_rain.py
"""

from pathlib import Path
import sys
import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from eval_common import add_blocks, fit_xgb, metrics  # noqa: E402

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output"
OUT_TXT  = OUT_DIR / "upstream_rain_results.txt"

ELEV = ["elev_mean", "elev_min", "elev_max", "elev_std"]
TERR = ["twi_mean", "slope_mean", "slope_max"]
SINK = ["sink_area_frac", "sink_storage_depth", "sink_depth_max",
        "sink_depth_mean", "sink_volume_sum"]
RAIN_LOCAL = ["rain_max1h", "rain_max3h", "rain_max6h", "rain_sum24h"]
RAIN_UP    = ["catchment_area_m2", "rain_up_mean24h", "rain_up_mean1h",
              "inflow_volume_m3_24h", "inflow_volume_m3_1h",
              "inflow_depth_mm", "inflow_per_storage"]
TERRAIN = SINK + ELEV + TERR


def load() -> pd.DataFrame:
    m = pd.read_parquet(OUT_DIR / "master_features.parquet",
                        columns=["mesh_id", "event", "label"] + ELEV + TERR + RAIN_LOCAL)
    s = pd.read_parquet(OUT_DIR / "sink_features.parquet")
    u = pd.read_parquet(OUT_DIR / "rain_upstream_features.parquet")
    d = m.merge(s, on="mesh_id", how="left").merge(u, on=["mesh_id", "event"], how="left")
    return add_blocks(d, 5.0)


def run(d: pd.DataFrame, feats: list, tr: str, te: str) -> tuple:
    """早期打ち切り・F1 閾値は訓練イベント内の空間ブロック検証で決める（eval_common）"""
    a, b = d[d.event == tr], d[d.event == te]
    mdl, thr = fit_xgb(a[feats], a.label.values, a.block.values, seed=42)
    sc = metrics(b.label.values, mdl.predict_proba(b[feats])[:, 1], thr)
    return sc["roc_auc"], sc["pr_auc"], sc["f1"]


def main() -> None:
    d = load()
    lines = []

    def out(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    out("=== 1. 単独 ROC-AUC：局所雨量 vs 上流積算雨量 ===")
    out("%-24s %8s %8s" % ("特徴量", "2020", "2023"))
    for c in RAIN_LOCAL + RAIN_UP:
        vals = []
        for ev in ["2020", "2023"]:
            s = d[d.event == ev]
            k = s[c].notna()
            a = roc_auc_score(s.loc[k, "label"], s.loc[k, c])
            vals.append(max(a, 1 - a))
        out("%-24s %8.3f %8.3f" % (c, vals[0], vals[1]))

    out()
    out("=== 2. モデル比較（地形特徴に何を足すか） ===")
    # catchment_area_m2 は両イベントで同一の「静的な地形量」であり、
    # inflow_volume も大部分が集水面積で決まる。
    # 「雨量が効いた」のか「集水面積という新しい地形特徴が効いた」のかを
    # 分離するため、静的成分だけを足した設定を挟む。
    CATCH  = ["catchment_area_m2"]                       # 静的（イベント不変）
    RAIN_D = ["rain_up_mean24h", "rain_up_mean1h"]       # 動的（上流の雨の強さ）
    RAIN_V = ["inflow_volume_m3_24h", "inflow_volume_m3_1h",
              "inflow_depth_mm", "inflow_per_storage"]   # 面積×雨量の複合

    cfgs = [
        ("地形のみ (12)",                  TERRAIN),
        ("地形 + 集水面積 (13) ←静的",      TERRAIN + CATCH),
        ("地形 + 集水面積 + 上流雨量 (15)", TERRAIN + CATCH + RAIN_D),
        ("地形 + 上流雨量フル (19)",        TERRAIN + RAIN_UP),
        ("地形 + 局所雨量 (16)",            TERRAIN + RAIN_LOCAL),
        ("地形 + 両方 (23)",                TERRAIN + RAIN_LOCAL + RAIN_UP),
        ("上流雨量のみ (7)",                RAIN_UP),
    ]
    for tr, te in [("2020", "2023"), ("2023", "2020")]:
        out(f"--- 訓練 {tr} → テスト {te} ---")
        out("%-24s %7s %7s %7s" % ("設定", "ROC", "PR", "F1"))
        for name, f in cfgs:
            r, p, f1 = run(d, f, tr, te)
            out("%-24s %7.3f %7.3f %7.3f" % (name, r, p, f1))
        out()

    OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"保存: {OUT_TXT}", flush=True)


if __name__ == "__main__":
    main()
