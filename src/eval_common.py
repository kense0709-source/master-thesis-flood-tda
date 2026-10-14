"""
評価の共通モジュール（全モデル比較はこれを経由する）

旧評価コード（10_model.py / 07c_eval_upstream_rain.py）の不備と対処:
  1. XGBoost の早期打ち切りに訓練データそのものを使っていた
     → 訓練データ内を空間ブロック単位で分割した「内部検証データ」を使う
  2. F1 の閾値をテストデータ上で最適化していた
     → 閾値も内部検証データで決め、テストには固定値として適用する
  3. 同じメッシュが訓練（2020）とテスト（2023）の両方に現れていた
     → 地形特徴はイベント不変なので、モデルは「2020年に浸水した場所」を覚えるだけで
        2023年を当てられる（＝コピーベースラインの学習）。
        空間ブロックで場所を分け、未見の場所で評価する設定を追加する

評価設定:
  event   … イベント間：訓練 tr の全メッシュ → テスト te の全メッシュ（旧設定。場所は重複）
  spatial … 空間ブロック CV：ラベル = いずれかのイベントで浸水。静的感受性評価の本筋
  sp_event… 空間ブロック × イベント間：訓練 tr の訓練ブロック → テスト te のテストブロック
            （別イベントかつ未見の場所。最も厳しい）

コピーベースライン（tr の浸水ラベルをそのまま te の予測とする）は
event / sp_event で同じテスト集合に対して算出し、併記する。

使い方:
  from eval_common import load_dataset, FEATURE_SETS, run_all
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output"

# ------------------------------------------------------------------ #
# 特徴量グループ
# ------------------------------------------------------------------ #

ELEV = ["elev_mean", "elev_min", "elev_max", "elev_std"]
TERR = ["twi_mean", "slope_mean", "slope_max"]
SINK = ["sink_area_frac", "sink_storage_depth", "sink_depth_max",
        "sink_depth_mean", "sink_volume_sum"]
RAIN = ["rain_max1h", "rain_max3h", "rain_max6h", "rain_sum24h"]
SAT  = ["ndvi_mean", "ndvi_min", "mndwi_mean", "mndwi_max"]
TDA  = [f"h0_{i:03d}" for i in range(400)] + [f"h1_{i:03d}" for i in range(400)]
# バッファ付きパッチで計算し直した PI（08b_tda_buffered.py、パッチ境界の修正版）
TDA_B = [f"h0b_{i:03d}" for i in range(400)] + [f"h1b_{i:03d}" for i in range(400)]
# birth を log(1+標高) にして低地に解像度を割いた PI（08c_tda_variants.py --variant log）
TDA_L = [f"h0l_{i:03d}" for i in range(400)] + [f"h1l_{i:03d}" for i in range(400)]
TERRAIN = SINK + ELEV + TERR                     # 地形12次元（現行の強いベースライン）
# 窪地単位の TDA 特徴（12_depression_features.py）。最後の3つは fill では出せない階層の特徴
DEP  = ["dep_count", "dep_pers_max", "dep_pers_mean", "dep_volume_sum", "dep_volume_max",
        "dep_area_sum", "dep_area_max", "dep_meandepth_max", "dep_width_min",
        "dep_storage_depth", "dep_nest_depth", "dep_n_nested", "dep_volume_ratio"]
# 土地利用・地形分類の面積率（14_landuse.py）。Model D 用。
# 流域にほぼない地形（後背低地・河川敷・凹地。平均面積率 0.001 以下）は除く
LU   = ["lu_paddy", "lu_farm", "lu_forest", "lu_building", "lu_transport", "lu_water", "lu_other"]
LF   = ["lf_mountain", "lf_terrace", "lf_fan", "lf_levee", "lf_floodplain", "lf_oldchannel",
        "lf_water", "lf_fill", "lf_cut", "lf_reclaim", "lf_covered"]

# キー「地形12」「地形12+TDA」は 10b_robustness.py が名前で参照する（= Model A / A+PI）
FEATURE_SETS = {
    "地形12":       TERRAIN,
    "地形12+TDA":   TERRAIN + TDA,
}
# 実験計画.md §5 のモデル定義（窪地特徴の計算後に使う）
MODEL_SETS = {
    "A":    TERRAIN,
    "A+PI": TERRAIN + TDA,
    "B":    DEP + TDA,
    "C'":   TERRAIN + DEP,
    "C":    TERRAIN + DEP + TDA,
    # パッチ境界を修正した PI 版
    "A+PIb": TERRAIN + TDA_B,
    "Cb":    TERRAIN + DEP + TDA_B,
    "Bb":    DEP + TDA_B,
    "A+PIl": TERRAIN + TDA_L,
    # Model D：土地利用・地形分類を足したうえで TDA がまだ効くか（D vs D0 が本題）。
    # PI は境界を直したバッファ付き版を使う（旧 PI は境界の偽ペアを含むため。method_choices.md ステップ9）
    "D0":   TERRAIN + LU + LF,
    "D'":   TERRAIN + LU + LF + DEP,
    "D":    TERRAIN + LU + LF + TDA_B,
}

EVENTS = ["2020", "2023"]


# ------------------------------------------------------------------ #
# データ
# ------------------------------------------------------------------ #

def load_dataset(block_km: float = 5.0, inner_km: float | None = None) -> pd.DataFrame:
    """master_features + sink_features を結合し、空間ブロック ID を付ける

    返り値は (mesh_id, event) 単位の縦持ち。
      block       … 外側（テスト用）の空間ブロック ID（block_km 四方）
      block_inner … 内側（木の本数・閾値を決める内部CV用）のブロック ID（inner_km 四方。
                    None なら block と同じ）
    """
    m = pd.read_parquet(OUT_DIR / "master_features.parquet")
    s = pd.read_parquet(OUT_DIR / "sink_features.parquet")
    d = m.merge(s, on="mesh_id", how="left")
    dep = OUT_DIR / "depression_features.parquet"
    if dep.exists():
        d = d.merge(pd.read_parquet(dep), on="mesh_id", how="left")
    for name in ("tda_buffered_features.parquet", "tda_log_features.parquet",
                 "landuse_features.parquet"):
        f = OUT_DIR / name
        if f.exists():
            d = d.merge(pd.read_parquet(f), on="mesh_id", how="left")
    d["event"] = d["event"].astype(str)
    return add_blocks(d, block_km, inner_km)


def add_blocks(d: pd.DataFrame, block_km: float,
               inner_km: float | None = None) -> pd.DataFrame:
    """500m メッシュの行・列番号から block_km 四方（と inner_km 四方）のブロック ID を作る

    外側と内側を分けるのは、大きいブロック（10km で 41 個）では内部CVの検証フォールドが
    数ブロックしかなく、木の本数の選択が不安定になるため（2026-10-03 頑健性確認）。
    内側に小さいブロックを使って漏れがあっても影響するのは本数・閾値の選び方だけで、
    テストの公正さは外側のブロックで保たれる。
    """
    mesh = gpd.read_file(OUT_DIR / "mesh_500m.gpkg",
                         columns=["mesh_id", "mesh_row", "mesh_col"],
                         ignore_geometry=True)

    def ids(km: float) -> pd.Series:
        n = int(round(km / 0.5))
        return (mesh["mesh_row"] // n).astype(str) + "_" + (mesh["mesh_col"] // n).astype(str)

    mesh["block"] = ids(block_km)
    mesh["block_inner"] = ids(inner_km) if inner_km else mesh["block"]
    d = d.drop(columns=["block", "block_inner"], errors="ignore")
    return d.merge(mesh[["mesh_id", "block", "block_inner"]], on="mesh_id", how="left")


def static_table(d: pd.DataFrame) -> pd.DataFrame:
    """1メッシュ1行の静的テーブル（ラベル = いずれかのイベントで浸水）

    イベントで値が変わる列（雨量・衛星）は含めない。
    """
    lab = d.pivot_table(index="mesh_id", columns="event", values="label")
    lab.columns = [f"label_{c}" for c in lab.columns]
    base = d[d["event"] == EVENTS[0]].drop(columns=["event", "label"] + RAIN + SAT,
                                            errors="ignore")
    out = base.merge(lab.reset_index(), on="mesh_id")
    out["label"] = (out[[f"label_{e}" for e in EVENTS]].max(axis=1) > 0).astype(int)
    return out


# ------------------------------------------------------------------ #
# 学習・閾値・指標
# ------------------------------------------------------------------ #

XGB_PARAMS = dict(max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric="aucpr", n_jobs=-1, verbosity=0)
MAX_TREES = 300      # 内部CVで調べる木の本数の上限（09-23 時点の最適本数は概ね 10〜250）
INNER_FOLDS = 5


def fit_xgb(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
            seed: int) -> tuple[xgb.XGBClassifier, float]:
    """内部 CV で木の本数と閾値を決め、訓練データ全体で再学習したモデルを返す

    1. 訓練データを空間ブロック単位で INNER_FOLDS 分割し、各フォールドで MAX_TREES 本まで学習
    2. 検証 PR-AUC の推移をフォールド間で平均し、平均が最大の本数を採用
       （1フォールドだけで早期打ち切りすると本数がシード間で ±100本以上ぶれたため）
    3. 採用本数での検証予測を全フォールド分集め、F1 最大の閾値を決める
    4. 採用本数で訓練データ全体に再学習（検証ブロックを外したままだと event 設定で
       テスト側の一部の場所だけ「未学習」になり、旧結果と比較できなくなるため）

    返り値: (モデル, 閾値)。テストデータはどの段階でも使わない。
    """
    sgkf = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))

    curves, models = [], []
    for fit_idx, val_idx in folds:
        yf = y[fit_idx]
        spw = (yf == 0).sum() / max((yf == 1).sum(), 1)
        mdl = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=MAX_TREES,
                                scale_pos_weight=spw, random_state=seed)
        mdl.fit(X.iloc[fit_idx], yf,
                eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        curves.append(mdl.evals_result()["validation_0"]["aucpr"])
        models.append(mdl)
    n_trees = int(np.argmax(np.mean(curves, axis=0))) + 1

    oof = np.empty(len(y))
    for mdl, (_, val_idx) in zip(models, folds):
        oof[val_idx] = mdl.predict_proba(X.iloc[val_idx],
                                         iteration_range=(0, n_trees))[:, 1]
    thr = f1_threshold(y, oof)

    spw_all = (y == 0).sum() / max((y == 1).sum(), 1)
    final = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=n_trees,
                              scale_pos_weight=spw_all, random_state=seed)
    final.fit(X, y, verbose=False)
    final.n_trees_ = n_trees
    return final, thr


def f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """F1 が最大となる閾値（検証データ専用。テストデータに使わないこと）"""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    # precision_recall_curve の最後の点は閾値を持たない
    return float(thr[np.argmax(f1[:-1])])


def metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """閾値非依存（PR/ROC/Recall@P0.5）＋ 固定閾値での F1 等"""
    pred = (p >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {
        "pr_auc":  average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "f1": f1, "precision": prec, "recall": rec,
        "recall_at_p50": recall_at_precision(y, p, 0.5),
        "thr": thr,
    }


def recall_at_precision(y: np.ndarray, p: np.ndarray, target: float) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= target
    return float(rec[ok].max()) if ok.any() else 0.0


# ------------------------------------------------------------------ #
# 評価設定
# ------------------------------------------------------------------ #

@dataclass
class Result:
    setting: str      # event / spatial / sp_event
    direction: str    # "2020→2023" など。spatial は "any"
    model: str        # 特徴量セット名 or "コピー"
    seed: int
    scores: dict
    # メッシュ別の予測（mesh_id, block, y, p, pred）。ブロック・ブートストラップ用
    pred: pd.DataFrame | None = field(default=None, repr=False)


def _pred_frame(mesh_id, block, y, p, pred) -> pd.DataFrame:
    return pd.DataFrame({"mesh_id": np.asarray(mesh_id), "block": np.asarray(block),
                         "y": np.asarray(y, dtype=int), "p": np.asarray(p, dtype=float),
                         "pred": np.asarray(pred, dtype=int)})


def eval_event(d: pd.DataFrame, feats: list, name: str, tr: str, te: str,
               seed: int) -> Result:
    """イベント間（旧設定）。早期打ち切り・閾値は訓練イベント内の空間ブロック検証で決める"""
    a = d[d["event"] == tr].reset_index(drop=True)
    b = d[d["event"] == te].reset_index(drop=True)
    mdl, thr = fit_xgb(a[feats], a["label"].values, a["block_inner"].values, seed)
    p = mdl.predict_proba(b[feats])[:, 1]
    sc = metrics(b["label"].values, p, thr)
    sc["n_trees"] = mdl.n_trees_
    pf = _pred_frame(b["mesh_id"], b["block"], b["label"], p, (p >= thr).astype(int))
    return Result("event", f"{tr}→{te}", name, seed, sc, pf)


def copy_event(d: pd.DataFrame, tr: str, te: str,
               mesh_ids: np.ndarray | None = None) -> dict:
    """コピーベースライン：tr の浸水ラベルを te の予測とする"""
    piv = d.pivot_table(index="mesh_id", columns="event", values="label")
    if mesh_ids is not None:
        piv = piv.loc[mesh_ids]
    return metrics(piv[te].values.astype(int), piv[tr].values.astype(float), 0.5)


def spatial_folds(groups: np.ndarray, y: np.ndarray, seed: int, n_folds: int):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def eval_spatial(st: pd.DataFrame, feats: list, name: str, seed: int,
                 n_folds: int = 5) -> Result:
    """空間ブロック CV（静的感受性）。テストブロックの予測を集めて一括で指標化"""
    y, g = st["label"].values, st["block"].values
    gi = st["block_inner"].values
    oof_p = np.full(len(st), np.nan)
    oof_pred = np.zeros(len(st), dtype=int)
    n_trees = []
    for tr_idx, te_idx in spatial_folds(g, y, seed, n_folds):
        a = st.iloc[tr_idx]
        mdl, thr = fit_xgb(a[feats], y[tr_idx], gi[tr_idx], seed)
        n_trees.append(mdl.n_trees_)
        p = mdl.predict_proba(st.iloc[te_idx][feats])[:, 1]
        oof_p[te_idx] = p
        oof_pred[te_idx] = (p >= thr).astype(int)   # 閾値はフォールドごとに異なる
    sc = metrics(y, oof_p, 0.5)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, oof_pred, average="binary", zero_division=0)
    sc.update(f1=f1, precision=prec, recall=rec, thr=np.nan, n_trees=np.mean(n_trees))
    pf = _pred_frame(st["mesh_id"], g, y, oof_p, oof_pred)
    return Result("spatial", "any", name, seed, sc, pf)


def eval_sp_event(d: pd.DataFrame, feats_by_name: dict, tr: str, te: str,
                  seed: int, n_folds: int = 5) -> list[Result]:
    """空間ブロック × イベント間。tr の訓練ブロックで学習 → te のテストブロックで評価

    フォールドの分け方は tr のラベルで層化し、全特徴量セット・コピーで共通にする。
    """
    a = d[d["event"] == tr].set_index("mesh_id")
    b = d[d["event"] == te].set_index("mesh_id").loc[a.index]
    y_tr, y_te, g = a["label"].values, b["label"].values, a["block"].values
    gi = a["block_inner"].values
    folds = spatial_folds(g, y_tr, seed, n_folds)

    out = []
    for name, feats in feats_by_name.items():
        oof_p = np.full(len(a), np.nan)
        oof_pred = np.zeros(len(a), dtype=int)
        n_trees = []
        for tr_idx, te_idx in folds:
            mdl, thr = fit_xgb(a.iloc[tr_idx][feats], y_tr[tr_idx], gi[tr_idx], seed)
            n_trees.append(mdl.n_trees_)
            p = mdl.predict_proba(b.iloc[te_idx][feats])[:, 1]
            oof_p[te_idx] = p
            oof_pred[te_idx] = (p >= thr).astype(int)
        sc = metrics(y_te, oof_p, 0.5)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, oof_pred, average="binary", zero_division=0)
        sc.update(f1=f1, precision=prec, recall=rec, thr=np.nan,
                  n_trees=np.mean(n_trees))
        pf = _pred_frame(a.index, g, y_te, oof_p, oof_pred)
        out.append(Result("sp_event", f"{tr}→{te}", name, seed, sc, pf))
    return out


# ------------------------------------------------------------------ #
# まとめて実行
# ------------------------------------------------------------------ #

def run_all(d: pd.DataFrame, feature_sets: dict, seeds=(0, 1, 2, 3, 4),
            settings=("event", "spatial", "sp_event"), log=print,
            return_preds: bool = False):
    """全設定を実行。return_preds=True なら (指標表, メッシュ別予測表) を返す"""
    rows: list[Result] = []
    st = static_table(d) if "spatial" in settings else None
    dirs = [(EVENTS[0], EVENTS[1]), (EVENTS[1], EVENTS[0])]

    for seed in seeds:
        log(f"  seed {seed}")
        for tr, te in dirs:
            if "event" in settings:
                for name, f in feature_sets.items():
                    rows.append(eval_event(d, f, name, tr, te, seed))
            if "sp_event" in settings:
                rows += eval_sp_event(d, feature_sets, tr, te, seed)
        if "spatial" in settings:
            for name, f in feature_sets.items():
                rows.append(eval_spatial(st, f, name, seed))

    # コピーはシードに依存しない（テスト集合は event / sp_event とも全メッシュ）
    for tr, te in dirs:
        for s in [x for x in ("event", "sp_event") if x in settings]:
            rows.append(Result(s, f"{tr}→{te}", "コピー", -1, copy_event(d, tr, te)))

    res = pd.DataFrame([{"setting": r.setting, "direction": r.direction,
                         "model": r.model, "seed": r.seed, **r.scores} for r in rows])
    if not return_preds:
        return res
    preds = pd.concat([r.pred.assign(setting=r.setting, direction=r.direction,
                                     model=r.model, seed=r.seed)
                       for r in rows if r.pred is not None], ignore_index=True)
    return res, preds


def summarize(res: pd.DataFrame,
              cols=("pr_auc", "roc_auc", "f1", "recall_at_p50", "n_trees")) -> pd.DataFrame:
    """シード間の平均 ± 標準偏差"""
    g = res.groupby(["setting", "direction", "model"], sort=False)[list(cols)]
    mean, sd = g.mean(), g.std().fillna(0.0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [f"{m:.3f}±{s:.3f}" for m, s in zip(mean[c], sd[c])]
    return out.reset_index()


def block_bootstrap(preds: pd.DataFrame, base: str, plus: str, n_boot: int = 1000,
                    seed: int = 0) -> pd.DataFrame:
    """(setting, direction) ごとに、空間ブロックの復元抽出で (plus − base) の差の分布を出す

    テスト予測（全メッシュの out-of-fold 予測）を固定し、ブロックを復元抽出して
    PR-AUC・F1 の差を再計算する。差はシードごとに計算してシード平均をとる。
    モデルの再学習はしない（どのブロックがテストに来たかの偶然性のみを見る）。
    preds は run_all(..., return_preds=True) の2つ目の返り値。
    """
    rng = np.random.default_rng(seed)
    rows = []
    preds = preds[preds["model"].isin([base, plus])]
    for (s, dr), g in preds.groupby(["setting", "direction"], sort=False):
        seeds = sorted(g["seed"].unique())
        # シードごとに base / plus の予測を同じメッシュ順で並べる
        ref = g[(g["model"] == base) & (g["seed"] == seeds[0])].sort_values("mesh_id")
        mesh = ref["mesh_id"].values
        y = ref["y"].values
        blocks = ref["block"].values
        P = {}
        for sd in seeds:
            for m in (base, plus):
                x = g[(g["model"] == m) & (g["seed"] == sd)].set_index("mesh_id").loc[mesh]
                P[(sd, m)] = (x["p"].values, x["pred"].values)

        ub, inv = np.unique(blocks, return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(ub))]

        def diffs(idx):
            dpr, df1 = [], []
            for sd in seeds:
                (pb, cb), (pp, cp) = P[(sd, base)], P[(sd, plus)]
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
