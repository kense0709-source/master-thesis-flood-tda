"""
Phase 3: モデル構築・評価

設計:
  - 訓練: 2020年イベント（10,025メッシュ）
  - テスト: 2023年イベント（10,025メッシュ）
  - モデル: XGBoost（クラス不均衡対応）
  - 評価指標: Recall, Precision, F1, ROC-AUC
  - 特徴重要度: SHAP

アブレーション実験（TDA有無の比較）:
  - Model A: TDA特徴量なし（地形統計 + 雨量 + NDVI/MNDWI）
  - Model B: TDA特徴量あり（Model A + H0/H1 Persistence Image）

出力:
  output/model_results.txt  … 評価指標サマリー
  output/shap_importance.parquet … SHAP特徴重要度
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, precision_recall_fscore_support)
import shap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output"

# TDA以外の特徴量列
BASE_FEAT_COLS = [
    "elev_mean", "elev_min", "elev_max", "elev_std",
    "twi_mean", "slope_mean", "slope_max",
    "rain_max1h", "rain_max3h", "rain_max6h", "rain_sum24h",
    "ndvi_mean", "ndvi_min", "mndwi_mean", "mndwi_max",
]
TDA_COLS_H0 = [f"h0_{i:03d}" for i in range(400)]
TDA_COLS_H1 = [f"h1_{i:03d}" for i in range(400)]


def train_xgb(X_train, y_train, scale_pos_weight: float) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train)],
              verbose=False)
    return model


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)

    result = {
        "model": model_name,
        "recall":    round(rec,  4),
        "precision": round(prec, 4),
        "f1":        round(f1,   4),
        "roc_auc":   round(auc,  4),
        "TP": int(cm[1, 1]),
        "FN": int(cm[1, 0]),
        "FP": int(cm[0, 1]),
        "TN": int(cm[0, 0]),
    }
    print(f"\n=== {model_name} ===")
    print(f"  Recall:    {rec:.4f}  (浸水を何割検出できたか)")
    print(f"  Precision: {prec:.4f}  (浸水と予測した中で本当に浸水の割合)")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  混同行列: TP={cm[1,1]}, FN={cm[1,0]}, FP={cm[0,1]}, TN={cm[0,0]}")
    return result


def shap_importance(model, X_train, feat_cols) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_train)
    importance = pd.DataFrame({
        "feature": feat_cols,
        "shap_mean_abs": np.abs(shap_vals).mean(axis=0),
    }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
    return importance


def main():
    print("データ読み込み中...")
    df = pd.read_parquet(OUT_DIR / "master_features.parquet")

    train = df[df["event"] == "2020"].reset_index(drop=True)
    test  = df[df["event"] == "2023"].reset_index(drop=True)

    y_train = train["label"].values
    y_test  = test["label"].values

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"  訓練: {len(train)}件 (浸水={n_pos}, 非浸水={n_neg})")
    print(f"  テスト: {len(test)}件 (浸水={y_test.sum()}, 非浸水={(y_test==0).sum()})")
    print(f"  scale_pos_weight: {scale_pos_weight:.1f}")

    results = []
    shap_dfs = {}

    # --- Model A: TDA なし ---
    print("\n[Model A] TDA特徴量なし")
    feat_a = BASE_FEAT_COLS
    X_train_a = train[feat_a].values
    X_test_a  = test[feat_a].values
    model_a = train_xgb(X_train_a, y_train, scale_pos_weight)
    results.append(evaluate(model_a, X_test_a, y_test, "Model A (TDAなし)"))
    shap_dfs["A"] = shap_importance(model_a, X_train_a, feat_a)
    print(f"  Top5 SHAP特徴量: {shap_dfs['A']['feature'].head(5).tolist()}")

    # --- Model B: TDA あり ---
    print("\n[Model B] TDA特徴量あり（H0+H1）")
    feat_b = BASE_FEAT_COLS + TDA_COLS_H0 + TDA_COLS_H1
    X_train_b = train[feat_b].values
    X_test_b  = test[feat_b].values
    model_b = train_xgb(X_train_b, y_train, scale_pos_weight)
    results.append(evaluate(model_b, X_test_b, y_test, "Model B (TDAあり)"))

    # TDAのSHAPは上位50特徴量のみ
    shap_b = shap_importance(model_b, X_train_b, feat_b)
    shap_dfs["B"] = shap_b
    top_b = shap_b.head(10)
    print(f"  Top10 SHAP特徴量:")
    print(top_b[["feature", "shap_mean_abs"]].to_string(index=False))

    # --- 結果保存 ---
    result_df = pd.DataFrame(results)
    print("\n\n===== アブレーション比較 =====")
    print(result_df[["model", "recall", "precision", "f1", "roc_auc"]].to_string(index=False))

    # TDA による改善
    diff_recall = results[1]["recall"]  - results[0]["recall"]
    diff_f1     = results[1]["f1"]      - results[0]["f1"]
    diff_auc    = results[1]["roc_auc"] - results[0]["roc_auc"]
    print(f"\nTDA追加による改善: Recall {diff_recall:+.4f}, F1 {diff_f1:+.4f}, AUC {diff_auc:+.4f}")

    result_df.to_parquet(OUT_DIR / "model_results.parquet", index=False)
    shap_dfs["B"].to_parquet(OUT_DIR / "shap_importance.parquet", index=False)

    # テキストサマリー
    summary_path = OUT_DIR / "model_results.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== Phase 3 モデル評価結果 =====\n\n")
        f.write("訓練: 2020年イベント / テスト: 2023年イベント\n\n")
        f.write(result_df[["model","recall","precision","f1","roc_auc"]].to_string(index=False))
        f.write(f"\n\nTDA追加による改善:\n")
        f.write(f"  Recall:  {diff_recall:+.4f}\n")
        f.write(f"  F1:      {diff_f1:+.4f}\n")
        f.write(f"  ROC-AUC: {diff_auc:+.4f}\n\n")
        f.write("Top20 SHAP特徴量 (Model B):\n")
        f.write(shap_dfs["B"].head(20)[["feature","shap_mean_abs"]].to_string(index=False))
    print(f"\n結果保存: {summary_path}")


if __name__ == "__main__":
    main()
