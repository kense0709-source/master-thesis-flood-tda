"""
全特徴量の統合

各parquetを結合して、モデル入力用マスターテーブルを作成する。

静的特徴量（メッシュ固有、イベントによらない）:
  mesh_dem_stats    → elev_mean/min/max/std
  terrain_features  → twi_mean, slope_mean, slope_max
  tda_features      → h0_000〜h0_399, h1_000〜h1_399

動的特徴量（イベントごとに異なる）:
  rainfall_features → rain_max1h/3h/6h/24h
  sentinel2_features→ ndvi_mean/min, mndwi_mean/max

ラベル:
  mesh_labels       → label (0/1)

出力:
  output/master_features.parquet
    行: 20,050（10,025メッシュ × 2イベント）
    列: mesh_id, event, label + 全特徴量
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output"

def main():
    print("各parquetを読み込み中...")

    labels    = pd.read_parquet(OUT_DIR / "mesh_labels.parquet")
    dem_stats = pd.read_parquet(OUT_DIR / "mesh_dem_stats.parquet")
    terrain   = pd.read_parquet(OUT_DIR / "terrain_features.parquet")
    tda       = pd.read_parquet(OUT_DIR / "tda_features.parquet")
    rainfall  = pd.read_parquet(OUT_DIR / "rainfall_features.parquet")
    sentinel  = pd.read_parquet(OUT_DIR / "sentinel2_features.parquet")

    print(f"  labels:    {labels.shape}")
    print(f"  dem_stats: {dem_stats.shape}")
    print(f"  terrain:   {terrain.shape}")
    print(f"  tda:       {tda.shape}")
    print(f"  rainfall:  {rainfall.shape}")
    print(f"  sentinel2: {sentinel.shape}")

    # ベーステーブル（ラベル）
    df = labels.copy()

    # 静的特徴量（mesh_id でjoin）
    dem_cols = ["mesh_id", "elev_mean", "elev_min", "elev_max", "elev_std"]
    df = df.merge(dem_stats[dem_cols], on="mesh_id", how="left")

    terrain_cols = ["mesh_id", "twi_mean", "slope_mean", "slope_max"]
    df = df.merge(terrain[terrain_cols], on="mesh_id", how="left")

    df = df.merge(tda, on="mesh_id", how="left")

    # 動的特徴量（mesh_id + event でjoin）
    df = df.merge(rainfall, on=["mesh_id", "event"], how="left")
    df = df.merge(sentinel,  on=["mesh_id", "event"], how="left")

    print(f"\n統合後: {df.shape}")
    print(f"  NaN件数（列別）:")
    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    if len(nan_cols):
        print(nan_cols.to_string())
    else:
        print("  NaNなし")

    # sentinel2のNaN（2023年タイル未カバー分）を列平均で補完
    sentinel_cols = [c for c in df.columns if c.startswith(("ndvi_", "mndwi_"))]
    for col in sentinel_cols:
        col_mean = df.groupby("event")[col].transform("mean")
        df[col] = df[col].fillna(col_mean)
    print(f"\n  Sentinel-2 NaN → イベント別平均値で補完")

    # 最終NaN確認
    remaining_nan = df.isna().sum().sum()
    print(f"  補完後のNaN合計: {remaining_nan}")

    # カラム順整理
    id_cols    = ["mesh_id", "event", "label"]
    feat_cols  = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feat_cols]

    out_path = OUT_DIR / "master_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\n保存完了: {out_path}")
    print(f"  形状: {df.shape[0]}行 × {df.shape[1]}列")
    print(f"  特徴量数: {len(feat_cols)}")

    # イベント別ラベル分布
    print(f"\nイベント別ラベル分布:")
    print(df.groupby(["event", "label"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
