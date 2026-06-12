"""
Phase 2: TDA特徴抽出（DEM → Persistence Image）

処理フロー:
  1. 各500mメッシュ分のDEMパッチを切り出し（約90×90px, 5m解像度）
  2. gudhi.CubicalComplex で sublevel set filtration
       H0: 閉じた窪み（局所最小値 → サドル点で合流）
       H1: 尾根で囲まれた盆地ループ
  3. Persistence Diagram → Persistence Image（20×20=400次元）
  4. H0(400次元) + H1(400次元) = 800次元ベクトルをメッシュごとに保存

チェックポイント機能あり（途中再開可能）

出力:
  output/tda_features.parquet
    列: mesh_id, h0_000〜h0_399, h1_000〜h1_399
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
import gudhi
from persim import PersistenceImager
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

BASE_DIR       = Path(__file__).parent.parent
DEM_PATH       = BASE_DIR / "output" / "dem_merged.tif"
MESH_PATH      = BASE_DIR / "output" / "mesh_500m.gpkg"
OUT_PATH       = BASE_DIR / "output" / "tda_features.parquet"
CHECKPOINT_PATH= BASE_DIR / "output" / "tda_checkpoint.parquet"

IMG_SIZE   = 20       # Persistence Image の1辺サイズ（20×20=400次元）
NODATA_VAL = -9999.0
SAMPLE_N   = 300      # PersistenceImager の fit に使うサンプル数
CKPT_EVERY = 200      # この件数ごとにチェックポイント保存


# ------------------------------------------------------------------ #
# 1. DEMパッチ切り出し
# ------------------------------------------------------------------ #

def read_patch(dem_src, geom, mesh_crs: str, dem_crs: str) -> np.ndarray | None:
    """メッシュジオメトリに対応するDEMパッチを読み込む"""
    left, bottom, right, top = transform_bounds(mesh_crs, dem_crs, *geom.bounds)
    window = window_from_bounds(left, bottom, right, top, dem_src.transform)
    patch = dem_src.read(1, window=window, boundless=True,
                         fill_value=NODATA_VAL).astype(np.float64)
    return patch


def preprocess_patch(patch: np.ndarray) -> np.ndarray | None:
    """NODATA処理。有効ピクセルが少なければ None を返す"""
    valid = patch != NODATA_VAL
    if valid.sum() < 50:
        return None
    max_valid = patch[valid].max()
    filled = patch.copy()
    # NODATAは地形の"壁"として扱う（sublevel set で最後まで残る）
    filled[~valid] = max_valid + 500.0
    return filled


# ------------------------------------------------------------------ #
# 2. Persistence Diagram 計算
# ------------------------------------------------------------------ #

def compute_diagrams(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gudhi CubicalComplex で H0, H1 を計算"""
    cc = gudhi.CubicalComplex(
        dimensions=list(patch.shape),
        top_dimensional_cells=patch.flatten().tolist()
    )
    cc.compute_persistence()

    h0 = np.array(cc.persistence_intervals_in_dimension(0))
    h1 = np.array(cc.persistence_intervals_in_dimension(1))

    # 無限大の death を除去（H0 の最後の成分は常に inf）
    if len(h0):
        h0 = h0[np.isfinite(h0[:, 1])]
    else:
        h0 = np.empty((0, 2))

    if len(h1):
        h1 = h1[np.isfinite(h1[:, 1])]
    else:
        h1 = np.empty((0, 2))

    return h0, h1


# ------------------------------------------------------------------ #
# 3. PersistenceImager の fit
# ------------------------------------------------------------------ #

def fit_imager(diagrams: list, img_size: int) -> PersistenceImager:
    """サンプルの Persistence Diagram から Imager を fit"""
    valid = [d for d in diagrams if len(d) > 0]
    if not valid:
        imager = PersistenceImager(birth_range=(0, 1), pers_range=(0, 1),
                                   pixel_size=1.0 / img_size)
        return imager

    imager = PersistenceImager(pixel_size=1.0)   # 仮の pixel_size
    imager.fit(valid)

    b_span = imager.birth_range[1] - imager.birth_range[0]
    p_span = imager.pers_range[1]  - imager.pers_range[0]
    pixel_size = max(b_span, p_span) / img_size
    imager.pixel_size = max(pixel_size, 1e-3)   # ゼロ除算防止

    return imager


# ------------------------------------------------------------------ #
# 4. Diagram → Persistence Image ベクトル
# ------------------------------------------------------------------ #

def diagram_to_vec(diagram: np.ndarray, imager: PersistenceImager,
                   img_size: int) -> np.ndarray:
    """Persistence Diagram を img_size×img_size のフラットベクトルに変換"""
    n = img_size * img_size
    if len(diagram) == 0:
        return np.zeros(n, dtype=np.float32)
    try:
        img = imager.transform([diagram])[0]
        # imager の出力サイズが img_size と異なる場合リサイズ
        if img.size != n:
            from skimage.transform import resize
            img = resize(img, (img_size, img_size), anti_aliasing=True)
        return img.flatten().astype(np.float32)
    except Exception:
        return np.zeros(n, dtype=np.float32)


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    t_start = time.time()

    # --- データ読み込み ---
    print("データ読み込み中...")
    mesh = gpd.read_file(MESH_PATH)
    mesh_crs = mesh.crs.to_string()
    print(f"  メッシュ数: {len(mesh)}")

    dem_src = rasterio.open(DEM_PATH)
    dem_crs = dem_src.crs.to_string()
    print(f"  DEM CRS: {dem_crs}")

    # --- チェックポイント確認 ---
    done_ids = set()
    if CHECKPOINT_PATH.exists():
        cp = pd.read_parquet(CHECKPOINT_PATH)
        done_ids = set(cp["mesh_id"].tolist())
        print(f"  チェックポイント: {len(done_ids)}件完了済み → 残り{len(mesh)-len(done_ids)}件")

    remaining = mesh[~mesh["mesh_id"].isin(done_ids)].reset_index(drop=True)

    if len(remaining) == 0:
        print("全件処理済み。最終結果を出力します。")
        result = pd.read_parquet(CHECKPOINT_PATH).sort_values("mesh_id").reset_index(drop=True)
        result.to_parquet(OUT_PATH, index=False)
        print(f"保存完了: {OUT_PATH}")
        return

    # --- サンプルで PersistenceImager を fit ---
    print(f"\nPersistenceImager を fit 中（最初の{SAMPLE_N}件）...")
    h0_sample, h1_sample = [], []

    for _, row in remaining.iloc[:SAMPLE_N].iterrows():
        patch = read_patch(dem_src, row.geometry, mesh_crs, dem_crs)
        patch = preprocess_patch(patch)
        if patch is None:
            continue
        h0, h1 = compute_diagrams(patch)
        if len(h0): h0_sample.append(h0)
        if len(h1): h1_sample.append(h1)

    h0_imager = fit_imager(h0_sample, IMG_SIZE)
    h1_imager = fit_imager(h1_sample, IMG_SIZE)
    print(f"  H0: birth={h0_imager.birth_range}, pers={h0_imager.pers_range}, pixel={h0_imager.pixel_size:.2f}m")
    print(f"  H1: birth={h1_imager.birth_range}, pers={h1_imager.pers_range}, pixel={h1_imager.pixel_size:.2f}m")

    # --- 全件処理 ---
    h0_cols = [f"h0_{i:03d}" for i in range(IMG_SIZE * IMG_SIZE)]
    h1_cols = [f"h1_{i:03d}" for i in range(IMG_SIZE * IMG_SIZE)]

    print(f"\n{len(remaining)}件のTDA計算開始...")
    buffer = []
    n_done = 0

    for _, row in remaining.iterrows():
        patch = read_patch(dem_src, row.geometry, mesh_crs, dem_crs)
        patch = preprocess_patch(patch)

        if patch is None:
            h0_vec = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.float32)
            h1_vec = np.zeros(IMG_SIZE * IMG_SIZE, dtype=np.float32)
        else:
            h0, h1 = compute_diagrams(patch)
            h0_vec = diagram_to_vec(h0, h0_imager, IMG_SIZE)
            h1_vec = diagram_to_vec(h1, h1_imager, IMG_SIZE)

        rec = {"mesh_id": row["mesh_id"]}
        rec.update(dict(zip(h0_cols, h0_vec)))
        rec.update(dict(zip(h1_cols, h1_vec)))
        buffer.append(rec)
        n_done += 1

        # チェックポイント保存
        if n_done % CKPT_EVERY == 0:
            df_new = pd.DataFrame(buffer)
            if CHECKPOINT_PATH.exists():
                df_new = pd.concat([pd.read_parquet(CHECKPOINT_PATH), df_new], ignore_index=True)
            df_new.to_parquet(CHECKPOINT_PATH, index=False)
            buffer = []
            elapsed = time.time() - t_start
            speed = n_done / elapsed
            remaining_s = (len(remaining) - n_done) / speed
            print(f"  {len(done_ids)+n_done}/{len(mesh)} 完了 | "
                  f"経過 {elapsed/60:.1f}分 | 残り推定 {remaining_s/60:.1f}分")

    # 残りバッファを保存
    if buffer:
        df_new = pd.DataFrame(buffer)
        if CHECKPOINT_PATH.exists():
            df_new = pd.concat([pd.read_parquet(CHECKPOINT_PATH), df_new], ignore_index=True)
        df_new.to_parquet(CHECKPOINT_PATH, index=False)

    # --- 最終結果 ---
    result = pd.read_parquet(CHECKPOINT_PATH).sort_values("mesh_id").reset_index(drop=True)
    result.to_parquet(OUT_PATH, index=False)

    elapsed = time.time() - t_start
    print(f"\n完了: {len(result)}メッシュ | {elapsed/60:.1f}分")
    print(f"特徴量次元: H0×{IMG_SIZE*IMG_SIZE} + H1×{IMG_SIZE*IMG_SIZE} = {IMG_SIZE*IMG_SIZE*2}次元")
    print(f"保存: {OUT_PATH}")
    dem_src.close()


if __name__ == "__main__":
    main()
