"""
TDA可視化：代表メッシュの DEM / Persistence Diagram / Persistence Image

浸水あり・なし各10件、合計20枚のPNGを出力する。
代表メッシュの選び方：H1ノルムで昇順ソートし等間隔に10件を抽出
（地形のトポロジー的豊富さが「低〜高」の範囲をカバーする）

出力先: output/tda_vis/
  flooded_01.png 〜 flooded_10.png
  nonflooded_01.png 〜 nonflooded_10.png
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
import gudhi
from persim import PersistenceImager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output" / "tda_vis"
OUT_DIR.mkdir(exist_ok=True)

NODATA_VAL = -9999.0
IMG_SIZE   = 20


# ------------------------------------------------------------------ #
# データ準備
# ------------------------------------------------------------------ #

def load_data():
    mesh   = gpd.read_file(BASE_DIR / "output" / "mesh_500m.gpkg")
    tda    = pd.read_parquet(BASE_DIR / "output" / "tda_features.parquet")
    labels = pd.read_parquet(BASE_DIR / "output" / "mesh_labels.parquet")
    lab2020 = labels[labels["event"] == "2020"][["mesh_id", "label"]]

    h1_cols = [f"h1_{i:03d}" for i in range(400)]
    tda["h1_norm"] = np.linalg.norm(tda[h1_cols].values, axis=1)

    merged = tda[["mesh_id", "h1_norm"]].merge(lab2020, on="mesh_id")
    merged = merged.merge(mesh[["mesh_id", "geometry"]], on="mesh_id")
    return merged


def select_representatives(df, label: int, n: int = 10) -> list:
    """H1ノルムで等間隔に n 件を選ぶ"""
    sub = df[df["label"] == label].sort_values("h1_norm").reset_index(drop=True)
    indices = np.linspace(0, len(sub) - 1, n, dtype=int)
    return sub.iloc[indices]["mesh_id"].tolist()


# ------------------------------------------------------------------ #
# DEM パッチ取得 & 前処理
# ------------------------------------------------------------------ #

def get_patch(dem_src, geom, mesh_crs, dem_crs):
    l, b, r, t = transform_bounds(mesh_crs, dem_crs, *geom.bounds)
    window = window_from_bounds(l, b, r, t, dem_src.transform)
    patch = dem_src.read(1, window=window, boundless=True,
                         fill_value=NODATA_VAL).astype(np.float64)
    valid = patch != NODATA_VAL
    if valid.sum() < 50:
        return None, None
    raw = patch.copy()
    filled = patch.copy()
    filled[~valid] = patch[valid].max() + 500.0
    return raw, filled


# ------------------------------------------------------------------ #
# Persistence Diagram 計算
# ------------------------------------------------------------------ #

def compute_diagrams(filled):
    cc = gudhi.CubicalComplex(
        dimensions=list(filled.shape),
        top_dimensional_cells=filled.flatten().tolist()
    )
    cc.compute_persistence()
    h0 = np.array(cc.persistence_intervals_in_dimension(0))
    h1 = np.array(cc.persistence_intervals_in_dimension(1))
    if len(h0): h0 = h0[np.isfinite(h0[:, 1])]
    else:       h0 = np.empty((0, 2))
    if len(h1): h1 = h1[np.isfinite(h1[:, 1])]
    else:       h1 = np.empty((0, 2))
    return h0, h1


# ------------------------------------------------------------------ #
# Persistence Image 作成
# ------------------------------------------------------------------ #

def make_persistence_image(dgm, img_size=IMG_SIZE):
    """birth-death 配列 → Persistence Image（img_size×img_size）"""
    if len(dgm) == 0:
        return np.zeros((img_size, img_size))
    pimgr = PersistenceImager(pixel_size=1.0)
    pimgr.fit([dgm])
    b_span = pimgr.birth_range[1] - pimgr.birth_range[0]
    p_span = pimgr.pers_range[1]  - pimgr.pers_range[0]
    pimgr.pixel_size = max(max(b_span, p_span) / img_size, 1e-3)
    img = pimgr.transform([dgm])[0]
    # img_size に合わせてリサイズ
    if img.shape != (img_size, img_size):
        from skimage.transform import resize
        img = resize(img, (img_size, img_size), anti_aliasing=True)
    return img


# ------------------------------------------------------------------ #
# 1メッシュ分の可視化
# ------------------------------------------------------------------ #

def plot_mesh(mesh_id, raw, filled, h0, h1, label_str, rank, save_path):
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(
        f"{label_str}  mesh_id={mesh_id}  "
        f"[H0点数={len(h0)}, H1点数={len(h1)}]  (H1ノルム順位 {rank})",
        fontsize=13, fontweight="bold"
    )

    gs = gridspec.GridSpec(2, 5, figure=fig,
                           hspace=0.45, wspace=0.35)

    # ---- ① DEM 2D heatmap（左上, 2行分）----
    ax_dem = fig.add_subplot(gs[:, 0])
    raw_plot = np.where(raw == NODATA_VAL, np.nan, raw)
    im = ax_dem.imshow(raw_plot, cmap="terrain", origin="upper")
    plt.colorbar(im, ax=ax_dem, label="標高 [m]", fraction=0.046)
    ax_dem.set_title("① DEM パッチ", fontsize=11)
    ax_dem.set_xlabel("列 [px × 5m]")
    ax_dem.set_ylabel("行 [px × 5m]")

    # ---- ① DEM 3D surface（右上エリアを使う）----
    ax_3d = fig.add_subplot(gs[0, 1:3], projection="3d")
    rows, cols = raw_plot.shape
    X = np.arange(cols)
    Y = np.arange(rows)
    X, Y = np.meshgrid(X, Y)
    Z = np.nan_to_num(raw_plot, nan=np.nanmin(raw_plot))
    ax_3d.plot_surface(X, Y, Z, cmap="terrain", linewidth=0,
                       antialiased=False, alpha=0.85)
    ax_3d.set_title("① DEM 3D", fontsize=11)
    ax_3d.set_xlabel("列", fontsize=8)
    ax_3d.set_ylabel("行", fontsize=8)
    ax_3d.set_zlabel("標高 [m]", fontsize=8)
    ax_3d.tick_params(labelsize=7)

    # ---- ② H0 Persistence Diagram ----
    ax_pd0 = fig.add_subplot(gs[0, 3])
    if len(h0):
        birth0 = h0[:, 0]; death0 = h0[:, 1]
        pers0  = death0 - birth0
        sc = ax_pd0.scatter(birth0, death0, c=pers0, cmap="Reds",
                            s=20, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax_pd0, label="Persistence", fraction=0.046)
        lim = max(death0.max(), birth0.max()) * 1.05
        ax_pd0.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
        ax_pd0.set_xlim(0, lim); ax_pd0.set_ylim(0, lim)
    ax_pd0.set_title(f"② H0 Persistence Diagram\n({len(h0)}点)", fontsize=10)
    ax_pd0.set_xlabel("Birth（標高）[m]"); ax_pd0.set_ylabel("Death（標高）[m]")

    # ---- ② H1 Persistence Diagram ----
    ax_pd1 = fig.add_subplot(gs[0, 4])
    if len(h1):
        birth1 = h1[:, 0]; death1 = h1[:, 1]
        pers1  = death1 - birth1
        sc = ax_pd1.scatter(birth1, death1, c=pers1, cmap="Blues",
                            s=20, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax_pd1, label="Persistence", fraction=0.046)
        lim = max(death1.max(), birth1.max()) * 1.05
        ax_pd1.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.4)
        ax_pd1.set_xlim(0, lim); ax_pd1.set_ylim(0, lim)
    ax_pd1.set_title(f"② H1 Persistence Diagram\n({len(h1)}点)", fontsize=10)
    ax_pd1.set_xlabel("Birth（標高）[m]"); ax_pd1.set_ylabel("Death（標高）[m]")

    # ---- ③ H0 Persistence Image ----
    ax_pi0 = fig.add_subplot(gs[1, 3])
    pi0 = make_persistence_image(h0)
    im0 = ax_pi0.imshow(pi0, cmap="Reds", origin="lower", aspect="auto")
    plt.colorbar(im0, ax=ax_pi0, label="密度", fraction=0.046)
    ax_pi0.set_title("③ H0 Persistence Image", fontsize=10)
    ax_pi0.set_xlabel("Birth 軸"); ax_pi0.set_ylabel("Persistence 軸")

    # ---- ③ H1 Persistence Image ----
    ax_pi1 = fig.add_subplot(gs[1, 4])
    pi1 = make_persistence_image(h1)
    im1 = ax_pi1.imshow(pi1, cmap="Blues", origin="lower", aspect="auto")
    plt.colorbar(im1, ax=ax_pi1, label="密度", fraction=0.046)
    ax_pi1.set_title("③ H1 Persistence Image", fontsize=10)
    ax_pi1.set_xlabel("Birth 軸"); ax_pi1.set_ylabel("Persistence 軸")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    print("データ読み込み中...")
    merged = load_data()

    flooded_ids    = select_representatives(merged, label=1, n=10)
    nonflooded_ids = select_representatives(merged, label=0, n=10)

    print(f"  浸水あり代表メッシュ:  {flooded_ids}")
    print(f"  浸水なし代表メッシュ: {nonflooded_ids}")

    dem_src  = rasterio.open(BASE_DIR / "output" / "dem_merged.tif")
    mesh_crs = merged.crs.to_string() if hasattr(merged, "crs") else "EPSG:32653"

    # GeoDataFrame に変換
    mesh_gdf = gpd.GeoDataFrame(merged, geometry="geometry")
    mesh_crs = mesh_gdf.crs.to_string()
    dem_crs  = dem_src.crs.to_string()

    groups = [
        ("flooded",    flooded_ids,    "【浸水あり】"),
        ("nonflooded", nonflooded_ids, "【浸水なし】"),
    ]

    for group_name, id_list, label_str in groups:
        print(f"\n{label_str} 10件を描画中...")
        for rank, mesh_id in enumerate(id_list, 1):
            geom = mesh_gdf.loc[mesh_gdf["mesh_id"] == mesh_id, "geometry"].values[0]
            raw, filled = get_patch(dem_src, geom, mesh_crs, dem_crs)

            if filled is None:
                print(f"  [{rank}] mesh_id={mesh_id}: 有効データ不足 → スキップ")
                continue

            h0, h1 = compute_diagrams(filled)

            save_path = OUT_DIR / f"{group_name}_{rank:02d}_mesh{mesh_id}.png"
            plot_mesh(mesh_id, raw, filled, h0, h1, label_str, rank, save_path)
            print(f"  [{rank}/10] mesh_id={mesh_id} → {save_path.name}  "
                  f"(H0={len(h0)}点, H1={len(h1)}点)")

    print(f"\n完了。出力先: {OUT_DIR}")


if __name__ == "__main__":
    main()
