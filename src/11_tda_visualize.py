"""
TDA visualization: DEM / Persistence Diagram / Persistence Image

Outputs 20 PNG files (10 flooded + 10 non-flooded) to output/tda_vis/.
Representative meshes selected by evenly-spaced H1-norm rank.

Layout per figure (2 rows × 4 cols):
  col 0 (full height) : DEM 2D heatmap
  col 1 (full height) : DEM 3D surface
  row0 col 2-3        : H0 / H1 Persistence Diagram  (birth-persistence space)
  row1 col 2-3        : H0 / H1 Persistence Image
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
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
OUT_DIR  = BASE_DIR / "output" / "tda_vis"
OUT_DIR.mkdir(exist_ok=True)

NODATA_VAL = -9999.0
IMG_SIZE   = 20


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_data():
    mesh    = gpd.read_file(BASE_DIR / "output" / "mesh_500m.gpkg")
    tda     = pd.read_parquet(BASE_DIR / "output" / "tda_features.parquet")
    labels  = pd.read_parquet(BASE_DIR / "output" / "mesh_labels.parquet")
    lab2020 = labels[labels["event"] == "2020"][["mesh_id", "label"]]

    h1_cols = [f"h1_{i:03d}" for i in range(400)]
    tda["h1_norm"] = np.linalg.norm(tda[h1_cols].values, axis=1)

    merged = tda[["mesh_id", "h1_norm"]].merge(lab2020, on="mesh_id")
    merged = merged.merge(mesh[["mesh_id", "geometry"]], on="mesh_id")
    return merged


def select_representatives(df, label: int, n: int = 10) -> list:
    sub = df[df["label"] == label].sort_values("h1_norm").reset_index(drop=True)
    indices = np.linspace(0, len(sub) - 1, n, dtype=int)
    return sub.iloc[indices]["mesh_id"].tolist()


# ------------------------------------------------------------------ #
# DEM patch
# ------------------------------------------------------------------ #

def get_patch(dem_src, geom, mesh_crs, dem_crs):
    l, b, r, t = transform_bounds(mesh_crs, dem_crs, *geom.bounds)
    window = window_from_bounds(l, b, r, t, dem_src.transform)
    patch = dem_src.read(1, window=window, boundless=True,
                         fill_value=NODATA_VAL).astype(np.float64)
    valid = patch != NODATA_VAL
    if valid.sum() < 50:
        return None, None
    raw    = patch.copy()
    filled = patch.copy()
    filled[~valid] = patch[valid].max() + 500.0
    return raw, filled


# ------------------------------------------------------------------ #
# Persistence Diagram computation
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
# Persistence Image
# ------------------------------------------------------------------ #

def make_persistence_image(dgm, img_size=IMG_SIZE):
    if len(dgm) == 0:
        return np.zeros((img_size, img_size))
    pimgr = PersistenceImager(pixel_size=1.0)
    pimgr.fit([dgm])
    b_span = pimgr.birth_range[1] - pimgr.birth_range[0]
    p_span = pimgr.pers_range[1]  - pimgr.pers_range[0]
    pimgr.pixel_size = max(max(b_span, p_span) / img_size, 1e-3)
    img = pimgr.transform([dgm])[0]
    if img.shape != (img_size, img_size):
        from skimage.transform import resize
        img = resize(img, (img_size, img_size), anti_aliasing=True)
    return img


# ------------------------------------------------------------------ #
# Persistence Diagram helper (birth-persistence space)
# ------------------------------------------------------------------ #

def _plot_pd(ax, dgm, cmap, title, min_pers_frac=0.01):
    """
    Plot PD in birth-persistence space.
    Points with persistence < min_pers_frac * max_pers are drawn smaller/lighter
    to de-emphasize noise.
    """
    if len(dgm) == 0:
        ax.set_title(title, fontsize=10)
        ax.text(0.5, 0.5, "no points", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return

    birth = dgm[:, 0]
    pers  = dgm[:, 1] - dgm[:, 0]   # persistence = death - birth
    max_p = pers.max()
    threshold = max_p * min_pers_frac

    # Separate noise vs significant
    sig   = pers >= threshold
    noise = ~sig

    if noise.any():
        ax.scatter(birth[noise], pers[noise], s=8, alpha=0.25,
                   color="lightgray", edgecolors="none", zorder=1,
                   label=f"noise (<{min_pers_frac*100:.0f}%)")

    if sig.any():
        sc = ax.scatter(birth[sig], pers[sig], c=pers[sig], cmap=cmap,
                        s=25, alpha=0.85, edgecolors="none", zorder=2,
                        vmin=0, vmax=max_p)
        plt.colorbar(sc, ax=ax, label="Persistence [m]", fraction=0.046, pad=0.04)

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Birth [m]", fontsize=9)
    ax.set_ylabel("Persistence [m]", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=-max_p * 0.03)


# ------------------------------------------------------------------ #
# Main figure
# ------------------------------------------------------------------ #

def plot_mesh(mesh_id, raw, filled, h0, h1, label_str, rank, save_path):
    fig = plt.figure(figsize=(22, 10))

    # Use ASCII parentheses to avoid font issues with brackets
    fig.suptitle(
        f"{label_str}  mesh_id={mesh_id}  "
        f"[H0 pts={len(h0)}, H1 pts={len(h1)}]  (H1-norm rank: {rank}/10)",
        fontsize=13, fontweight="bold", y=0.99
    )

    # 2 rows x 4 cols; first 2 cols are DEM panels
    gs = gridspec.GridSpec(
        2, 4, figure=fig,
        hspace=0.55, wspace=0.45,
        width_ratios=[1.3, 1.5, 1.1, 1.1],
    )

    raw_plot = np.where(raw == NODATA_VAL, np.nan, raw)
    vmin = np.nanmin(raw_plot)
    vmax = np.nanmax(raw_plot)

    # ---- DEM 2D heatmap (col 0, full height) ----
    ax_dem = fig.add_subplot(gs[:, 0])
    im = ax_dem.imshow(raw_plot, cmap="terrain", origin="upper",
                       vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax_dem, label="Elevation [m]",
                      fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    ax_dem.set_title("DEM patch (2D)", fontsize=11, pad=6)
    ax_dem.set_xlabel("col [px x 5m]", fontsize=9)
    ax_dem.set_ylabel("row [px x 5m]", fontsize=9)
    ax_dem.tick_params(labelsize=8)

    # ---- DEM 3D surface (col 1, full height) ----
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")
    rows, cols = raw_plot.shape
    Xg, Yg = np.meshgrid(np.arange(cols), np.arange(rows))
    Z = np.nan_to_num(raw_plot, nan=vmin)
    ax_3d.plot_surface(Xg, Yg, Z, cmap="terrain",
                       linewidth=0, antialiased=False, alpha=0.9,
                       vmin=vmin, vmax=vmax)
    ax_3d.set_title("DEM patch (3D)", fontsize=11, pad=2)
    ax_3d.set_xlabel("col", fontsize=7, labelpad=1)
    ax_3d.set_ylabel("row", fontsize=7, labelpad=1)
    ax_3d.set_zlabel("Elev [m]", fontsize=7, labelpad=1)
    ax_3d.tick_params(labelsize=6)
    ax_3d.view_init(elev=30, azim=-120)

    # ---- H0 Persistence Diagram (row 0, col 2) ----
    ax_pd0 = fig.add_subplot(gs[0, 2])
    _plot_pd(ax_pd0, h0, cmap="Reds",
             title=f"H0 Persistence Diagram\n({len(h0)} pts)")

    # ---- H1 Persistence Diagram (row 0, col 3) ----
    ax_pd1 = fig.add_subplot(gs[0, 3])
    _plot_pd(ax_pd1, h1, cmap="Blues",
             title=f"H1 Persistence Diagram\n({len(h1)} pts)")

    # ---- H0 Persistence Image (row 1, col 2) ----
    ax_pi0 = fig.add_subplot(gs[1, 2])
    pi0 = make_persistence_image(h0)
    im0 = ax_pi0.imshow(pi0, cmap="Reds", origin="lower", aspect="auto")
    cb0 = plt.colorbar(im0, ax=ax_pi0, label="Density", fraction=0.046, pad=0.04)
    cb0.ax.tick_params(labelsize=8)
    ax_pi0.set_title("H0 Persistence Image", fontsize=10)
    ax_pi0.set_xlabel("Birth axis", fontsize=9)
    ax_pi0.set_ylabel("Persistence axis", fontsize=9)
    ax_pi0.tick_params(labelsize=8)

    # ---- H1 Persistence Image (row 1, col 3) ----
    ax_pi1 = fig.add_subplot(gs[1, 3])
    pi1 = make_persistence_image(h1)
    im1 = ax_pi1.imshow(pi1, cmap="Blues", origin="lower", aspect="auto")
    cb1 = plt.colorbar(im1, ax=ax_pi1, label="Density", fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=8)
    ax_pi1.set_title("H1 Persistence Image", fontsize=10)
    ax_pi1.set_xlabel("Birth axis", fontsize=9)
    ax_pi1.set_ylabel("Persistence axis", fontsize=9)
    ax_pi1.tick_params(labelsize=8)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    print("Loading data...")
    merged = load_data()

    flooded_ids    = select_representatives(merged, label=1, n=10)
    nonflooded_ids = select_representatives(merged, label=0, n=10)

    print(f"  Flooded meshes:     {flooded_ids}")
    print(f"  Non-flooded meshes: {nonflooded_ids}")

    dem_src  = rasterio.open(BASE_DIR / "output" / "dem_merged.tif")
    mesh_gdf = gpd.GeoDataFrame(merged, geometry="geometry")
    mesh_crs = mesh_gdf.crs.to_string()
    dem_crs  = dem_src.crs.to_string()

    groups = [
        ("flooded",    flooded_ids,    "FLOODED"),
        ("nonflooded", nonflooded_ids, "NON-FLOODED"),
    ]

    for group_name, id_list, label_str in groups:
        print(f"\n[{label_str}] rendering 10 figures...")
        for rank, mesh_id in enumerate(id_list, 1):
            geom = mesh_gdf.loc[mesh_gdf["mesh_id"] == mesh_id, "geometry"].values[0]
            raw, filled = get_patch(dem_src, geom, mesh_crs, dem_crs)

            if filled is None:
                print(f"  [{rank}] mesh_id={mesh_id}: insufficient valid data — skipped")
                continue

            h0, h1 = compute_diagrams(filled)

            save_path = OUT_DIR / f"{group_name}_{rank:02d}_mesh{mesh_id}.png"
            plot_mesh(mesh_id, raw, filled, h0, h1, label_str, rank, save_path)
            print(f"  [{rank}/10] mesh_id={mesh_id} -> {save_path.name}  "
                  f"(H0={len(h0)} pts, H1={len(h1)} pts)")

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
