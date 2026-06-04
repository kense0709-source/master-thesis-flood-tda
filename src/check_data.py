"""
データ目視確認用スクリプト
DEM・500mメッシュ・浸水推定図を地図上に重ねて表示する
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import show
import geopandas as gpd

BASE_DIR = Path(__file__).parent.parent
DEM_PATH   = BASE_DIR / "output" / "dem_merged.tif"
MESH_PATH  = BASE_DIR / "output" / "mesh_500m.gpkg"
FLOOD_2020 = list((BASE_DIR / "浸水推定図" / "2020_第2報").rglob("*.geojson"))[0]
FLOOD_2023 = list((BASE_DIR / "浸水推定図" / "2023").rglob("*.geojson"))[0]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("データ目視確認", fontsize=14)

# ---- 1. DEM全体 ----
ax = axes[0]
ax.set_title("① DEM（標高）")
with rasterio.open(DEM_PATH) as src:
    data = src.read(1)
    bounds = src.bounds
    nodata = src.nodata

data_vis = np.where(data == nodata, np.nan, data)
im = ax.imshow(
    data_vis,
    cmap="terrain",
    extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
    origin="upper",
    vmin=0, vmax=1000,
)
plt.colorbar(im, ax=ax, label="標高 [m]", shrink=0.7)
ax.set_xlabel("経度")
ax.set_ylabel("緯度")

# ---- 2. 500mメッシュ ----
ax = axes[1]
ax.set_title("② 500mメッシュ（標高平均）")
gdf_mesh = gpd.read_file(MESH_PATH).to_crs("EPSG:6668")
gdf_mesh.plot(
    column="elev_mean",
    cmap="terrain",
    ax=ax,
    legend=True,
    legend_kwds={"label": "標高平均 [m]", "shrink": 0.7},
    vmin=0, vmax=1000,
    edgecolor="none",
)
ax.set_xlabel("経度")
ax.set_ylabel("緯度")

# ---- 3. 浸水推定図の重ね合わせ ----
ax = axes[2]
ax.set_title("③ メッシュ + 浸水推定図（2020・2023）")
gdf_mesh.plot(
    column="elev_mean",
    cmap="Greys",
    ax=ax,
    vmin=0, vmax=500,
    edgecolor="none",
    alpha=0.6,
)
flood_2020 = gpd.read_file(FLOOD_2020).to_crs("EPSG:6668")
flood_2023 = gpd.read_file(FLOOD_2023).to_crs("EPSG:6668")
flood_2020.plot(ax=ax, color="blue",   alpha=0.5, label="2020年浸水")
flood_2023.plot(ax=ax, color="orange", alpha=0.5, label="2023年浸水")
ax.legend(loc="lower right", fontsize=8)
ax.set_xlabel("経度")
ax.set_ylabel("緯度")

plt.tight_layout()
out_path = BASE_DIR / "output" / "check_data.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"保存: {out_path}")
plt.show()
