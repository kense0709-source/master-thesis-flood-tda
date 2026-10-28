"""国土地理院の標高タイルを取得して、任意の格子（UTM など）にそろえる部品

  dem5a（z15）：5m、航空レーザ測量  ← Phase H の本筋（method_choices.md ステップ10、10-28）
  dem5b（z15）：5m、写真測量
  dem  （z14）：10m（DEM10B、地形図の等高線から作成）。タイル名は dem_png
タイルは DEM/gsi_tiles/{layer}/{z}/{x}/{y}.npy に標高（m、値なしは NaN）で保存して使い回す。
出典表記：国土地理院 標高タイル（https://maps.gsi.go.jp/development/demtile.html）
"""
import io
import math
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "DEM/gsi_tiles"
ZOOM = {"dem5a": 15, "dem5b": 15, "dem": 14}
URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}_png/{z}/{x}/{y}.png"
R = 6378137.0
_session = requests.Session()
_session.headers["User-Agent"] = "master-thesis-research (Kyushu Inst. Tech.)"


def _decode(png: bytes) -> np.ndarray:
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.int64)
    x = a[..., 0] * 65536 + a[..., 1] * 256 + a[..., 2]
    h = np.where(x < 2 ** 23, x, x - 2 ** 24) * 0.01
    return np.where(x == 2 ** 23, np.nan, h).astype(np.float32)


def tile(layer: str, x: int, y: int) -> np.ndarray | None:
    """1 枚のタイル（256×256、標高 m）。無いタイルは None"""
    z = ZOOM[layer]
    f = CACHE / layer / str(z) / str(x) / f"{y}.npy"
    if f.exists():
        return np.load(f)
    none = f.with_suffix(".none")
    if none.exists():
        return None
    for i in range(4):
        try:
            r = _session.get(URL.format(layer=layer, z=z, x=x, y=y), timeout=30)
            break
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    else:
        raise RuntimeError(f"tile fetch failed {layer} {x} {y}")
    f.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(0.03)
    if r.status_code == 404:
        none.touch()
        return None
    r.raise_for_status()
    a = _decode(r.content)
    np.save(f, a)
    return a


def _xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def mosaic(layer: str, lon0, lat0, lon1, lat1):
    """経緯度の範囲を覆うタイルをつなげる。返り値：(標高, EPSG:3857 の transform)"""
    z = ZOOM[layer]
    fx0, fy1 = _xy(lon0, lat0, z)
    fx1, fy0 = _xy(lon1, lat1, z)
    tx0, tx1, ty0, ty1 = int(fx0), int(fx1), int(fy0), int(fy1)
    out = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.nan, np.float32)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            a = tile(layer, tx, ty)
            if a is not None:
                out[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = a
    size = 2 * math.pi * R / 2 ** z / 256          # 1 画素（EPSG:3857 の m）
    west = -math.pi * R + tx0 * 256 * size
    north = math.pi * R - ty0 * 256 * size
    return out, from_origin(west, north, size, size)


def to_grid(layer: str, dst_crs, x0, y_top, res, nrow, ncol, resampling=Resampling.bilinear):
    """layer のタイルを、dst_crs の格子（左上 x0, y_top、画素 res m、nrow×ncol）にそろえる。値なしは NaN"""
    lon0, lat0, lon1, lat1 = transform_bounds(dst_crs, "EPSG:4326", x0, y_top - nrow * res, x0 + ncol * res, y_top,
                                              densify_pts=5)
    pad = 0.002
    src, st = mosaic(layer, lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad)
    dst = np.full((nrow, ncol), np.nan, np.float32)
    reproject(src, dst, src_transform=st, src_crs="EPSG:3857", src_nodata=np.nan,
              dst_transform=from_origin(x0, y_top, res, res), dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=resampling)
    return dst
