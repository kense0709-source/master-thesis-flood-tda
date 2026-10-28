"""Phase H 手順4-①：候補12水系で、5m DEM と 10m DEM がどれくらい揃っているかを調べる

国土地理院の標高タイル（ログイン不要、https://maps.gsi.go.jp/development/demtile.html）を使う。
  dem5a_png（z15）：5m、航空レーザ測量
  dem5b_png（z15）：5m、写真測量
  dem_png（z14）：10m（DEM10B）、全国。タイル名は dem10b_png ではない（404 になる）
各水系の使えるマス（label_2km が 1 または 0）から 300 個を抜き取り、マスの中心の画素に値があるかを見る。
1 のマスと 0 のマスを分けて数える（5m は平野に偏るので、1 のマスだけ揃っても足りない）。
出力：output/hazard/H5_dem_coverage.csv
"""
import io
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
H3B = ROOT / "output/hazard/H3b"
OUT = ROOT / "output/hazard/H5_dem_coverage.csv"
CAND = {"890906": "筑後川", "830304": "荒川", "810103": "石狩川", "810108": "十勝川", "820211": "最上川",
        "860607": "九頭竜川", "850505": "天竜川", "860603": "大和川", "860610": "円山川", "890911": "球磨川",
        "830306": "鶴見川", "220079": "太田川"}
LAYERS = {"dem5a": 15, "dem5b": 15, "dem": 14}
N = 300
URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}_png/{z}/{x}/{y}.png"
session = requests.Session()
session.headers["User-Agent"] = "master-thesis-research (Kyushu Inst. Tech.)"
_cache = {}


def tile_pixel(lon, lat, z):
    n = 2 ** z
    fx = (lon + 180) / 360 * n
    fy = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return int(fx), int(fy), int((fx % 1) * 256), int((fy % 1) * 256)


def fetch(layer, z, x, y):
    key = (layer, z, x, y)
    if key not in _cache:
        for _ in range(3):
            try:
                r = session.get(URL.format(layer=layer, z=z, x=x, y=y), timeout=30)
                break
            except requests.RequestException:
                time.sleep(2)
        else:
            _cache[key] = None
            return None
        if r.status_code == 404:
            _cache[key] = "none"
        else:
            a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(np.int64)
            _cache[key] = a[..., 0] * 65536 + a[..., 1] * 256 + a[..., 2]
        time.sleep(0.05)
    return _cache[key]


def has_value(layer, lon, lat):
    z = LAYERS[layer]
    x, y, px, py = tile_pixel(lon, lat, z)
    t = fetch(layer, z, x, y)
    if t is None:
        return np.nan
    if isinstance(t, str):
        return False
    return bool(t[py, px] != 2 ** 23)          # 2^23 は「値なし」


cells = pd.concat(pd.read_parquet(f, columns=["suikei", "lon", "lat", "label_2km"])
                  for f in sorted(H3B.glob("basin_*.parquet")))
cells = cells[cells["suikei"].isin(CAND) & (cells["label_2km"] != "使わない")]
rows = []
for code, name in CAND.items():
    c = cells[cells["suikei"] == code]
    s = c.sample(min(N, len(c)), random_state=0)
    for layer in LAYERS:
        with ThreadPoolExecutor(4) as ex:
            v = list(ex.map(lambda r: has_value(layer, r.lon, r.lat), s.itertuples()))
        s[layer] = v
    s["5m"] = s["dem5a"].fillna(False) | s["dem5b"].fillna(False)
    row = {"水系": name, "抜き取り": len(s)}
    for lab in ["1", "0"]:
        t = s[s["label_2km"] == lab]
        row[f"{lab}_数"] = len(t)
        for col in ["dem5a", "5m", "dem"]:
            row[f"{lab}_{col}"] = round(float(t[col].astype(float).mean()), 3) if len(t) else None
    rows.append(row)
    print(row, flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
