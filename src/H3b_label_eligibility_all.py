"""Phase H 手順3-②：全国の対象県で、500m マスごとに 1 / 0 / 使わない を判定する

H3_label_eligibility.py（九州4県、平面直角 II 系の1枚の格子）を全国用にしたもの。ルールは同じ
（method_choices.md ステップ10）：
  1       ：L2 の区域がマスの 0.5 以上
  0       ：それ以外で、マスから D km 以内に「L2 未計算の一級・二級河川」が無い
  使わない：それ以外
違い
  ・格子は県ごとに JGD2011 / UTM（県の重心の経度で 51〜55 帯）。500m の倍数にそろえる
  ・メモリを抑えるため、L2 は元の GeoJSON 1 個ずつを GeoParquet に変換し（ハザードマップ/l2_cache/）、
    100km 四方の区画ごとに範囲の重なるものだけ読む
  ・未計算の川は、県境の外（隣の県）の W05 も含めて判定する
  ・L2 は手元にある zip をすべて使う（県境の外の川の区域も入る）
使い方：python src/H3b_label_eligibility_all.py [県コード ...]   （省略時は対象の16道府県）
出力：output/hazard/H3b/cells_{県コード}.parquet、output/hazard/H3b_summary.csv
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from rasterio.features import rasterize

# 中部地整（85）に 200MB を超えるポリゴンがあり、GDAL の既定の上限で読めない
pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": "0"})
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
HZ = ROOT / "ハザードマップ"
CACHE = HZ / "l2_cache"
OUT = ROOT / "output/hazard/H3b"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(exist_ok=True)

CELL, SUB = 500.0, 20
PIX = CELL / SUB
TILE = 200                              # 区画 = 200 マス（100km）四方
DISTS = [1, 2, 5]
LEGAL = {1, 2, 3, 5, 6, 7}
TARGET = ["28", "06", "10", "17", "40", "26", "27", "42", "11", "24", "22", "18", "01", "14", "29", "43"]
PREF = ("北海道 青森 岩手 宮城 秋田 山形 福島 茨城 栃木 群馬 埼玉 千葉 東京 神奈川 新潟 富山 石川 福井 山梨 長野 "
        "岐阜 静岡 愛知 三重 滋賀 京都 大阪 兵庫 奈良 和歌山 鳥取 島根 岡山 広島 山口 徳島 香川 愛媛 高知 "
        "福岡 佐賀 長崎 熊本 大分 宮崎 鹿児島 沖縄").split()


def log(*a):
    print(*a, flush=True)


# --- ① L2 を GeoParquet に変換（済みなら飛ばす）。索引＝ファイルごとの範囲（EPSG:6668）---
def build_cache():
    idx_path = CACHE / "index.json"
    idx = json.loads(idx_path.read_text(encoding="utf-8")) if idx_path.exists() else {}
    for z in sorted((HZ / "raw").glob("A31a-25_*_GEOJSON.zip")):
        if z.name in idx:
            continue
        entries = []
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                name = i.filename.replace("\\", "/").split("/")[-1]
                if not name.startswith("A31a-20-"):
                    continue
                g = gpd.read_file(io.BytesIO(zf.read(i)))[["geometry"]]
                g = g[g.geometry.notna()]
                if g.crs is None:
                    g = g.set_crs(6668)
                g = g.to_crs(6668)
                out = CACHE / name.replace(".geojson", ".parquet")
                g.to_parquet(out, write_covering_bbox=True)
                entries.append({"file": out.name, "bounds": list(map(float, g.total_bounds))})
        idx[z.name] = entries
        idx_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        log("cache", z.name, len(entries))
    return [e for v in idx.values() for e in v]


def intersects(b, q):
    return not (b[2] < q[0] or b[0] > q[2] or b[3] < q[1] or b[1] > q[3])


# --- ② L2 未計算の一級・二級河川（全国、EPSG:6668）---
def undone_rivers():
    lst = pd.read_excel(HZ / "A31_R7_datalist.xlsx", sheet_name="想定最大規模", dtype=str)
    lst.columns = [str(c).replace("\n", "") for c in lst.columns]
    lst = lst[lst[lst.columns[13]].notna()]
    done = set(lst["河川コード"].str.strip().str.zfill(10))
    parts = []
    for f in sorted((HZ / "raw_w05").glob("W05-*_GML.zip")):
        shp = [n for n in zipfile.ZipFile(f).namelist() if n.endswith("_Stream.shp")][0]
        g = gpd.read_file(f"zip://{f}!{shp}", encoding="cp932", on_invalid="ignore",
                          columns=["W05_002", "W05_003", "W05_004"])
        g = g[g.geometry.notna()].set_crs(4612, allow_override=True).to_crs(6668)
        g["rc"] = g["W05_002"].astype(str).str.strip().str.zfill(10)
        k = pd.to_numeric(g["W05_003"], errors="coerce").fillna(0).astype(int)
        g = g[k.isin(LEGAL) & ~g["rc"].str.endswith("0000") & ~g["rc"].isin(done)]
        parts.append(g[["rc", "W05_004", "geometry"]].assign(w05=f.stem.split("_")[1]))
    return pd.concat(parts, ignore_index=True)


def utm_epsg(lon):
    zone = int((lon + 180) // 6) + 1
    return 6688 + (min(max(zone, 51), 55) - 51)          # JGD2011 / UTM 51〜55 帯


def run_pref(code, files, undone):
    n03 = HZ / f"raw_n03/N03-20250101_{code}_GML.zip"
    shp = [n for n in zipfile.ZipFile(n03).namelist() if n.endswith(".shp")][0]
    pref = gpd.read_file(f"zip://{n03}!{shp}").to_crs(6668)
    pref_geom = pref.geometry.union_all()
    epsg = utm_epsg(pref_geom.centroid.x)
    pg = gpd.GeoSeries([pref_geom], crs=6668).to_crs(epsg)
    xmin, ymin, xmax, ymax = pg.total_bounds
    x0, y0 = np.floor(xmin / CELL) * CELL, np.floor(ymin / CELL) * CELL
    NC, NR = int(np.ceil((xmax - x0) / CELL)), int(np.ceil((ymax - y0) / CELL))
    top = y0 + NR * CELL
    land = rasterize([(pg.iloc[0], 1)], out_shape=(NR, NC), transform=from_origin(x0, top, CELL, CELL),
                     fill=0, dtype="uint8").astype(bool)
    log(code, PREF[int(code) - 1], "EPSG", epsg, "grid", NR, NC, "land", int(land.sum()))

    t0 = time.time()
    frac = np.zeros((NR, NC), dtype=np.float32)
    for r0 in range(0, NR, TILE):
        for c0 in range(0, NC, TILE):
            nr, nc = min(TILE, NR - r0), min(TILE, NC - c0)
            if not land[r0:r0 + nr, c0:c0 + nc].any():
                continue
            tx0, ttop = x0 + c0 * CELL, top - r0 * CELL
            box = gpd.GeoSeries.from_xy([tx0, tx0 + nc * CELL], [ttop - nr * CELL, ttop], crs=epsg).to_crs(6668)
            q = [box.x.min() - 0.01, box.y.min() - 0.01, box.x.max() + 0.01, box.y.max() + 0.01]
            geoms = []
            for e in files:
                if intersects(e["bounds"], q):
                    g = gpd.read_parquet(CACHE / e["file"], bbox=q)
                    if len(g):
                        geoms.append(g.to_crs(epsg).geometry)
            if not geoms:
                continue
            arr = np.zeros((nr * SUB, nc * SUB), dtype="uint8")
            rasterize(((geom, 1) for s in geoms for geom in s), out=arr,
                      transform=from_origin(tx0, ttop, PIX, PIX))
            frac[r0:r0 + nr, c0:c0 + nc] = arr.reshape(nr, SUB, nc, SUB).sum(axis=(1, 3)) / (SUB * SUB)
            log("  tile", r0, c0, sum(len(s) for s in geoms), f"{time.time() - t0:.0f}s")

    # 県の範囲＋5km にかかる未計算の川
    # 県の形を buffer すると島の多い県（長崎）で30分以上かかるので、四角い範囲を広げるだけにする
    m = max(DISTS) * 1000 + CELL
    b = gpd.GeoSeries.from_xy([xmin - m, xmax + m], [ymin - m, ymax + m], crs=epsg).to_crs(6668).total_bounds
    u = undone.cx[b[0]:b[2], b[1]:b[3]].to_crs(epsg)
    log("  rivers", len(u), f"{time.time() - t0:.0f}s")
    near = {}
    for d in DISTS:
        buf = u.geometry.buffer(d * 1000)
        near[d] = (rasterize(((geom, 1) for geom in buf), out_shape=(NR, NC),
                             transform=from_origin(x0, top, CELL, CELL), fill=0, all_touched=True,
                             dtype="uint8").astype(bool) if len(buf) else np.zeros((NR, NC), bool))

    rr, cc = np.nonzero(land)
    cells = pd.DataFrame({"pref": code, "epsg": epsg, "row": rr, "col": cc,
                          "x": x0 + (cc + 0.5) * CELL, "y": top - (rr + 0.5) * CELL,
                          "frac_l2": frac[rr, cc]})
    for d in DISTS:
        cells[f"near_undone_{d}km"] = near[d][rr, cc]
        cells[f"label_{d}km"] = np.where(cells["frac_l2"] >= 0.5, "1",
                                         np.where(cells[f"near_undone_{d}km"], "使わない", "0"))
    ll = gpd.GeoSeries.from_xy(cells["x"], cells["y"], crs=epsg).to_crs(6668)
    cells["lon"], cells["lat"] = ll.x.values, ll.y.values
    cells.to_parquet(OUT / f"cells_{code}.parquet", index=False)

    row = {"code": code, "県": PREF[int(code) - 1], "epsg": epsg, "陸のマス": len(cells), "未計算の川_本数": u["rc"].nunique()}
    for d in DISTS:
        vc = cells[f"label_{d}km"].value_counts()
        row[f"1_{d}km"], row[f"0_{d}km"], row[f"使わない_{d}km"] = int(vc.get("1", 0)), int(vc.get("0", 0)), int(vc.get("使わない", 0))
        row[f"使える割合_{d}km"] = round((row[f"1_{d}km"] + row[f"0_{d}km"]) / len(cells), 3)
    log(row)
    return row


if __name__ == "__main__":
    codes = sys.argv[1:] or TARGET
    files = build_cache()
    undone = undone_rivers()
    log("未計算の一級・二級河川（全国）", undone["rc"].nunique())
    sp = ROOT / "output/hazard/H3b_summary.csv"
    done_rows = pd.read_csv(sp, dtype={"code": str}) if sp.exists() else pd.DataFrame()
    rows = []
    for c in codes:
        rows.append(run_pref(c, files, undone))
        allr = pd.concat([done_rows[~done_rows["code"].isin([r["code"] for r in rows])] if len(done_rows) else None,
                          pd.DataFrame(rows)])
        allr.to_csv(sp, index=False, encoding="utf-8-sig")
    log("done")
