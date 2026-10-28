"""Phase H 手順3-③：全国の対象県のマス（H3b）を水系ごとに集計し、地域の候補を出す

H4_region_candidates.py（九州4県）を全国用にしたもの。
  水系：国土数値情報 W07 流域メッシュ（100m）。マスの中心が入る 100m メッシュの水系コード
  地形：G04-a 標高・傾斜度 3次メッシュ（1km）の平均標高・平均傾斜
ラベルは D＝2km。水系が複数の県にまたがる場合は県ごとのマスを合わせる（対象外の県の部分は入らない）
出力：output/hazard/H4b_basins.csv、output/hazard/H3b/basin_{県コード}.parquet
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

ROOT = Path(__file__).resolve().parents[1]
HZ = ROOT / "ハザードマップ"
H3B = ROOT / "output/hazard/H3b"
LAB = "label_2km"


def shp_in(f, suffix):
    n = [i for i in zipfile.ZipFile(f).namelist() if i.endswith(suffix)][0]
    return f"zip://{f}!{n}"


def mesh_codes(lat, lon):
    a = lat * 1.5
    p, u = np.floor(a).astype(int), np.floor(lon - 100).astype(int)
    fa, fo = (a - p) * 8, (lon - 100 - u) * 8
    q, v = np.floor(fa).astype(int), np.floor(fo).astype(int)
    ga, go = (fa - q) * 10, (fo - v) * 10
    r, w = np.floor(ga).astype(int), np.floor(go).astype(int)
    s, x = np.floor((ga - r) * 10).astype(int), np.floor((go - w) * 10).astype(int)
    m1 = np.char.add(p.astype(str), np.char.zfill(u.astype(str), 2))
    m3 = np.char.add(np.char.add(np.char.add(np.char.add(m1, q.astype(str)), v.astype(str)), r.astype(str)), w.astype(str))
    m100 = np.char.add(np.char.add(m3, s.astype(str)), x.astype(str))
    return m1, m3, m100


def attach(cells):
    m1, m3, m100 = mesh_codes(cells["lat"].values, cells["lon"].values)
    cells = cells.assign(mesh1=m1, mesh3=m3, mesh100=m100)
    w07, g04 = [], []
    for m in sorted(set(m1)):
        fw = HZ / f"raw_w07/W07-09_{m}-jgd_GML.zip"
        if fw.exists():
            need = set(cells.loc[cells["mesh1"] == m, "mesh100"])
            df = pyogrio.read_dataframe(shp_in(fw, "_ValleyMesh.shp"), read_geometry=False,
                                        columns=["W07_001", "W07_002", "W07_004"], encoding="cp932")
            w07.append(df[df["W07_001"].isin(need)])
        fg = HZ / f"raw_g04/G04-a-11_{m}-jgd_GML.zip"
        if fg.exists():
            g04.append(pyogrio.read_dataframe(shp_in(fg, "TertiaryMesh.shp"), read_geometry=False,
                                              columns=["G04a_001", "G04a_002", "G04a_010"]))
    w = pd.concat(w07).rename(columns={"W07_001": "mesh100", "W07_002": "suikei", "W07_004": "suikei_name"})
    g = pd.concat(g04).rename(columns={"G04a_001": "mesh3", "G04a_002": "elev", "G04a_010": "slope"})
    for c in ["elev", "slope"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").where(lambda s_: s_ > -999)
    cells = cells.merge(w.drop_duplicates("mesh100"), on="mesh100", how="left")
    return cells.merge(g.drop_duplicates("mesh3"), on="mesh3", how="left")


if __name__ == "__main__":
    allc = []
    for f in sorted(H3B.glob("cells_*.parquet")):
        code = f.stem.split("_")[1]
        out = H3B / f"basin_{code}.parquet"
        if out.exists() and out.stat().st_mtime > f.stat().st_mtime:
            c = pd.read_parquet(out)
        else:
            c = attach(pd.read_parquet(f))
            c.to_parquet(out, index=False)
        print(code, len(c), "水系あり", int(c["suikei"].notna().sum()), flush=True)
        allc.append(c[["pref", "suikei", "suikei_name", LAB, "elev", "slope", "frac_l2", "lon", "lat"]])
    c = pd.concat(allc)
    c = c[c["suikei"].notna() & (c["suikei"] != "unknown")]
    c["is1"], c["use"] = c[LAB] == "1", c[LAB] != "使わない"
    g = c.groupby("suikei")
    t = pd.DataFrame({
        "水系名": g["suikei_name"].first(),
        "県": g["pref"].agg(lambda s_: "・".join(s_.value_counts().index[:3])),
        "マス": g.size(),
        "使える": g["use"].sum(),
        "1": g["is1"].sum(),
        "平均標高m": g["elev"].mean().round(0),
        "標高20m未満の割合": g["elev"].apply(lambda s_: (s_ < 20).mean()).round(2),
        "平均傾斜度": g["slope"].mean().round(1),
        "傾斜3度未満の割合": g["slope"].apply(lambda s_: (s_ < 3).mean()).round(2),
        "1のマスの平均標高m": c[c["is1"]].groupby("suikei")["elev"].mean().round(0),
        "1のマスの平均傾斜度": c[c["is1"]].groupby("suikei")["slope"].mean().round(1),
        "lon": g["lon"].median().round(3), "lat": g["lat"].median().round(3),
    })
    t["0"] = t["使える"] - t["1"]
    t["使える割合"] = (t["使える"] / t["マス"]).round(2)
    t["1の割合"] = (t["1"] / t["使える"].where(t["使える"] > 0)).round(3)
    t = t.reset_index().sort_values("1", ascending=False)
    t.to_csv(ROOT / "output/hazard/H4b_basins.csv", index=False, encoding="utf-8-sig")
    print(t.head(60).to_string(index=False))
