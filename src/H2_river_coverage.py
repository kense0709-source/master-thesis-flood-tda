"""Phase H 手順2：川ごとに「想定区域（L2）が計算済みか」を照合できるか確かめる

国土数値情報 W05 河川（川の中心線、河川コード 10 桁）と、A31a の掲載河川一覧（A31_R7_datalist.xlsx）を
河川コードで突き合わせ、県ごとに「計算済みの川の割合」を出す。
出力：output/hazard/H2_river_coverage.txt
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HZ = ROOT / "ハザードマップ"
OUT = ROOT / "output/hazard"
PREFS = {"40": "福岡", "41": "佐賀", "42": "長崎", "43": "熊本"}
KUBUN = {1: "一級直轄", 2: "一級指定", 3: "二級", 4: "指定区間外", 5: "一級直轄(対象兼)", 6: "一級指定(対象兼)",
         7: "二級(対象兼)", 8: "指定区間外(対象兼)", 0: "不明"}

lst = pd.read_excel(HZ / "A31_R7_datalist.xlsx", sheet_name="想定最大規模", dtype=str)
lst.columns = [str(c).replace("\n", "") for c in lst.columns]
lst = lst.rename(columns={lst.columns[13]: "ファイル名称"})
lst = lst[lst["ファイル名称"].notna()]           # 「該当なし」の行（ファイルが無い）を除く
done = set(lst["河川コード"].str.strip().str.zfill(10))

out = open(OUT / "H2_river_coverage.txt", "w", encoding="utf-8")
p = lambda *a: print(*a, file=out, flush=True)
p("L2 掲載河川コード数（全国）", len(done))

rows = []
for code, name in PREFS.items():
    zp = HZ / f"raw_w05/W05-07_{code}_GML.zip"
    g = gpd.read_file(f"zip://{zp}!W05-07_{code}-g_Stream.shp", encoding="cp932")
    if code == "40":
        p("W05 の列", list(g.columns)); p(g.head(3).drop(columns="geometry").to_string())
    g = g.set_crs(6668, allow_override=True).to_crs(6670 if code != "43" else 6670)  # 長さ用（第II系、九州）
    g["rc"] = g["W05_002"].astype(str).str.strip().str.zfill(10)
    g["kubun"] = pd.to_numeric(g["W05_003"], errors="coerce").fillna(0).astype(int).map(KUBUN)
    g["done"] = g["rc"].isin(done)
    g["km"] = g.length / 1000
    per = g.groupby("rc").agg(km=("km", "sum"), done=("done", "first"), kubun=("kubun", "first"))
    p(f"===== {name}  川 {len(per)} 本・総延長 {g['km'].sum():.0f} km")
    t = per.groupby("kubun").agg(n=("km", "size"), n_done=("done", "sum"), km=("km", "sum"),
                                 km_done=("km", lambda s: s[per.loc[s.index, "done"]].sum()))
    t["本数の割合"] = (t["n_done"] / t["n"]).round(3)
    t["延長の割合"] = (t["km_done"] / t["km"]).round(3)
    p(t.round(1).to_string())
    p("全体：本数", round(per["done"].mean(), 3), " 延長", round(g.loc[g["done"], "km"].sum() / g["km"].sum(), 3))
    # 逆向き：L2 一覧にあるこの県の川のうち、W05 に見つかるもの
    pref_codes = set(lst.loc[lst["河川管理者名"].str.contains(name), "河川コード"].str.zfill(10))
    p("L2 一覧（この県の管理）の川", len(pref_codes), "うち W05 にある", len(pref_codes & set(per.index)))
    rows.append(per.assign(pref=name))

out.close()
pd.concat(rows).reset_index().to_parquet(OUT / "H2_river_coverage.parquet", index=False)
print(open(OUT / "H2_river_coverage.txt", encoding="utf-8").read())
