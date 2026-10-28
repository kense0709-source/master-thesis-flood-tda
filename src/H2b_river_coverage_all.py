"""Phase H 手順3-①：全都道府県で、一級・二級河川のうち L2（想定最大規模）が計算済みの割合

H2_river_coverage.py（九州4県）を全国に広げたもの。判定は method_choices.md ステップ10 と同じ：
  川＝W05 で個別の河川コード（末尾 0000 でない）があり、区間種別が一級直轄・一級指定・二級のもの
  計算済み＝河川コードが A31a 2025年度版の L2 掲載河川一覧（ファイルあり）に載っている
出力：output/hazard/H2b_coverage_by_pref.csv
"""
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HZ = ROOT / "ハザードマップ"
OUT = ROOT / "output/hazard"
LEGAL = {1, 2, 3, 5, 6, 7}
PREF = ("北海道 青森 岩手 宮城 秋田 山形 福島 茨城 栃木 群馬 埼玉 千葉 東京 神奈川 新潟 富山 石川 福井 山梨 長野 "
        "岐阜 静岡 愛知 三重 滋賀 京都 大阪 兵庫 奈良 和歌山 鳥取 島根 岡山 広島 山口 徳島 香川 愛媛 高知 "
        "福岡 佐賀 長崎 熊本 大分 宮崎 鹿児島 沖縄").split()

lst = pd.read_excel(HZ / "A31_R7_datalist.xlsx", sheet_name="想定最大規模", dtype=str)
lst.columns = [str(c).replace("\n", "") for c in lst.columns]
lst = lst[lst[lst.columns[13]].notna()]
done = set(lst["河川コード"].str.strip().str.zfill(10))
small = lst[lst["管理種別"] == "中小河川"].groupby("河川管理者コード").size()

rows = []
for f in sorted((HZ / "raw_w05").glob("W05-*_GML.zip")):
    code = f.stem.split("_")[1]
    shp = [n for n in zipfile.ZipFile(f).namelist() if n.endswith("_Stream.shp")][0]
    g = gpd.read_file(f"zip://{f}!{shp}", encoding="cp932", on_invalid="ignore")   # 三重に壊れた線が1本ある
    g = g[g.geometry.notna()]
    g = g.set_crs(4612, allow_override=True).to_crs(6933)        # 長さ用（全国一律の正積）
    g["rc"] = g["W05_002"].astype(str).str.strip().str.zfill(10)
    g["kubun"] = pd.to_numeric(g["W05_003"], errors="coerce").fillna(0).astype(int)
    g = g[g["kubun"].isin(LEGAL) & ~g["rc"].str.endswith("0000")]
    g["km"] = g.length / 1000
    g["done"] = g["rc"].isin(done)
    per = g.groupby("rc").agg(km=("km", "sum"), done=("done", "first"), kubun=("kubun", "first"))
    sec = per[per["kubun"].isin({2, 3, 6, 7})]                    # 県が管理する区間（一級指定・二級）
    rows.append({
        "code": code, "県": PREF[int(code) - 1],
        "一級二級_本数": len(per), "計算済み_本数": int(per["done"].sum()),
        "本数の割合": round(per["done"].mean(), 3),
        "延長の割合": round(per.loc[per["done"], "km"].sum() / per["km"].sum(), 3),
        "県管理区間_本数の割合": round(sec["done"].mean(), 3) if len(sec) else None,
        "L2中小河川_掲載数": int(small.get(code, 0)),
        "W05年度": f.stem.split("_")[0],
    })
    print(code, rows[-1]["県"], rows[-1]["本数の割合"], flush=True)

t = pd.DataFrame(rows).sort_values("本数の割合", ascending=False)
t.to_csv(OUT / "H2b_coverage_by_pref.csv", index=False, encoding="utf-8-sig")
print(t.to_string(index=False))
