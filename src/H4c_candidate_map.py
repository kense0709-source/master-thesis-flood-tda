"""Phase H 手順3-④：全国の地域候補（案）を地図にする

H4b の水系ごとの集計から選んだ候補の水系を、対象16道府県のマスの上に色分けして描く。
候補は CANDIDATES（水系コード → 表示名・地形の型）。ユーザー確認前の案。
出力：output/hazard/fig_H4c_candidates.png、output/hazard/H4c_candidates.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H3B = ROOT / "output/hazard/H3b"
plt.rcParams.update({"font.family": "Yu Gothic", "font.size": 9})

CANDIDATES = {
    "890906": ("筑後川", "低平地・干拓（九州）"),
    "830304": ("荒川", "低平地（関東、埼玉側）"),
    "810103": ("石狩川", "大平野・泥炭低地（北海道）"),
    "810108": ("十勝川", "台地と扇状地（北海道）"),
    "820211": ("最上川", "盆地の連なり（東北）"),
    "860607": ("九頭竜川", "扇状地と海岸平野（北陸）"),
    "850505": ("天竜川", "急流河川・下流扇状地（中部）"),
    "860603": ("大和川", "内陸盆地＋都市（近畿）"),
    "860610": ("円山川", "谷底平野・小盆地（近畿）"),
    "890911": ("球磨川", "山間盆地・狭い谷（九州）"),
    "830306": ("鶴見川", "都市の小流域（関東）"),
    "220079": ("太田川", "海岸平野・中小河川（静岡）"),
}

cells = pd.concat(pd.read_parquet(f, columns=["suikei", "lon", "lat", "label_2km"])
                  for f in sorted(H3B.glob("basin_*.parquet")))
basins = pd.read_csv(ROOT / "output/hazard/H4b_basins.csv", dtype={"suikei": str})
sel = basins[basins["suikei"].isin(CANDIDATES)].copy()
sel["表示名"] = sel["suikei"].map(lambda s: CANDIDATES[s][0])
sel["地形の型"] = sel["suikei"].map(lambda s: CANDIDATES[s][1])
sel.to_csv(ROOT / "output/hazard/H4c_candidates.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(10, 11), facecolor="#fcfcfb")
bg = cells.sample(frac=0.25, random_state=0)
ax.scatter(bg["lon"], bg["lat"], s=0.3, c="#dcdad2", linewidths=0)
cmap = plt.get_cmap("tab20")
for i, (code, (name, kind)) in enumerate(CANDIDATES.items()):
    d = cells[cells["suikei"] == code]
    ax.scatter(d["lon"], d["lat"], s=0.4, color=cmap(i * 2 % 20), linewidths=0)
    ax.annotate(f"{name}\n{kind}", (d["lon"].median(), d["lat"].median()), fontsize=8, fontweight="bold",
                ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
ax.set_aspect(1 / 0.8)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("地域の候補（案）。灰色は判定した16道府県", loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "output/hazard/fig_H4c_candidates.png", dpi=120, facecolor="#fcfcfb")
print(sel[["表示名", "地形の型", "県", "使える", "1", "0", "1の割合", "1のマスの平均標高m", "1のマスの平均傾斜度"]].to_string(index=False))
