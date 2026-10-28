"""Phase H 手順4-②：DEM の作り方で TDA の値がどれだけ変わるかを筑後川で比べる

比べる DEM（method_choices.md ステップ10、10-28）
  5m      ：dem5a（航空レーザ）を UTM 5m に（bilinear）
  10m間引き：dem5a を UTM 10m に平均で間引く      ← Phase H の本筋
  DEM10B  ：dem（地形図の等高線から作った 10m）を UTM 10m に（bilinear）
TDA は 08b と同じ作り方：500m マス＋周囲 500m（1.5km 四方）で sublevel の PH を計算し、
birth が中心マスにあり death が壁でない有限ペアだけ使う。値なしは「最高点＋500m の壁」で埋める。
標本：H3b の福岡側の筑後川（水系 890906）から 1 と 0 を 150 マスずつ。
5m で値なし（水路・ため池など）がパッチの 20% を超えるものは除く。値なしは 08b と同じく壁として扱う
（初版は値なしが 1 画素でもあれば除いたため、300 中 284 が落ちた）。
出力：output/hazard/H6_dem_check_cells.csv、H6_dem_check.txt
"""
import sys
from pathlib import Path

import gudhi
import numpy as np
import pandas as pd
from rasterio.warp import Resampling
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import dem_tiles  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/hazard"
N_EACH = 150
MAX_NAN = 0.2
VARIANTS = {"5m": ("dem5a", 5.0, Resampling.bilinear),
            "10m間引き": ("dem5a", 10.0, Resampling.average),
            "DEM10B": ("dem", 10.0, Resampling.bilinear)}


def tda_summary(z: np.ndarray, res: float) -> dict:
    valid = ~np.isnan(z)
    zz = np.where(valid, z, np.nanmax(z) + 500.0).astype(np.float64)
    nr = zz.shape[0]
    c0, c1 = int(round(500 / res)), int(round(1000 / res))
    cc = gudhi.CubicalComplex(top_dimensional_cells=zz)
    cc.compute_persistence()
    regular, _ = cc.cofaces_of_persistence_pairs()
    zf, vf = zz.ravel(order="F"), valid.ravel(order="F")
    out = {}
    for dim in (0, 1):
        if dim < len(regular) and len(regular[dim]):
            bi, di = regular[dim][:, 0], regular[dim][:, 1]
            rr, cl = bi % nr, bi // nr
            keep = (rr >= c0) & (rr < c1) & (cl >= c0) & (cl < c1) & vf[di]
            p = zf[di[keep]] - zf[bi[keep]]
            p = p[p > 0]
        else:
            p = np.empty(0)
        h = f"h{dim}"
        out[f"{h}_n01"] = int((p >= 0.1).sum())
        out[f"{h}_n05"] = int((p >= 0.5).sum())
        out[f"{h}_n1"] = int((p >= 1.0).sum())
        out[f"{h}_sum"] = float(p[p >= 0.1].sum())
        out[f"{h}_max"] = float(p.max()) if len(p) else 0.0
    out["elev_mean_center"] = float(np.nanmean(z[c0:c1, c0:c1]))
    return out


cells = pd.read_parquet(ROOT / "output/hazard/H3b/basin_40.parquet")
cells = cells[cells["suikei"] == "890906"]
smp = pd.concat([cells[cells["label_2km"] == lab].sample(N_EACH, random_state=0) for lab in ["1", "0"]])
epsg = int(smp["epsg"].iloc[0])
rows = []
for i, r in enumerate(smp.itertuples()):
    x0, ytop = r.x - 750.0, r.y + 750.0
    rec = {"row": r.row, "col": r.col, "label": int(r.label_2km), "x": r.x, "y": r.y}
    grids = {}
    for name, (layer, res, rs) in VARIANTS.items():
        n = int(1500 / res)
        grids[name] = dem_tiles.to_grid(layer, f"EPSG:{epsg}", x0, ytop, res, n, n, rs)
    rec["nan5m"] = float(np.isnan(grids["5m"]).mean())
    if rec["nan5m"] > MAX_NAN:
        rows.append(rec)
        continue
    for name, (layer, res, rs) in VARIANTS.items():
        for k, v in tda_summary(grids[name], res).items():
            rec[f"{name}:{k}"] = v
    rows.append(rec)
    if i % 50 == 0:
        print(i, flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / "H6_dem_check_cells.csv", index=False, encoding="utf-8-sig")
ok = d[d["nan5m"] <= MAX_NAN]
metrics = ["h0_n01", "h0_n05", "h0_n1", "h0_sum", "h0_max", "h1_n01", "elev_mean_center"]
lines = [f"標本 {len(d)}（5m の値なしが 20% 超で除外 {int((d['nan5m'] > MAX_NAN).sum())}）。使用 {len(ok)}：1 が {int(ok.label.sum())}、0 が {int((ok.label == 0).sum())}",
         "", "■ 中央値（1 のマス / 0 のマス）"]
for m in metrics:
    s = "  ".join(f"{v}: {ok.loc[ok.label == 1, f'{v}:{m}'].median():.2f} / {ok.loc[ok.label == 0, f'{v}:{m}'].median():.2f}"
                  for v in VARIANTS)
    lines.append(f"{m:18s} {s}")
lines += ["", "■ 5m との順位相関（Spearman）"]
for m in metrics:
    s = "  ".join(f"{v}: {spearmanr(ok[f'5m:{m}'], ok[f'{v}:{m}']).statistic:.2f}" for v in ["10m間引き", "DEM10B"])
    lines.append(f"{m:18s} {s}")
lines += ["", "■ その値1つで 1 と 0 を見分ける力（AUC。0.5 は当てずっぽう、1 から遠いほど逆向きにも使える）"]
for m in metrics:
    s = "  ".join(f"{v}: {roc_auc_score(ok.label, ok[f'{v}:{m}']):.3f}" for v in VARIANTS)
    lines.append(f"{m:18s} {s}")
txt = "\n".join(lines)
(OUT / "H6_dem_check.txt").write_text(txt, encoding="utf-8")
print(txt)
