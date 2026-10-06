"""
PI の変種：birth（標高）の軸を変えて、低地に解像度を割く

背景（progress.md 2026-10-05、逆解析）:
  08b の PI は birth の1画素が 52m（範囲 0〜1034m を20分割）で、
  筑後川の氾濫原（標高 0〜20m 程度）がまるごと最下段1画素に入る。
  PI は「低地か台地か」しか区別できず、低地の中で浸水する所としない所を表せない。

方法:
  1. バッファ付きパッチの PD（中心で生まれたペア）を全メッシュ分計算して保存する（08b と同じ関数）
     → 以後、軸の変換を変えた PI は PD から数分で作り直せる
  2. birth を変換してから [0,1] に正規化し、20×20 の PI にする（persistence 軸は 08b と同じ）

変種:
  log … birth' = log(1 + max(標高, 0))。0〜20m が約9画素に広がる

出力:
  output/diagrams_buffered.parquet         … mesh_id, dim, birth, death（全ペア）
  output/tda_{変種}_features.parquet        … mesh_id, h0{t}_000〜, h1{t}_000〜（t = 変種の頭文字）
  output/tda_{変種}_imager.json             … 正規化に使った範囲（逆解析で座標を戻すのに使う）

実行:
  python -u src/08c_tda_variants.py --variant log
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "output"
DG_PATH = OUT_DIR / "diagrams_buffered.parquet"

_spec = importlib.util.spec_from_file_location("b08", Path(__file__).parent / "08b_tda_buffered.py")
b08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b08)
IMG = b08.IMG
PERS_Q = b08.PERS_Q

# birth の変換（標高 [m] → 変換後の値）と、その逆変換（逆解析で座標を m に戻す用）
VARIANTS = {
    "log": (lambda b: np.log1p(np.maximum(b, 0.0)), lambda t: np.expm1(t)),
}


def compute_diagrams() -> pd.DataFrame:
    """全メッシュの PD を計算して保存（既にあれば読み込む）"""
    if DG_PATH.exists():
        return pd.read_parquet(DG_PATH)
    t0 = time.time()
    mesh = gpd.read_file(b08.tda08.MESH_PATH)
    mcrs = mesh.crs.to_string()
    parts = []
    with rasterio.open(b08.tda08.DEM_PATH) as src:
        dcrs = src.crs.to_string()
        for k, row in mesh.iterrows():
            patch, center = b08.read_buffered(src, row.geometry, mcrs, dcrs)
            z = b08.tda08.preprocess_patch(patch)
            if z is not None:
                h0, h1 = b08.diagrams_born_in_center(z, patch != b08.NODATA, center)
                for dim, dg in ((0, h0), (1, h1)):
                    if len(dg):
                        parts.append(pd.DataFrame({"mesh_id": row["mesh_id"], "dim": dim,
                                                   "birth": dg[:, 0], "death": dg[:, 1]}))
            if (k + 1) % 1000 == 0:
                print(f"  PD {k + 1}/{len(mesh)}  経過 {(time.time() - t0) / 60:.1f} 分", flush=True)
    dg = pd.concat(parts, ignore_index=True)
    dg.to_parquet(DG_PATH, index=False)
    print(f"保存: {DG_PATH}（{len(dg):,} ペア、{(time.time() - t0) / 60:.0f} 分）", flush=True)
    return dg


def build(dg: pd.DataFrame, variant: str) -> None:
    fwd, _ = VARIANTS[variant]
    t = variant[0]
    dg = dg.assign(tb=fwd(dg["birth"].values), pers=dg["death"] - dg["birth"])
    axes = {}
    for dim in (0, 1):
        s = dg[dg.dim == dim]
        axes[f"H{dim}"] = {"b0": float(s.tb.min()), "b1": float(s.tb.max()),
                           "p1": float(np.percentile(s.pers, PERS_Q)), "n_pairs": int(len(s)),
                           "birth_transform": variant}
    (OUT_DIR / f"tda_{variant}_imager.json").write_text(
        json.dumps(axes, ensure_ascii=False, indent=1), encoding="utf-8")
    for h, a in axes.items():
        _, inv = VARIANTS[variant]
        edges = inv(np.linspace(a["b0"], a["b1"], IMG + 1))
        print(f"{h}: birth 画素の境界 [m] = {np.round(edges[:8], 1)} …、persistence 0〜{a['p1']:.2f} m")

    imager = b08.make_imager()
    mesh_ids = gpd.read_file(b08.tda08.MESH_PATH, columns=["mesh_id"],
                             ignore_geometry=True)["mesh_id"].values
    groups = {k: g for k, g in dg.groupby(["mesh_id", "dim"])}
    rows = []
    for mid in mesh_ids:
        rec = {"mesh_id": mid}
        for dim in (0, 1):
            a = axes[f"H{dim}"]
            g = groups.get((mid, dim))
            if g is None:
                v = np.zeros(IMG * IMG, np.float32)
            else:
                b = np.clip((g.tb.values - a["b0"]) / (a["b1"] - a["b0"]), 0.0, 1.0 - 1e-9)
                p = np.minimum(g.pers.values, a["p1"]) / a["p1"]
                v = b08.tda08.diagram_to_vec(np.column_stack([b, b + p]), imager, IMG)
            rec.update({f"h{dim}{t}_{i:03d}": x for i, x in enumerate(v)})
        rows.append(rec)
    out = pd.DataFrame(rows)
    path = OUT_DIR / f"tda_{variant}_features.parquet"
    out.to_parquet(path, index=False)
    print(f"保存: {path}（{len(out)} 件）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="log")
    args = ap.parse_args()
    dg = compute_diagrams()
    build(dg, args.variant)


if __name__ == "__main__":
    main()
