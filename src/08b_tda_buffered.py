"""
Step 3: バッファ付きパッチで Persistence Image を再計算（実験計画.md §4.4）

08_tda_features.py は 500m メッシュを単独で切り出して PH を計算しており、
メッシュ境界をまたぐ窪地が切断され、death 標高が本来より低く出ていた。
ここでは 500m のバッファを付けた 1.5km 四方で PH を計算し、
birth セルが中心メッシュ内にあるペアだけを採用する。

08 からの変更点:
  1. バッファ付きパッチ・中心で生まれたペアのみ（本題）
     birth セルの位置は GUDHI の cofaces_of_persistence_pairs()（番号は Fortran 順）
  2. death セルが NODATA の壁にあるペアを捨てる
  3. Imager の座標範囲：08 は「最初の300件」で fit しており、DEM 端の NODATA の壁が作る
     深さ数百 m の偽ペアで範囲が引き伸ばされていた（初版の 08b では persistence 0〜513m、
     1画素 25.65m → 実際の窪地（ほぼ 5m 未満）が最下段の1画素に潰れる）。
     ここではランダム1000メッシュから birth の範囲と persistence の 99.5% 点を決め、
     [0,1] に正規化して 20×20 にそのまま収める（範囲は tda_buffered_imager.json に保存）
  DEM（dem_merged.tif）と NODATA の埋め方（最大値+500m）は 08 と同じ

出力:
  output/tda_buffered_features.parquet … mesh_id, h0b_000〜h0b_399, h1b_000〜h1b_399
  （08 の tda_features.parquet は残す。チェックポイントあり、途中再開可能）

実行:
  python -u src/08b_tda_buffered.py --check   … 数パッチで旧方式との違いと速度を確認
  python -u src/08b_tda_buffered.py           … 全メッシュ（約1.5〜2時間）
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
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds
import gudhi

BASE_DIR = Path(__file__).parent.parent
OUT_PATH = BASE_DIR / "output" / "tda_buffered_features.parquet"
CKPT     = BASE_DIR / "output" / "tda_buffered_checkpoint.parquet"
BUF_M    = 500.0
CKPT_EVERY = 500

# 08 の関数を再利用（ファイル名が数字始まりなので importlib で読む）
_spec = importlib.util.spec_from_file_location("tda08", Path(__file__).parent / "08_tda_features.py")
tda08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tda08)
IMG = tda08.IMG_SIZE
NODATA = tda08.NODATA_VAL


def read_buffered(src, geom, mesh_crs, dem_crs):
    """バッファ付きパッチと、パッチ内での中心メッシュの行・列範囲を返す"""
    big = geom.buffer(BUF_M, join_style=2)          # 角を保った正方形のバッファ
    l, b, r, t = transform_bounds(mesh_crs, dem_crs, *big.bounds)
    win = window_from_bounds(l, b, r, t, src.transform)
    patch = src.read(1, window=win, boundless=True, fill_value=NODATA).astype(np.float64)
    # 中心メッシュの範囲を、パッチ内の画素座標に変換
    cl, cb, cr, ct = transform_bounds(mesh_crs, dem_crs, *geom.bounds)
    inv = ~src.window_transform(win)
    c0, r0 = inv * (cl, ct)
    c1, r1 = inv * (cr, cb)
    center = (int(round(r0)), int(round(r1)), int(round(c0)), int(round(c1)))
    return patch, center


def diagrams_born_in_center(z: np.ndarray, valid: np.ndarray,
                            center) -> tuple[np.ndarray, np.ndarray]:
    """パッチ全体で PH を計算し、birth セルが中心にある有限ペアだけ返す

    death セルが NODATA（08 の「最大値+500m の壁」）のペアは捨てる。
    壁で囲われてできた偽の窪地で、persistence が数百 m になり Imager の範囲を壊すため
    （初版では H0 の persistence の範囲が 0〜513m、1画素 25.65m になった）。
    """
    nr = z.shape[0]
    r0, r1, c0, c1 = center
    cc = gudhi.CubicalComplex(top_dimensional_cells=z)
    cc.compute_persistence()
    regular, _ = cc.cofaces_of_persistence_pairs()
    zf = z.ravel(order="F")
    vf = valid.ravel(order="F")
    out = []
    for dim in (0, 1):
        if dim >= len(regular) or len(regular[dim]) == 0:
            out.append(np.empty((0, 2)))
            continue
        bi, di = regular[dim][:, 0], regular[dim][:, 1]
        rr, cc_ = bi % nr, bi // nr                 # Fortran 順の番号 → 行・列
        keep = (rr >= r0) & (rr < r1) & (cc_ >= c0) & (cc_ < c1) & vf[di]
        d = np.column_stack([zf[bi[keep]], zf[di[keep]]])
        d = d[d[:, 1] > d[:, 0]]
        out.append(d)
    return out[0], out[1]


# ------------------------------------------------------------------ #
# Imager（座標の範囲を代表的な標本から決め、20×20 にそのまま収める）
# ------------------------------------------------------------------ #

FIT_N = 1000         # 範囲を決めるランダム標本のメッシュ数
PERS_Q = 99.5        # persistence の範囲の上限（パーセンタイル）。超えるペアは最上段に入れる


def make_imager():
    """[0,1]×[0,1] に正規化した図を 20×20 画素にする Imager（1画素 = 0.05）"""
    from persim import PersistenceImager
    px = 1.0 / IMG
    return PersistenceImager(birth_range=(0.0, 1.0), pers_range=(0.0, 1.0), pixel_size=px,
                             kernel_params={"sigma": [[px ** 2, 0.0], [0.0, px ** 2]]})


def normalize(dg: np.ndarray, ax: dict) -> np.ndarray:
    """(birth, death) → 正規化した (birth, death)。persistence は上限で頭打ち"""
    if len(dg) == 0:
        return dg
    b = (dg[:, 0] - ax["b0"]) / (ax["b1"] - ax["b0"])
    p = np.minimum(dg[:, 1] - dg[:, 0], ax["p1"]) / ax["p1"]
    b = np.clip(b, 0.0, 1.0 - 1e-9)
    return np.column_stack([b, b + p])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    mesh = gpd.read_file(tda08.MESH_PATH)
    mesh_crs = mesh.crs.to_string()
    src = rasterio.open(tda08.DEM_PATH)
    dem_crs = src.crs.to_string()

    def diagrams(row):
        patch, center = read_buffered(src, row.geometry, mesh_crs, dem_crs)
        z = tda08.preprocess_patch(patch)
        if z is None:
            return None
        return diagrams_born_in_center(z, patch != NODATA, center)

    if args.check:
        rng = np.random.default_rng(0)
        print("旧方式（500m単独）とバッファ付きの H0 ペア数・最大 persistence の比較")
        ts = []
        for i in rng.choice(len(mesh), 20, replace=False):
            row = mesh.iloc[i]
            old = tda08.preprocess_patch(tda08.read_patch(src, row.geometry, mesh_crs, dem_crs))
            if old is None:
                continue
            h0o, h1o = tda08.compute_diagrams(old)
            t = time.time()
            h0b, h1b = diagrams(row)
            ts.append(time.time() - t)
            po = (h0o[:, 1] - h0o[:, 0]).max() if len(h0o) else 0
            pb = (h0b[:, 1] - h0b[:, 0]).max() if len(h0b) else 0
            print(f"  mesh {row.mesh_id}: H0 {len(h0o)}→{len(h0b)}  H1 {len(h1o)}→{len(h1b)}  "
                  f"最大pers {po:.2f}→{pb:.2f} m")
        print(f"1パッチ {np.mean(ts):.2f} 秒 → 全 {len(mesh)} で約 {np.mean(ts) * len(mesh) / 60:.0f} 分")
        return

    done = set()
    if CKPT.exists():
        done = set(pd.read_parquet(CKPT)["mesh_id"])
        print(f"チェックポイント: {len(done)} 件済み")
    rest = mesh[~mesh["mesh_id"].isin(done)].reset_index(drop=True)

    # 座標の範囲：ランダムな FIT_N メッシュの図から決めて保存（再開時は保存値を使う）
    ax_path = OUT_PATH.with_name("tda_buffered_imager.json")
    if ax_path.exists():
        axes = json.loads(ax_path.read_text(encoding="utf-8"))
    else:
        rng = np.random.default_rng(0)
        pool = {0: [], 1: []}
        for i in rng.choice(len(mesh), FIT_N, replace=False):
            dg = diagrams(mesh.iloc[i])
            if dg is None:
                continue
            for k in (0, 1):
                if len(dg[k]):
                    pool[k].append(dg[k])
        axes = {}
        for k in (0, 1):
            a = np.vstack(pool[k])
            axes[f"H{k}"] = {"b0": float(a[:, 0].min()), "b1": float(a[:, 0].max()),
                             "p1": float(np.percentile(a[:, 1] - a[:, 0], PERS_Q)),
                             "n_pairs": int(len(a))}
        ax_path.write_text(json.dumps(axes, ensure_ascii=False, indent=1), encoding="utf-8")
    for h, a in axes.items():
        print(f"{h}: birth {a['b0']:.1f}〜{a['b1']:.1f} m、persistence 0〜{a['p1']:.2f} m"
              f"（{PERS_Q}%点、1画素 {(a['b1'] - a['b0']) / IMG:.1f} m × {a['p1'] / IMG:.3f} m）",
              flush=True)
    imager = make_imager()

    c0 = [f"h0b_{i:03d}" for i in range(IMG * IMG)]
    c1 = [f"h1b_{i:03d}" for i in range(IMG * IMG)]
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        df = pd.DataFrame(buf)
        if CKPT.exists():
            df = pd.concat([pd.read_parquet(CKPT), df], ignore_index=True)
        df.to_parquet(CKPT, index=False)
        buf = []

    for k, row in rest.iterrows():
        dg = diagrams(row)
        if dg is None:
            v0 = np.zeros(IMG * IMG, np.float32)
            v1 = np.zeros(IMG * IMG, np.float32)
        else:
            v0 = tda08.diagram_to_vec(normalize(dg[0], axes["H0"]), imager, IMG)
            v1 = tda08.diagram_to_vec(normalize(dg[1], axes["H1"]), imager, IMG)
        buf.append({"mesh_id": row["mesh_id"], **dict(zip(c0, v0)), **dict(zip(c1, v1))})
        if (k + 1) % CKPT_EVERY == 0:
            flush()
            el = time.time() - t0
            print(f"  {len(done) + k + 1}/{len(mesh)}  経過 {el / 60:.1f} 分  "
                  f"残り約 {el / (k + 1) * (len(rest) - k - 1) / 60:.0f} 分", flush=True)
    flush()
    res = pd.read_parquet(CKPT).drop_duplicates("mesh_id").sort_values("mesh_id")
    res.to_parquet(OUT_PATH, index=False)
    print(f"保存: {OUT_PATH}（{len(res)} 件、{(time.time() - t0) / 60:.0f} 分）")
    src.close()


if __name__ == "__main__":
    main()
