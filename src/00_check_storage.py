"""
Step 0: 貯留容量の識別力を事前確認する

目的:
  「窪地の貯留容量」という量に、そもそも浸水の識別力があるかを
  本実装（窪地単位のTDA特徴抽出）に入る前に最小コストで確認する。

処理フロー:
  1. DEM（EPSG:6668）→ UTM53N 再投影
  2. pysheds: fill_pits → fill_depressions
  3. 窪地深 depth = filled - DEM   ← これが従来手法（TDA不使用）の窪地
  4. 500mメッシュへゾーン統計
       sink_depth_max     … メッシュ内の最大窪地深 [m]
       sink_volume_sum    … 貯留容量 [m3]
       sink_storage_depth … 貯留深 [mm] = volume / メッシュ面積
       sink_area_frac     … 窪地が占める面積率
  5. 浸水ラベル・雨量と結合
       sink_overflow_ratio = rain_sum24h / sink_storage_depth （無次元）
  6. 浸水あり／なしで分布を比較、各特徴量の単独 ROC-AUC を算出

判断基準:
  単独AUC 0.5付近 → 貯留容量に識別力なし。TDAの階層的体積に進んでも見込み薄
  単独AUC 0.6以上 → 投資する価値あり。窪地単位特徴の実装へ進む

備考:
  ここで計算する sink_* は Model A に必要な特徴量そのものなので、
  結果がどうであれこの計算は無駄にならない。

出力:
  output/sink_depth.tif        … 窪地深ラスタ（Step 1 で再利用）
  output/sink_features.parquet … メッシュ別 sink 特徴量
  output/check_storage.png     … 分布比較・ROC曲線
  output/check_storage.txt     … AUC サマリー
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from rasterstats import zonal_stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR   = Path(__file__).parent.parent
DEM_PATH   = BASE_DIR / "output" / "dem_merged.tif"
# 再投影済みDEMのキャッシュ。13,029x12,130px（1.58億セル）の再投影は数時間かかるため、
# 一時ファイルにせず output/ に残して以降のスクリプトで使い回す
DEM_UTM_PATH = BASE_DIR / "output" / "dem_utm.tif"
MESH_PATH  = BASE_DIR / "output" / "mesh_500m.gpkg"
LABEL_PATH = BASE_DIR / "output" / "mesh_labels.parquet"
RAIN_PATH  = BASE_DIR / "output" / "rainfall_features.parquet"

OUT_DEPTH  = BASE_DIR / "output" / "sink_depth.tif"
OUT_PARQ   = BASE_DIR / "output" / "sink_features.parquet"
OUT_PNG    = BASE_DIR / "output" / "check_storage.png"
OUT_TXT    = BASE_DIR / "output" / "check_storage.txt"

DST_CRS    = "EPSG:32653"    # UTM Zone 53N
NODATA     = -9999.0
DEPTH_EPS  = 0.01            # これ未満の窪地深はノイズとして無視 [m]

# 配色（dataviz skill の categorical slot 1/2、light mode で検証済み）
C_FLOOD    = "#eb6834"       # 浸水あり
C_DRY      = "#2a78d6"       # 浸水なし
INK        = "#0b0b0b"
INK_SUB    = "#52514e"
GRID       = "#d8d7d2"


# ------------------------------------------------------------------ #
# 1. DEM を UTM53N へ再投影
# ------------------------------------------------------------------ #

def reproject_dem(src_path: Path, dst_path: Path) -> None:
    if dst_path.exists():
        with rasterio.open(dst_path) as r:
            print(f"[1/6] 再投影済みDEMを再利用: {dst_path.name} "
                  f"({r.width}x{r.height} px, res={r.res[0]:.2f} m)", flush=True)
        return

    print("[1/6] DEM を UTM53N に再投影中...（1.58億セル・数時間かかる）", flush=True)
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, DST_CRS, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": DST_CRS, "transform": transform,
                       "width": width, "height": height,
                       "nodata": NODATA, "dtype": "float32"})
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=DST_CRS,
                resampling=Resampling.bilinear,
            )
    with rasterio.open(dst_path) as r:
        print(f"      {r.width}x{r.height} px, res={r.res[0]:.2f} m")


# ------------------------------------------------------------------ #
# 2. 窪地深の計算（従来手法：fill 差分）
# ------------------------------------------------------------------ #

def compute_sink_depth(dem_utm_path: Path) -> float:
    """fill_depressions との差分から窪地深ラスタを作り保存。cellsize を返す

    NODATA境界アーティファクトへの対処:
      DEM をそのまま fill すると、流出先が NODATA で塞がれた領域が
      内部の分水界の高さまで埋められ、窪地深が 100m〜1,200m に達する偽の値が出る。
      （検証: NODATA から 2.7km 以上離れた内陸部には深さ 20m 超の窪地が皆無）
      そこで NODATA セルに十分低い標高を与え、「常に排水される外部」として扱う。
      これにより NODATA に接する領域は外へ流出でき、偽の窪地が発生しない。
    """
    print("[2/6] pysheds: fill_pits -> fill_depressions ...", flush=True)
    grid = Grid.from_raster(str(dem_utm_path))
    dem  = grid.read_raster(str(dem_utm_path))

    dem_arr = np.asarray(dem, dtype=np.float32)
    valid   = dem_arr != NODATA

    # NODATA を外部排水口として扱う（上記の対処）
    sink_floor = float(dem_arr[valid].min()) - 1000.0
    work = np.where(valid, dem_arr, sink_floor).astype(np.float32)

    vf = dem.viewfinder
    work_vf = ViewFinder(affine=vf.affine, shape=vf.shape, crs=vf.crs,
                         nodata=np.float32(-1e9),
                         mask=np.ones(vf.shape, dtype=bool))
    work_raster = Raster(work, viewfinder=work_vf)

    pit_filled = grid.fill_pits(work_raster)
    flooded    = grid.fill_depressions(pit_filled)
    filled_arr = np.asarray(flooded, dtype=np.float32)

    with rasterio.open(dem_utm_path) as src:
        profile  = src.profile.copy()
        cellsize = src.res[0]

    depth = np.where(valid, filled_arr - work, NODATA).astype(np.float32)
    # 数値誤差による微小な負値・ノイズを除去
    depth = np.where(valid & (depth < DEPTH_EPS), 0.0, depth).astype(np.float32)

    n_sink = int((valid & (depth > 0)).sum())
    n_all  = int(valid.sum())
    pos    = depth[valid & (depth > 0)]
    print(f"      窪地セル: {n_sink:,} / {n_all:,} ({n_sink / n_all * 100:.1f}%)",
          flush=True)
    print(f"      窪地深 中央 {np.median(pos):.2f} m / "
          f"95%tile {np.percentile(pos, 95):.2f} m / "
          f"最大 {pos.max():.2f} m", flush=True)
    if pos.max() > 50.0:
        print("      [警告] 最大窪地深が 50m を超えています。"
              "境界アーティファクトが残っていないか確認してください", flush=True)

    profile.update({"dtype": "float32", "count": 1, "nodata": NODATA})
    with rasterio.open(OUT_DEPTH, "w", **profile) as dst:
        dst.write(depth, 1)
    print(f"      保存: {OUT_DEPTH.name}")

    return cellsize


# ------------------------------------------------------------------ #
# 3. メッシュへのゾーン統計
# ------------------------------------------------------------------ #

def zonal_sink(mesh: gpd.GeoDataFrame, cellsize: float) -> pd.DataFrame:
    print("[3/6] ゾーン統計（メッシュ別 sink 特徴量）...")
    stats = zonal_stats(mesh, str(OUT_DEPTH),
                        stats=["mean", "max", "sum", "count"],
                        nodata=NODATA, all_touched=False)
    df = pd.DataFrame(stats, index=mesh.index)

    cell_area = cellsize ** 2
    mesh_area = mesh.geometry.area          # UTM なので m2

    out = pd.DataFrame(index=mesh.index)
    out["sink_depth_mean"]   = df["mean"]
    out["sink_depth_max"]    = df["max"]
    out["sink_volume_sum"]   = df["sum"] * cell_area                        # [m3]
    out["sink_storage_depth"] = out["sink_volume_sum"] / mesh_area * 1000.0  # [mm]

    # 窪地が占める面積率（depth>0 のセル比率）
    out["sink_area_frac"] = _sink_area_fraction(mesh)

    out.index.name = "mesh_id"
    print(out.describe().round(3).to_string())
    return out.reset_index()


def _sink_area_fraction(mesh: gpd.GeoDataFrame) -> pd.Series:
    """depth>0 のセル比率をメッシュごとに算出"""
    from rasterstats import zonal_stats as zs

    def frac_positive(masked):
        if masked.count() == 0:
            return np.nan
        return float((masked > 0).sum()) / float(masked.count())

    res = zs(mesh, str(OUT_DEPTH), stats=[], nodata=NODATA,
             add_stats={"frac": frac_positive})
    return pd.Series([r["frac"] for r in res], index=mesh.index)


# ------------------------------------------------------------------ #
# 4. ラベル・雨量と結合
# ------------------------------------------------------------------ #

def build_table(sink: pd.DataFrame) -> pd.DataFrame:
    print("[4/6] 浸水ラベル・雨量と結合...")
    labels = pd.read_parquet(LABEL_PATH)     # mesh_id, event, label
    rain   = pd.read_parquet(RAIN_PATH)      # mesh_id, event, rain_*

    df = labels.merge(rain, on=["mesh_id", "event"], how="left")
    df = df.merge(sink, on="mesh_id", how="left")

    # 貯留深0のメッシュで発散しないよう微小値を加える
    df["sink_overflow_ratio"] = df["rain_sum24h"] / (df["sink_storage_depth"] + 1.0)
    df["has_depression"] = (df["sink_storage_depth"] > 0).astype(int)

    print(f"      {len(df):,} 行 / イベント: {sorted(df['event'].unique())}")
    print(f"      陽性率: {df.groupby('event')['label'].mean().round(4).to_dict()}")
    return df


# ------------------------------------------------------------------ #
# 5. 単独 AUC
# ------------------------------------------------------------------ #

FEATURES = [
    ("sink_storage_depth",  "貯留深 [mm]"),
    ("sink_volume_sum",     "貯留容量 [m3]"),
    ("sink_depth_max",      "最大窪地深 [m]"),
    ("sink_area_frac",      "窪地面積率"),
    ("sink_overflow_ratio", "overflow_ratio"),
]


def single_auc(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] 単独 ROC-AUC 算出...")
    rows = []
    for event, sub in df.groupby("event"):
        for col, jp in FEATURES:
            m = sub[col].notna() & sub["label"].notna()
            if m.sum() == 0 or sub.loc[m, "label"].nunique() < 2:
                continue
            auc = roc_auc_score(sub.loc[m, "label"], sub.loc[m, col])
            rows.append({
                "event": event, "feature": col, "label_jp": jp,
                "auc": auc,
                # 0.5 からの距離。向きによらない識別力
                "auc_abs": max(auc, 1 - auc),
                "direction": "高いほど浸水" if auc > 0.5 else "低いほど浸水",
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# 6. 図
# ------------------------------------------------------------------ #

def make_figure(df: pd.DataFrame, auc_df: pd.DataFrame) -> None:
    print("[6/6] 図を作成中...")
    events = sorted(df["event"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor("#fcfcfb")

    # --- (a) 貯留深の分布（浸水あり／なし × イベント） -------------- #
    ax = axes[0]
    positions, data, colors = [], [], []
    for i, ev in enumerate(events):
        for j, lab in enumerate([0, 1]):
            v = df[(df["event"] == ev) & (df["label"] == lab)]["sink_storage_depth"]
            v = v[v.notna()].clip(lower=0.01)      # log軸のため下限
            data.append(v.values)
            positions.append(i * 2.6 + j * 0.9)
            colors.append(C_DRY if lab == 0 else C_FLOOD)

    bp = ax.boxplot(data, positions=positions, widths=0.62, patch_artist=True,
                    showfliers=False, medianprops=dict(color="#fcfcfb", linewidth=2))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("#fcfcfb")
        patch.set_linewidth(2)
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color(INK_SUB)
            line.set_linewidth(1.2)

    ax.set_yscale("log")
    ax.set_xticks([i * 2.6 + 0.45 for i in range(len(events))])
    ax.set_xticklabels(events, color=INK_SUB)
    ax.set_ylabel("貯留深 [mm]（対数軸）", color=INK_SUB)
    ax.set_title("(a) 窪地の貯留深の分布", color=INK, loc="left", fontsize=11)
    _style_axes(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=C_DRY, ec="none"),
               plt.Rectangle((0, 0), 1, 1, fc=C_FLOOD, ec="none")]
    ax.legend(handles, ["浸水なし", "浸水あり"], frameon=False,
              labelcolor=INK_SUB, loc="upper left", fontsize=9)

    # --- (b) ROC 曲線（貯留深・最大窪地深） ------------------------- #
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", zorder=1)
    palette = [C_DRY, C_FLOOD, "#1baf7a", "#eda100"]
    ci = 0
    for ev in events:
        sub = df[df["event"] == ev]
        for col, jp in [("sink_storage_depth", "貯留深"),
                        ("sink_depth_max", "最大窪地深")]:
            m = sub[col].notna()
            if m.sum() == 0 or sub.loc[m, "label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub.loc[m, "label"], sub.loc[m, col])
            auc = roc_auc_score(sub.loc[m, "label"], sub.loc[m, col])
            ax.plot(fpr, tpr, color=palette[ci % len(palette)], linewidth=2,
                    label=f"{ev} {jp}  AUC={auc:.3f}", zorder=2)
            ci += 1
    ax.set_xlabel("偽陽性率", color=INK_SUB)
    ax.set_ylabel("真陽性率", color=INK_SUB)
    ax.set_title("(b) 単独特徴量の ROC 曲線", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK_SUB, loc="lower right", fontsize=8)
    _style_axes(ax)

    # --- (c) AUC 一覧（0.5 からの乖離） ----------------------------- #
    ax = axes[2]
    piv = auc_df.pivot_table(index="label_jp", columns="event", values="auc")
    piv = piv.reindex([jp for _, jp in FEATURES])
    y = np.arange(len(piv))
    h = 0.36
    for k, ev in enumerate(piv.columns):
        c = C_DRY if k == 0 else C_FLOOD
        ax.barh(y + (k - 0.5) * h, piv[ev] - 0.5, left=0.5, height=h * 0.92,
                color=c, edgecolor="#fcfcfb", linewidth=2, label=ev, zorder=2)
        for yi, v in zip(y + (k - 0.5) * h, piv[ev]):
            if np.isnan(v):
                continue
            ax.text(v + (0.004 if v >= 0.5 else -0.004), yi, f"{v:.3f}",
                    va="center", ha="left" if v >= 0.5 else "right",
                    fontsize=8, color=INK_SUB)
    ax.axvline(0.5, color=INK_SUB, linewidth=1.2, zorder=3)
    ax.axvline(0.6, color=GRID, linewidth=1.5, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, color=INK_SUB)
    ax.set_xlabel("単独 ROC-AUC（破線 0.6 = 投資判断の目安）", color=INK_SUB)
    ax.set_title("(c) 特徴量別の単独 AUC", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK_SUB, loc="lower right", fontsize=9)
    _style_axes(ax)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, facecolor=fig.get_facecolor())
    print(f"      保存: {OUT_PNG.name}")


def _style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SUB, labelsize=9)
    ax.grid(axis="both", color=GRID, linewidth=0.7, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main() -> None:
    # 日本語フォント（環境にあれば）
    for font in ("Meiryo", "Yu Gothic", "MS Gothic"):
        if font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

    reproject_dem(DEM_PATH, DEM_UTM_PATH)
    cellsize = compute_sink_depth(DEM_UTM_PATH)

    mesh = gpd.read_file(MESH_PATH).set_index("mesh_id")
    sink = zonal_sink(mesh, cellsize)
    sink.to_parquet(OUT_PARQ, index=False)
    print(f"      保存: {OUT_PARQ.name}", flush=True)

    df     = build_table(sink)
    auc_df = single_auc(df)

    # サマリー出力
    lines = ["=== Step 0: 貯留容量の識別力 ===", ""]
    lines.append(auc_df[["event", "label_jp", "auc", "auc_abs", "direction"]]
                 .to_string(index=False))
    best = auc_df["auc_abs"].max()
    lines += ["", f"最大 |AUC-0.5| ベース: {best:.3f}", ""]
    if best >= 0.60:
        lines.append("判断: 0.60 以上 → 貯留容量に識別力あり。窪地単位特徴の実装へ進む")
    elif best >= 0.55:
        lines.append("判断: 0.55〜0.60 → 弱い識別力。他特徴との組合せで効く可能性。要検討")
    else:
        lines.append("判断: 0.55 未満 → 単独では識別力ほぼなし。"
                     "窪地体積の深追いは見送り、PI・H1 中心の方針を再検討")
    text = "\n".join(lines)
    OUT_TXT.write_text(text, encoding="utf-8")
    print("\n" + text, flush=True)

    make_figure(df, auc_df)


if __name__ == "__main__":
    main()
