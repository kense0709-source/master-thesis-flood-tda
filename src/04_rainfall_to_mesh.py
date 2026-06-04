"""
解析雨量 GRIB2 デコーダ + 500mメッシュへの対応づけ

処理フロー:
  1. GRIB2 バイナリを解析（セクション読み込み）
  2. ランレングス展開 → レベル値配列 (3360×2560)
  3. レベル値 → 降水量[mm] 変換（Section5の変換テーブル使用）
  4. 研究エリア（筑後川流域）を切り出し
  5. 各タイムステップの降水量を 500mメッシュに対応づけ
  6. イベント単位の特徴量（最大1h・3h・6h・24h雨量）を計算

出力:
  output/rainfall_features.parquet
    列: mesh_id, event, rain_max1h, rain_max3h, rain_max6h, rain_sum24h
"""

import struct
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

BASE_DIR = Path(__file__).parent.parent
RAINFALL_DIR = BASE_DIR / "解析雨量データ"
OUT_PATH = BASE_DIR / "output" / "rainfall_features.parquet"
MESH_PATH = BASE_DIR / "output" / "mesh_dem_stats.parquet"

# 豪雨イベント期間（UTC、ファイル名に記載の時刻）
EVENTS = {
    "2020": ("20200704000000", "20200709230000"),
    "2023": ("20230629000000", "20230705230000"),
}

# 筑後川流域バウンディングボックス
BBOX = {"lon_min": 130.25, "lon_max": 131.00,
        "lat_min": 32.92,  "lat_max": 33.50}

# GRIB2グリッド固定パラメータ（format.txt より）
NI        = 2560          # 東西方向格子数
NJ        = 3360          # 南北方向格子数
LAT_FIRST = 47.995833     # 最北端緯度 [度]
LON_FIRST = 118.006250    # 最西端経度 [度]
DJ        = 8333 / 1e6    # 緯度方向間隔 [度] ≈ 0.00833°
DI        = 12500 / 1e6   # 経度方向間隔 [度] = 0.0125°


# ------------------------------------------------------------------ #
# 1. GRIB2 パーサー
# ------------------------------------------------------------------ #

def read_sections(data: bytes) -> dict:
    """GRIB2 バイナリをセクション番号 → バイト列の辞書に分解"""
    assert data[:4] == b"GRIB" and data[7] == 2, "GRIB2 形式ではありません"
    pos = 16  # Section 0 は固定 16 バイト
    sections = {}
    while pos < len(data):
        if data[pos:pos+4] == b"7777":
            break
        sec_len = struct.unpack_from(">I", data, pos)[0]
        sec_num = data[pos + 4]
        sections[sec_num] = data[pos: pos + sec_len]
        pos += sec_len
    return sections


def parse_section5(sec5: bytes):
    """Section5 からビット数・閾値・変換テーブル R を取得

    V: ランレングス符号化の閾値（0〜V がレベル値、V+1〜255 がランレングス）
    M: データの取り得るレベル最大値（R テーブルのサイズ）
    ※ decode_rlen_nbit に渡す maxv は M ではなく V （C コードと同じ）
    """
    nbit = sec5[11]
    V    = struct.unpack_from(">H", sec5, 12)[0]  # ランレングス閾値 (=55)
    M    = struct.unpack_from(">H", sec5, 14)[0]  # レベル最大値 (=98)
    E    = sec5[16]                                # 尺度因子 (=1)

    rain_table = np.zeros(M + 1, dtype=np.float32)
    for m in range(1, M + 1):
        r_val = struct.unpack_from(">H", sec5, 17 + 2 * (m - 1))[0]
        rain_table[m] = r_val * (10.0 ** (-E))

    return nbit, V, M, rain_table


def decode_runlength(packed: np.ndarray, nout: int, maxv: int, nbit: int) -> np.ndarray:
    """気象庁式ランレングス展開（rlencmp.c の decode_rlen_nbit を Python 移植）

    packed : 圧縮済み 8bit 整数配列
    nout   : 展開後の格子点数 (3360×2560 = 8,601,600)
    maxv   : レベル最大値 (=98)
    nbit   : ビット数 (=8)
    """
    l = (1 << nbit) - 1 - maxv   # ランレングス用値域幅 = 157

    result = np.empty(nout, dtype=np.int32)
    k = 0       # 出力インデックス
    p = -1      # 直前のレベル値
    m = 1       # 繰り返し回数
    n = 0       # 冪乗インデックス

    for raw_v in packed:
        v = int(raw_v)
        if v <= maxv:
            # レベル値 → 直前の値 p を m 回出力してから更新
            if p >= 0:
                end = min(k + m, nout)
                result[k:end] = p
                k = end
                if k >= nout:
                    break
            p, m, n = v, 1, 0
        else:
            # ランレングス延長
            m += (l ** n) * (v - maxv - 1)
            n += 1

    # 最後のレベル値を出力
    if p >= 0 and k < nout:
        result[k: min(k + m, nout)] = p

    return result


def decode_grib2(filepath: Path) -> np.ndarray:
    """GRIB2 ファイル 1 本を読んで降水量[mm]の 2D 配列 (NJ×NI) を返す"""
    data = filepath.read_bytes()
    sections = read_sections(data)

    nbit, V, M, rain_table = parse_section5(sections[5])

    # Section 7: 先頭 5 バイト（長さ+節番号）を除いた部分が圧縮データ
    compressed = np.frombuffer(sections[7][5:], dtype=np.uint8)
    levels = decode_runlength(compressed, NI * NJ, V, nbit)  # maxv=V (not M)

    rain = rain_table[levels].reshape(NJ, NI)  # (3360, 2560)
    return rain


# ------------------------------------------------------------------ #
# 2. 研究エリアの切り出し & 格子インデックス計算
# ------------------------------------------------------------------ #

def get_area_slice():
    """筑後川流域に対応する行・列スライスと緯経度配列を返す"""
    j_min = max(0, int((LAT_FIRST - BBOX["lat_max"]) / DJ))
    j_max = min(NJ, int((LAT_FIRST - BBOX["lat_min"]) / DJ) + 2)
    i_min = max(0, int((BBOX["lon_min"] - LON_FIRST) / DI))
    i_max = min(NI, int((BBOX["lon_max"] - LON_FIRST) / DI) + 2)

    lats = LAT_FIRST - np.arange(j_min, j_max) * DJ
    lons = LON_FIRST + np.arange(i_min, i_max) * DI
    return slice(j_min, j_max), slice(i_min, i_max), lats, lons


# ------------------------------------------------------------------ #
# 3. ファイル一覧取得 & タイムスタンプ解析
# ------------------------------------------------------------------ #

TS_PATTERN = re.compile(r"(\d{14})")

def file_timestamp(path: Path) -> datetime:
    m = TS_PATTERN.search(path.name)
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")

def get_event_files(event_key: str) -> list[Path]:
    """イベント期間内の .bin ファイルをタイムスタンプ昇順で返す"""
    start_str, end_str = EVENTS[event_key]
    start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    end   = datetime.strptime(end_str,   "%Y%m%d%H%M%S")

    year = start.year
    rain_dir = next(RAINFALL_DIR.rglob(f"DATA/{year}"), None)
    if rain_dir is None:
        raise FileNotFoundError(f"解析雨量 {year} 年データが見つかりません")

    files = sorted(
        [f for f in rain_dir.rglob("*.bin") if start <= file_timestamp(f) <= end],
        key=file_timestamp,
    )
    return files


# ------------------------------------------------------------------ #
# 4. イベント単位の特徴量計算
# ------------------------------------------------------------------ #

def compute_event_rainfall(files: list[Path], row_sl, col_sl) -> np.ndarray:
    """
    全ファイルを読んで降水量時系列を返す
    shape: (n_timesteps, n_rows_area, n_cols_area)
    各ファイルは前 1 時間累積雨量 [mm]
    """
    n = len(files)
    sample = decode_grib2(files[0])[row_sl, col_sl]
    cube = np.zeros((n, *sample.shape), dtype=np.float32)
    cube[0] = sample

    for i, f in enumerate(files[1:], 1):
        if i % 20 == 0:
            print(f"  {i}/{n} ファイル処理中...")
        cube[i] = decode_grib2(f)[row_sl, col_sl]

    return cube   # shape: (n, nj_area, ni_area)


def rolling_max(cube: np.ndarray, window: int) -> np.ndarray:
    """時間軸(axis=0)に沿ったローリング最大値"""
    n = cube.shape[0]
    if n < window:
        return cube.sum(axis=0)
    result = np.zeros(cube.shape[1:], dtype=np.float32)
    for start in range(n - window + 1):
        s = cube[start: start + window].sum(axis=0)
        result = np.maximum(result, s)
    return result


def extract_features(cube: np.ndarray) -> dict:
    """1h/3h/6h/24h の最大累積雨量を計算"""
    # cube の 1 タイムステップ = 1 時間累積（30分毎ファイルのため 2step = 1h）
    # 毎正時のみ（偶数インデックス）を使って重複を避ける
    hourly = cube[::2]  # 30分→1時間サンプリング
    return {
        "rain_max1h":  hourly.max(axis=0),
        "rain_max3h":  rolling_max(hourly, 3),
        "rain_max6h":  rolling_max(hourly, 6),
        "rain_sum24h": rolling_max(hourly, 24),
    }


# ------------------------------------------------------------------ #
# 5. メッシュへの対応づけ（最近傍）
# ------------------------------------------------------------------ #

def assign_to_mesh(features: dict, lats: np.ndarray, lons: np.ndarray,
                   mesh_df: pd.DataFrame) -> pd.DataFrame:
    """各 500m メッシュセルに最近傍の 1km 格子の雨量特徴量を割り当てる"""
    # 格子のインデックスを meshgrid で展開
    lon_grid, lat_grid = np.meshgrid(lons, lats)   # shape: (nj_area, ni_area)

    records = []
    for _, row in mesh_df.iterrows():
        # 最近傍格子点を探索
        dlat = np.abs(lat_grid - row["center_lat"])
        dlon = np.abs(lon_grid - row["center_lon"])
        dist = dlat + dlon  # L1距離で十分
        j_near, i_near = np.unravel_index(dist.argmin(), dist.shape)

        rec = {"mesh_id": row.name}
        for feat_name, feat_arr in features.items():
            rec[feat_name] = float(feat_arr[j_near, i_near])
        records.append(rec)

    return pd.DataFrame(records).set_index("mesh_id")


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    row_sl, col_sl, lats, lons = get_area_slice()
    print(f"対象グリッド: {lats.shape[0]} 行 × {lons.shape[0]} 列")
    print(f"  緯度: {lats[-1]:.3f} 〜 {lats[0]:.3f}°N")
    print(f"  経度: {lons[0]:.3f} 〜 {lons[-1]:.3f}°E\n")

    mesh_df = pd.read_parquet(MESH_PATH)

    all_dfs = []
    for event_key in ["2020", "2023"]:
        print(f"=== {event_key}年イベント ===")
        files = get_event_files(event_key)
        print(f"  対象ファイル数: {len(files)} 本")

        t0 = time.time()
        cube = compute_event_rainfall(files, row_sl, col_sl)
        features = extract_features(cube)
        print(f"  デコード完了: {time.time()-t0:.1f}s")

        df = assign_to_mesh(features, lats, lons, mesh_df)
        df["event"] = event_key
        all_dfs.append(df)
        print(f"  メッシュ対応づけ完了: {len(df)} セル\n")

    result = pd.concat(all_dfs).reset_index()
    result.to_parquet(OUT_PATH)
    print(f"保存: {OUT_PATH}")
    print("\n--- サンプル ---")
    print(result.head())
    print(f"\nイベント別サンプル数: {result.groupby('event').size().to_dict()}")


if __name__ == "__main__":
    main()
