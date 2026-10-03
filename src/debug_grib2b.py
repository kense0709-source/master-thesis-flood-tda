"""研究エリアの雨量値を直接確認"""
import struct, numpy as np, glob

# 実際のファイルパスを使う
files = sorted(glob.glob("解析雨量データ/**/*.bin", recursive=True))
hit = next(f for f in files if "20200705120000" in f)  # 7月5日昼（豪雨ピーク）
print("使用ファイル:", hit.split("\\")[-1])

# デコード
from pathlib import Path
import importlib, types

# inline で decode する
data = Path(hit).read_bytes()

pos = 16
sections = {}
while pos < len(data):
    if data[pos:pos+4] == b"7777": break
    sec_len = struct.unpack_from(">I", data, pos)[0]
    sec_num = data[pos+4]
    sections[sec_num] = data[pos:pos+sec_len]
    pos += sec_len

sec5 = sections[5]
nbit = sec5[11]
M    = struct.unpack_from(">H", sec5, 14)[0]
E    = sec5[16]
rain_table = np.zeros(M+1, dtype=np.float32)
for m in range(1, M+1):
    r = struct.unpack_from(">H", sec5, 17+2*(m-1))[0]
    rain_table[m] = r * (10.0 ** (-E))

sec7 = sections[7]
compressed = np.frombuffer(sec7[5:], dtype=np.uint8)

NI, NJ = 2560, 3360
LAT_FIRST = 47.995833
LON_FIRST = 118.006250
DJ = 8333/1e6
DI = 12500/1e6

j_min = int((LAT_FIRST - 33.50) / DJ)
j_max = int((LAT_FIRST - 32.92) / DJ) + 2
i_min = int((130.25 - LON_FIRST) / DI)
i_max = int((131.00 - LON_FIRST) / DI) + 2
print(f"エリア: rows {j_min}:{j_max}, cols {i_min}:{i_max}")
print(f"必要な展開点数: {j_max * NI:,}")

# ランレングス展開（必要点数まで）
l = (1 << nbit) - 1 - M
needed = j_max * NI + 100

result = []
p, mc, n = -1, 1, 0
for raw_v in compressed:
    v = int(raw_v)
    if v <= M:
        if p >= 0:
            result.extend([p] * mc)
        p, mc, n = v, 1, 0
    else:
        mc += (l**n) * (v-M-1)
        n += 1
    if len(result) >= needed:
        break
if p >= 0 and len(result) < needed:
    result.extend([p] * mc)

arr = np.array(result, dtype=np.int32)
print(f"展開済み点数: {len(arr):,}")

if len(arr) >= j_max * NI:
    # 2D に変換して研究エリアを切り出し
    grid2d = arr[:NJ*NI].reshape(-1, NI) if len(arr) >= NJ*NI else arr.reshape(-1, NI)
    area = grid2d[j_min:j_max, i_min:i_max]
    rain_area = rain_table[area]
    print(f"\n研究エリアのレベル値統計:")
    print(f"  非ゼロレベル: {(area>0).sum()} / {area.size}")
    print(f"  最大レベル: {area.max()}")
    print(f"\n研究エリアの雨量統計 [mm]:")
    print(f"  最大: {rain_area.max():.1f} mm")
    print(f"  平均: {rain_area.mean():.2f} mm")
    print(f"  非ゼロセル: {(rain_area>0).sum()}")
else:
    print("展開が研究エリアに届かなかった")
    # 全展開済み点での非ゼロ確認
    print(f"非ゼロ点数: {(arr>0).sum()} / {len(arr)}")
    print(f"最大レベル: {arr.max()}")
