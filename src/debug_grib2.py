import struct, numpy as np, glob

files = sorted(glob.glob("解析雨量データ/**/*.bin", recursive=True))
hits = [f for f in files if "20200705000000" in f]
f = hits[0]
print("file:", f.split("\\")[-1])

data = open(f, "rb").read()

pos = 16
sections = {}
while pos < len(data):
    if data[pos:pos+4] == b"7777":
        break
    sec_len = struct.unpack_from(">I", data, pos)[0]
    sec_num = data[pos+4]
    sections[sec_num] = data[pos:pos+sec_len]
    pos += sec_len

print("sections:", list(sections.keys()))
sec5 = sections[5]
print("sec5 length:", len(sec5))
print("sec5 hex[0:25]:", sec5[:25].hex())

nbit = sec5[11]
M    = struct.unpack_from(">H", sec5, 14)[0]
E    = sec5[16]
print(f"nbit={nbit}, M={M}, E={E}")

# 変換テーブル
rain_table = np.zeros(M+1, dtype=np.float32)
for m in range(1, M+1):
    r = struct.unpack_from(">H", sec5, 17+2*(m-1))[0]
    rain_table[m] = r * (10.0 ** (-E))
print("rain_table[1:11]:", rain_table[1:11])
print("rain_table[90:99]:", rain_table[90:99])

# 圧縮データ
sec7 = sections[7]
compressed = np.frombuffer(sec7[5:], dtype=np.uint8)
print(f"\ncompressed size: {len(compressed)} bytes")
print("first 50 values:", compressed[:50].tolist())
print("value range:", compressed.min(), "to", compressed.max())
print("values > M:", (compressed > M).sum(), "/", len(compressed))

# ランレングス展開（最初の1000点だけ）
l = (1 << nbit) - 1 - M
print(f"\nl={l}")
result = []
p, m_count, n = -1, 1, 0
for raw_v in compressed:
    v = int(raw_v)
    if v <= M:
        if p >= 0:
            result.extend([p] * min(m_count, 100))
        p, m_count, n = v, 1, 0
    else:
        m_count += (l**n) * (v - M - 1)
        n += 1
    if len(result) > 1000:
        break

arr = np.array(result)
print("decoded levels (first 100):", arr[:100])
print("non-zero levels:", (arr > 0).sum())

# 筑後川エリアの格子をフル展開して確認
LAT_FIRST = 47.995833
LON_FIRST = 118.006250
DJ = 8333 / 1e6
DI = 12500 / 1e6
NI, NJ = 2560, 3360

j_min = int((LAT_FIRST - 33.50) / DJ)
j_max = int((LAT_FIRST - 32.92) / DJ) + 2
i_min = int((130.25 - LON_FIRST) / DI)
i_max = int((131.00 - LON_FIRST) / DI) + 2
print(f"\nArea slice: rows {j_min}:{j_max}, cols {i_min}:{i_max}")
print(f"Need to decode at least {j_max * NI} points")
