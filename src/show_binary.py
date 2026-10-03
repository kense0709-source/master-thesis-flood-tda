import glob, os

files = sorted(glob.glob("解析雨量データ/**/*.bin", recursive=True))
f = files[0]
print("file:", f.split("\\")[-1])
print("size:", f"{os.path.getsize(f):,}", "bytes")
print()

with open(f, "rb") as fh:
    raw = fh.read(80)

print("=== first 80 bytes (hex dump) ===")
for i in range(0, len(raw), 16):
    chunk = raw[i:i+16]
    hex_part = " ".join(f"{b:02X}" for b in chunk)
    ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
    print(f"{i:04X}  {hex_part:<47}  |{ascii_part}|")

print()
print("--- annotation ---")
print(f"[0:4]  magic number : {raw[0:4]}  -> must be 'GRIB'")
print(f"[6]    discipline   : 0x{raw[6]:02X}          -> 0=Meteorology")
print(f"[7]    edition      : 0x{raw[7]:02X}          -> GRIB edition 2")
msg_len = int.from_bytes(raw[8:16], "big")
print(f"[8:16] total length : {msg_len:,} bytes = {msg_len/1024/1024:.2f} MB")
print()

# Section 1 starts at byte 16
s1_len = int.from_bytes(raw[16:20], "big")
print(f"[16:20] Section1 length : {s1_len} bytes")
print(f"[20]    Section number  : {raw[20]}  (=1: Identification)")

# Show raw data section (Section 7) size approximately
print()
print("=== data section image ===")
print("Section 7 (actual rainfall data):")
print("  8,601,600 grid points (2560 col x 3360 row)")
print("  run-length compressed -> 1 byte per decoded point")
print()

# Read last 8 bytes (end marker)
with open(f, "rb") as fh:
    fh.seek(-8, 2)
    tail = fh.read(8)
print(f"last 8 bytes: {' '.join(f'{b:02X}' for b in tail)}")
print(f"              -> must end with '7777' = {tail[-4:]} (GRIB end marker)")
