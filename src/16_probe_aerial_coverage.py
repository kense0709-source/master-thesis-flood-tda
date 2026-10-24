"""年度別写真を組み合わせたときの被覆率。全マスと浸水マスから各100を抜き出し、ズーム16のタイルの有無を年ごとに記録する。
組み合わせ：2020年より前（2007〜2019）の和集合、直前の数年（2016〜2019）の和集合。参考に seamlessphoto（全国最新写真）と ort（電子国土基本図オルソ）。"""
import math, sys, time, urllib.request, urllib.error
import geopandas as gpd, pandas as pd

mesh_path, label_path = sys.argv[1], sys.argv[2]
mesh = gpd.read_file(mesh_path)
lab = pd.read_parquet(label_path)
flooded_ids = set(lab.loc[lab.label == 1, "mesh_id"])
mesh = mesh.to_crs(3857)
cent = gpd.GeoSeries(mesh.geometry.centroid, crs=3857).to_crs(4326)
mesh["lon"], mesh["lat"] = cent.x.values, cent.y.values
idcol = "mesh_id" if "mesh_id" in mesh.columns else mesh.columns[0]
samples = {
    "全マス": mesh.sample(100, random_state=1),
    "浸水マス": mesh[mesh[idcol].isin(flooded_ids)].sample(100, random_state=1),
}

def tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

cache = {}
def has(layer, x, y, z=16):
    key = (layer, x, y)
    if key in cache:
        return cache[key]
    found = False
    for ext in ("png", "jpg"):
        url = f"https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.{ext}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "thesis-probe"}), timeout=15) as r:
                found = r.status == 200 and len(r.read()) > 0
        except urllib.error.HTTPError:
            pass
        if found:
            break
    time.sleep(0.05)
    cache[key] = found
    return found

years = list(range(2007, 2020))
for name, s in samples.items():
    tiles = [tile(lo, la, 16) for lo, la in zip(s.lon, s.lat)]
    table = {y: [has(f"nendophoto{y}", x, t) for x, t in tiles] for y in years}
    n = len(tiles)
    print(f"--- {name}（{n}）")
    print("  年ごと:", " ".join(f"{y}:{sum(v)}" for y, v in table.items()))
    for lo_y in (2007, 2013, 2016):
        u = sum(any(table[y][i] for y in range(lo_y, 2020)) for i in range(n))
        print(f"  {lo_y}〜2019 のどれかにある: {u}/{n}")
    for layer in ("seamlessphoto", "ort"):
        print(f"  {layer}: {sum(has(layer, x, t) for x, t in tiles)}/{n}")
