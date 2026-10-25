"""筑後川（国管理）の洪水浸水想定区域 L1/L2 を取り出して中身を確認する（調査用）"""
import sys, zipfile, pathlib
import geopandas as gpd, pandas as pd

root = pathlib.Path(sys.argv[1])          # 修論フォルダ
zp = root / "ハザードマップ/raw/A31a-25_89_10_GEOJSON.zip"
ex = root / "ハザードマップ/extract_89"
ex.mkdir(exist_ok=True)
out = open(sys.argv[2], "w", encoding="utf-8")
p = lambda *a: print(*a, file=out, flush=True)

want = {"A31a-10-25_89_8909060001_10.geojson": "L1", "A31a-20-25_89_8909060001_10.geojson": "L2",
        "A31a-30-25_89_8909060001_10.geojson": "dur"}
with zipfile.ZipFile(zp) as z:
    for info in z.infolist():
        name = info.filename.replace("\\", "/").split("/")[-1]
        if name in want and not (ex / name).exists():
            (ex / name).write_bytes(z.read(info))

mesh = gpd.read_file(root / "output/mesh_500m.gpkg")
p("mesh crs", mesh.crs, len(mesh), mesh.total_bounds)
for name, tag in want.items():
    g = gpd.read_file(ex / name)
    p("=====", tag, name, len(g), g.crs)
    p(g.dtypes.to_string()); p(g.head(3).drop(columns="geometry").to_string())
    p("geom types", g.geom_type.value_counts().to_dict())
    p("bounds", g.total_bounds)
    gu = g.to_crs(mesh.crs)
    rank_col = [c for c in g.columns if c.endswith("05")]
    if rank_col:
        rc = rank_col[0]
        a = gu.assign(area_km2=gu.area / 1e6).groupby(rc)["area_km2"].agg(["count", "sum"])
        p(a.to_string()); p("total km2", gu.area.sum() / 1e6)
        # ランク間の重なり（同じ場所に複数ランクがあるか）
        u = gu.geometry.union_all()
        p("union km2", u.area / 1e6)
        # メッシュとの重なり率
        inter = gpd.overlay(mesh[["mesh_id", "geometry"]], gpd.GeoDataFrame(geometry=[u], crs=mesh.crs), how="intersection")
        frac = inter.area / 250000.0
        p("meshes touched", len(inter), "of", len(mesh))
        p("overlap frac quantiles", frac.describe(percentiles=[.1, .25, .5, .75, .9]).to_string())
        for th in [0.0, 0.1, 0.25, 0.5]:
            p(f"mesh frac>{th}: {(frac > th).sum()}  ({(frac > th).sum() / len(mesh):.1%})")
out.close()
