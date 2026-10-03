"""流路網の診断：集水面積が小さすぎる原因を調べる"""
import numpy as np, rasterio
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder

P = 'output/dem_utm_route.tif'
NOD = -9999.0
grid = Grid.from_raster(P)
dem = grid.read_raster(P)
arr = np.asarray(dem, dtype=np.float32)
valid = arr != NOD
floor = float(arr[valid].min()) - 1000.0
work = np.where(valid, arr, floor).astype(np.float32)
vf = dem.viewfinder
wvf = ViewFinder(affine=vf.affine, shape=vf.shape, crs=vf.crs,
                 nodata=np.float32(-1e9), mask=np.ones(vf.shape, bool))
wr = Raster(work, viewfinder=wvf)
print('fill_pits...', flush=True)
a1 = grid.fill_pits(wr)
print('fill_depressions...', flush=True)
a2 = grid.fill_depressions(a1)
print('resolve_flats...', flush=True)
a3 = grid.resolve_flats(a2)
print('flowdir...', flush=True)
fdir = grid.flowdir(a3)
print('accumulation...', flush=True)
acc = np.asarray(grid.accumulation(fdir), dtype=np.float32)

with rasterio.open(P) as r:
    cs = r.res[0]; prof = r.profile.copy(); tr = r.transform
km2 = acc * cs**2 / 1e6
print()
print('集水面積 [km2] 全体最大: %.0f' % km2.max())
print('  有効域内の最大      : %.0f' % km2[valid].max())
for t in [1,10,50,100,200,300,500]:
    print('  > %4d km2 のセル数: %s' % (t, format(int((km2>t).sum()), ',')))
# 最大位置
i,j = np.unravel_index(np.argmax(km2), km2.shape)
x,y = tr * (j+0.5, i+0.5)
import pyproj
tf = pyproj.Transformer.from_crs('EPSG:32653','EPSG:4326', always_xy=True)
lon,lat = tf.transform(x,y)
print('  最大位置: %.4fE %.4fN  標高 %.1f m' % (lon,lat,arr[i,j]))
# 有効域の縁に接するセルのうち集水面積が大きいもの＝流入/流出点
from scipy import ndimage
edge = valid & ~ndimage.binary_erosion(valid)
ek = km2[edge]
print()
print('有効域の縁のセル: {:,}'.format(int(edge.sum())))
print('  縁での集水面積 最大 %.0f km2 / >10km2 の縁セル数 %d' % (ek.max(), int((ek>10).sum())))
big = np.argsort(ek)[-5:][::-1]
ei, ej = np.where(edge)
for k in big:
    x,y = tr * (ej[k]+0.5, ei[k]+0.5)
    lon,lat = tf.transform(x,y)
    print('    %.4fE %.4fN : %.0f km2' % (lon,lat,ek[k]))
prof.update(dtype='float32', nodata=-9999.0)
with rasterio.open('output/flowacc_km2.tif','w',**prof) as d:
    d.write(np.where(valid, km2, -9999.0).astype(np.float32), 1)
print('\n保存: output/flowacc_km2.tif')
