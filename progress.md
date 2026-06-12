# 修論進捗ノート
TDA＋土地利用＋雨量によるメッシュ単位浸水予測

---

## フェーズ全体

| Phase | 内容 | 状態 |
|-------|------|------|
| Phase 0 | 事前調査・流域確定 | ✅ 完了 |
| Phase 1 | データ整備・前処理 | 🔄 進行中 |
| Phase 2 | TDA特徴抽出 | ✅ 完了 |
| Phase 3 | モデル構築・比較実験 | ⬜ 未着手 |
| Phase 4 | 検証流域への適用・考察 | ⬜ 未着手 |
| Phase 5 | 論文執筆 | ⬜ 未着手 |

---

## Phase 1 チェックリスト

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 1 | 浸水推定図GeoJSON取得 | ✅ 完了 | 2020第1報(1900件)・第2報(2055件)・2023(1266件) |
| 2 | Python環境構築 | ✅ 完了 | 下記パッケージ一覧参照 |
| 3 | DEM前処理（GML→GeoTIFF→500mメッシュ） | ✅ 完了 | 10,025メッシュ生成 |
| 3b | データ目視確認（DEM・メッシュ・浸水推定図） | ✅ 完了 | 地形・浸水範囲ともに妥当 |
| 4 | 解析雨量GRIB2の読み込み・メッシュ対応づけ | ✅ 完了 | 2020・2023イベント各10,025メッシュ |
| 5 | 浸水推定図GeoJSON → メッシュラベル化（0/1） | ✅ 完了 | 2020: 6.2%浸水, 2023: 5.8%浸水 |
| 6 | TWI・傾斜・流向の計算 | ✅ 完了 | TWI:0.43〜30.8, 傾斜:0〜84° |
| 7 | Sentinel-2 NDVI/NDWI取得・メッシュ平均化 | ✅ 完了 | NDVI/MNDWI 各イベント取得完了 |

---

---

## Phase 2 チェックリスト

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 8 | DEMパッチ切り出し・TDA特徴抽出 | ✅ 完了 | H0×400+H1×400=800次元、11.6分 |
| 9 | 全特徴量統合 | ✅ 完了 | 20,050行×818列のマスターテーブル |
| 11 | Persistence Diagram/Image可視化 | ✅ 完了 | 浸水あり・なし各10件×20枚 |

---

## Phase 2 作業ログ

### 2026-06-12

#### ✅ TDA特徴抽出（Step 8）
- スクリプト: `src/08_tda_features.py`
- 手法: gudhi.CubicalComplex（sublevel set filtration）
- 処理: 各500mメッシュのDEMパッチ（約84×100px）を切り出し → H0/H1 Persistence Diagram → Persistence Image（20×20）
- 所要時間: **11.6分**（10,025メッシュ）
- 出力: `output/tda_features.parquet`（10,025行 × 801列）
- 特徴量次元: **H0×400 + H1×400 = 800次元**

| 指標 | H0（窪み）| H1（ループ）|
|------|---------|---------|
| 浸水ありノルム平均 | 370.8 | **3,177.5** |
| 浸水なしノルム平均 | 871.5 | 2,162.4 |
| ゼロベクトル（浸水あり）| 0.0% | — |
| ゼロベクトル（浸水なし）| 22.5% | — |

#### ✅ 全特徴量統合（Step 9）
- スクリプト: `src/09_merge_features.py`
- 出力: `output/master_features.parquet`（20,050行 × 818列）
- 特徴量数: 815（静的+動的+TDA）
- Sentinel-2 NaN（664件）→ イベント別平均で補完

#### ✅ TDA可視化（Step 11）
- スクリプト: `src/11_tda_visualize.py`
- 出力: `output/tda_vis/`（flooded_01〜10.png, nonflooded_01〜10.png 計20枚）
- 各図の構成: ①DEMヒートマップ ②DEM 3Dサーフェス ③H0/H1 Persistence Diagram ④H0/H1 Persistence Image

---

## 作業ログ

### 2026-06-05

#### ✅ 浸水推定図GeoJSON取得
- ソース: 国土地理院（無償公開）
- 保存先: `浸水推定図/`
  - `2020_第1報/` — 2020年7月8日, 1,900 features (Polygon)
  - `2020_第2報/` — 2020年7月9日, 2,055 features (Polygon) ← 学習ラベルに使用予定
  - `2023/` — 2023年7月10日, 1,266 features (MultiPolygon) ← テストラベルに使用予定

#### ✅ Python環境構築
- Python 3.12.3
- インストール済みパッケージ:

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| geopandas | 1.1.3 | GeoJSON・空間演算 |
| rasterio | 1.5.0 | ラスタ(DEM)処理 |
| pysheds | 0.5 | TWI・流向・流域計算 |
| xgboost | 3.2.0 | 予測モデル |
| lightgbm | 4.6.0 | 予測モデル |
| shap | 0.52.0 | 特徴重要度可視化 |
| ripser | 0.6.12 | TDA計算 |
| persim | 0.3.8 | Persistence Image変換 |
| cfgrib | 0.9.15.1 | GRIB2雨量データ読み込み |
| pyarrow | (最新) | Parquet読み書き |

#### ✅ DEM前処理
- 入力: 国土地理院 DEM5A GMLファイル 2,364枚
- 処理内容:
  1. `src/01_gml_to_geotiff.py` — GML(XML) → GeoTIFF変換（2,364タイル）
  2. `src/02_merge_dem.py` — タイル結合 → `output/dem_merged.tif`
  3. `src/03_create_mesh.py` — 500m×500mメッシュ作成 → `output/mesh_500m.gpkg`
- 結果:
  - DEMモザイク: 13,500 cols × 10,500 rows
  - 範囲: 経度 130.25〜131.00°E, 緯度 32.92〜33.50°N
  - 有効メッシュ数: **10,025セル**（有効ピクセル率30%以上）
- 出力ファイル:
  - `output/dem_merged.tif` — 統合DEM（5m解像度）
  - `output/mesh_500m.gpkg` — 500mメッシュ（GeoPackage）
  - `output/mesh_dem_stats.parquet` — 標高統計（elev_mean/min/max/std）

#### ✅ 解析雨量GRIB2の読み込み・メッシュ対応づけ
- スクリプト: `src/04_rainfall_to_mesh.py`
- 処理内容: 気象庁GRIB2バイナリをPythonでデコード（ランレングス展開）→ 筑後川流域抽出 → 500mメッシュへ最近傍割り当て
- バグ修正: maxvにV=55（ランレングス閾値）を使用すべきところをM=98（レベル最大値）にしていたのを修正
- 出力: `output/rainfall_features.parquet`
- 結果:

| イベント | 最大1h雨量 | 最大24h雨量 |
|---------|-----------|-----------|
| 2020年7月豪雨 | 105mm | 631mm |
| 2023年6月大雨 | 77mm | 514mm |

#### ✅ Sentinel-2 NDVI/MNDWI 取得・メッシュ平均化
- スクリプト: `src/07_sentinel2_ndvi.py`
- データソース: Microsoft Planetary Computer（無料・認証不要）
- 使用バンド: B03(Green), B04(Red), B08(NIR), B11(SWIR) @ 10m解像度
- 指数定義:
  - NDVI = (B08-B04)/(B08+B04) … 植生量
  - MNDWI = (B03-B11)/(B03+B11) … 水域検出
- 使用シーン（豪雨前の晴天日）:

| イベント | 取得日 | 使用タイル | 雲量 |
|---------|--------|-----------|------|
| 2020年7月豪雨 | 2020-05-11 | T52SFB + T52SFC (4シーンモザイク) | 0.4〜3.5% |
| 2023年6月大雨 | 2023-05-16 | T52SFB | 8.1% |

- 出力: `output/sentinel2_features.parquet`（列: mesh_id, event, ndvi_mean, ndvi_min, mndwi_mean, mndwi_max）
- 結果:

| イベント | NDVI平均 | MNDWI平均 | 有効メッシュ数 |
|---------|---------|---------|------------|
| 2020年 | 0.086 | 0.151 | 10,025/10,025 (100%) |
| 2023年 | 0.096 | -0.011 | 9,361/10,025 (93.4%) |

#### ✅ TWI・傾斜・流向の計算
- スクリプト: `src/06_terrain_features.py`
- 処理内容:
  1. DEM（EPSG:6668）→ UTM53N（EPSG:32653）再投影（5.58m解像度）
  2. pysheds: pit fill → resolve flats → D8流向 → 流域累積
  3. numpy gradient で傾斜[度]計算
  4. TWI = ln(acc × cellsize / tan(slope)) 計算
  5. rasterstats でゾーン統計 → 500mメッシュ平均・最大割り当て
- 出力:
  - `output/terrain_features.parquet`（列: mesh_id, twi_mean, slope_mean, slope_max）
  - `output/twi.tif`（TWIラスタ、確認用）
  - `output/slope.tif`（傾斜ラスタ、確認用）
- 結果:

| 特徴量 | 最小 | 平均 | 最大 |
|--------|------|------|------|
| TWI | 0.43 | 5.97 | 13.11 |
| slope_mean [°] | 1.81 | 43.18 | 83.86 |
| slope_max [°] | 0.20 | 16.52 | 42.76 |

#### ✅ 浸水推定図GeoJSON → メッシュラベル化（0/1）
- スクリプト: `src/05_make_labels.py`
- 処理内容: 浸水推定図GeoJSON（EPSG:4326）をUTM53Nに変換 → 500mメッシュと空間結合（intersects）→ ラベル付与
- 出力: `output/mesh_labels.parquet`（列: mesh_id, event, label）
- 結果:

| イベント | 浸水セル (label=1) | 非浸水セル (label=0) | 陽性率 |
|---------|-------------------|---------------------|--------|
| 2020年7月豪雨 | 621 | 9,404 | 6.2% |
| 2023年6月大雨 | 583 | 9,442 | 5.8% |

- クラス不均衡: 陽性率5〜6% → XGBoostの`scale_pos_weight`で補正予定（≈15〜17倍）

#### ✅ データ目視確認
- スクリプト: `src/check_data.py`
- 出力: `output/check_data.png`
- 確認結果:
  - DEM: 山地（東側）・平野（筑後川沿い）の地形が正常に再現されている
  - 500mメッシュ: DEMと同範囲に正しく配置されている
  - 浸水推定図: 2020年（青）・2023年（橙）の浸水域が低標高の川沿いに集中 → 地理的に妥当

---

## ファイル構成

```
修論/
├── progress.md              ← このファイル
├── research_note.pdf        ← 研究ノート
├── src/
│   ├── 01_gml_to_geotiff.py
│   ├── 02_merge_dem.py
│   └── 03_create_mesh.py
├── output/
│   ├── dem_tiles/           ← GeoTIFFタイル (2364枚)
│   ├── dem_merged.tif       ← 統合DEM
│   ├── mesh_500m.gpkg       ← 500mメッシュ
│   └── mesh_dem_stats.parquet
├── DEM/                     ← 国土地理院 DEM5A GMLデータ
├── 解析雨量データ/           ← 気象庁 解析雨量 GRIB2 (2020・2023)
└── 浸水推定図/              ← 国土地理院 浸水推定図 GeoJSON
    ├── 2020_第1報/
    ├── 2020_第2報/
    └── 2023/
```
