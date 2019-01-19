# Deropy

自作Pythonモジュール<br>
(2019/01/19更新)

## 構成

-   common.py: 広く使われる関数
-   scraping.py: スクレイピング関係
-   neural.py: ニューラルネット関係
-   visual.py: 可視化関係（画像表示，グラフプロット等）
-   google.py: Google検索関係
-   datasets.py: データセット関係

## common.py

### 関数

-   `path()`: ユーザディレクトリからのパス
-   `dpath()`: データセットディレクトリからのパス
-   `get_files()`: ディレクトリ下のファイルを取得
-   `filename()`: ファイル名変換
-   `shuffle_lists()`: リストをまとめてシャッフル
-   `nfd()`: NFD変換
-   `name_index()`: インデックス付きで名前生成
-   `system()`: OS判定
-   `system_func()`: OSに応じて戻り値を変える

## scraping.py

### 定数

-   `CHROME_CANARY`: Chrome Canaryのパス
-   `FIREFOX`: Firefoxのパス

### 関数

-   `get_driver()`: ウェブドライバー取得
-   `encode_bytes()`: バイト列をutf-8文字列に変換
-   `imageExt()`: 画像形式の判定
-   `imageSize()`: 画像サイズ
-   `save_images()`: 画像をインデックス付きでまとめて保存
-   `pageSize()`: ページサイズ取得
-   `screenShotFull()`: フルページ スクリーンショット

## neural.py

### 関数

-   `keras_gpu_options()`: keras gpu設定
-   `save_model()`: モデル・重みの保存
-   `_save_model()`: `save_model`の廃止版
-   `load_model()`: モデル・重みの読み込み
-   `_load_model()`: `load_model`の廃止版
-   `save_hist()`: 学習履歴を保存
-   `cal_eval()`: 評価指標の保存

### クラス

-   `ImageDataGenerator()`: keras用DataGenerator

## visual.py

### 関数

-   `show_image()`: 画像表示
-   `plot_df()`: データフレームからグラフをプロット
-   `plot_csv()`: csvファイルからグラフをプロット

## google.py

### クラス

-   `Google()`:
    -   `Search()`: Google検索
    _   `Suggest()`: サジェスト取得
    _   `Value()`: 検索回数取得

## datasets.py

### 関数

-   `Dogs_vs_Cats()`: Dogs vs Catsデータ読み込み
-   `Cifar10()`: Cifar10 データジェネレーター
-   `Sudoku()`: 数独データジェネレーター
