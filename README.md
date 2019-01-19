# Deropy
自作Pythonモジュール<br>

## 構成
* common.py : 広く使われる関数
* neural.py : ニューラルネット関係


## common.py
### 関数
* `path(path)`: ユーザディレクトリからのパス
* `dpath(path)`: データセットディレクトリからのパス
* `get_files(directory)`: ディレクトリ下のファイルを取得
* `filename(url, maxlen=53)`: ファイル名変換
* `shuffle_lists(list1, list2, seed=None)`: リストをまとめてシャッフル
* `nfd(filename)`: NFD変換
* `name_index(basename, digits=2, first=0)`: インデックス付きで名前生成
* `system()`: OS判定
* `system_func(mac, win, lin=None, others=None)`: OSに応じて戻り値を変える


## neural.py
### 関数
* `keras_gpu_options`: keras gpu設定
* `save_model(model, filename)`: モデル・重みの保存
* `_save_model`: `save_model`の廃止版
* `load_model(filename)`: モデル・重みの読み込み
* `_load_model`: `load_model`の廃止版
* `save_hist(history, filename)`: 学習履歴を保存
* `cal_eval(labels, predict, stride=0.05)`: 評価指標の保存
### クラス
* `ImageDataGenerator(rescale=None)`: keras用DataGenerator
