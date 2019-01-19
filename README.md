# Deropy
自作Pythonモジュール<br>

## 構成
* common.py : 広く使われる関数


## common.py
### 関数
* `path(path)` : ユーザディレクトリからのパス
* `dpath(path)` : データセットディレクトリからのパス
* `get_files(directory)` : ディレクトリ下のファイルを取得
* `filename(url, maxlen=53)` : ファイル名変換
* `shuffle_lists(list1, list2, seed=None)` : リストをまとめてシャッフル
* `nfd(filename)` : NFD変換
* `name_index(basename, digits=2, first=0)` : インデックス付きで名前生成
* `system()` : OS判定
* `system_func(mac, win, lin=None, others=None)` : OSに応じて戻り値を変える
