import imghdr
import io
import os
import subprocess
from urllib.request import unquote

import chardet
import chromedriver_binary
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as Options_ch
from selenium.webdriver.firefox.options import Options as Options_ff
from tqdm import tqdm

import Deropy.common as cmn

# ブラウザパス
CHROME_CANARY = cmn.system_func(
    mac='/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    win='C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe')
FIREFOX = cmn.system_func(
    mac='/Applications/Firefox Developer Edition.app/Contents/MacOS/firefox',
    win='')


def get_driver(width=960, height=540, firefox=False):
    ''' ウェブドライバー取得 '''
    if not firefox:
        options = Options_ch()
        options.binary_location = CHROME_CANARY
        options.add_argument('--headless')
        options.add_argument('--hide-scrollbars')
        options.add_argument('--incognito')  # シークレッドモード
        options.add_argument('--window-size=' + str(width) + ',' + str(height))
        driver = webdriver.Chrome(chrome_options=options)
    else:
        options = Options_ff()
        options.binary_location = FIREFOX
        options.add_argument('-headless')
        options.add_argument('-private')
        driver = webdriver.Firefox(firefox_options=options,
                                   log_path=os.devnull)
    return driver


def encode_bytes(b_content):
    ''' バイト列をutf-8文字列に変換 '''
    # デコード処理
    encoding = chardet.detect(b_content)['encoding']
    try:
        encoded = b_content.decode(encoding)
        return encoded
    except Exception as ex:
        return b_content


def imageExt(b_content):
    ''' 画像形式の判定 '''
    ext = imghdr.what(None, b_content)
    if ext is None and b_content[:2] == b'\xff\xd8':
        ext = 'jpeg'
    return ext


def imageSize(b_content):
    ''' 画像サイズ取得 (横px, 縦px) '''
    img_bin = io.BytesIO(b_content)  # bytesオブジェクトを生成
    pil_img = Image.open(img_bin)  # PILで読み込み
    return pil_img.size


def save_images(url_list, basename, dir='./', digits=3):
    ''' 画像をインデックス付きでまとめて保存 '''
    dir = os.path.join(dir, basename)
    os.makedirs(dirname, exist_ok=True)
    name = cmn.name_index(basename, digits, first=1)  # ファイル名ジェネレータ

    for url in tqdm(url_list):
        url = unquote(url)  # urlの文字長エラー回避
        # アクセス
        try:
            content = requests.get(url).content
        except Exception as ex:
            print(url, '\n->', ex)
            continue

        # 画像形式を判定
        ext = imageExt(content)
        if ext is None:
            ext = os.path.splitext(url)[1][1:]

        # 保存
        filename = os.path.join(dir, next(name) + '.' + ext)
        with open(filename, 'wb') as f:
            f.write(content)


def pageSize(driver):
    ''' ページサイズ取得 (横, 縦) '''
    w = driver.execute_script("return document.body.scrollWidth;")
    h = driver.execute_script("return document.body.scrollHeight;")
    # print(w, h)
    return (w, h)


def screenShotFull(driver, filename, timeout=30, firefox=False):
    ''' フルページ スクリーンショット '''
    url = driver.current_url  # url取得
    # コマンド作成
    if not firefox:
        size = pageSize(driver)  # ページサイズ取得
        cmd = 'gtimeout ' + str(timeout)  \
            + ' "' + CHROME_CANARY + '"' \
            + ' --headless' \
            + ' --hide-scrollbars' \
            + ' --incognito' \
            + ' --screenshot=' + filename + '.png' \
            + ' --window-size=' + str(size[0]) + ',' + str(size[1]) \
            + ' ' + url
    else:
        cmd = 'gtimeout ' + str(timeout)  \
            + ' "' + FIREFOX + '"' \
            + ' -headless' \
            + ' -private' \
            + ' -screenshot=' + filename + '.png' \
            + ' ' + url
    # コマンド実行
    subprocess.Popen(cmd, shell=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)


if __name__ == '__main__':
    pass
