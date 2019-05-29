import os
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import Deropy.common as cmn


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = deepcopy(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def add_alpha(image):
    ''' αチャンネル追加 '''
    new_image = deepcopy(image)
    if type(image) == np.ndarray:
        new_image = cv2pil(new_image)
    new_image = new_image.convert('RGBA')
    if type(image) == np.ndarray:
        new_image = pil2cv(new_image)
    return new_image


def show_image(image, wait=0):
    ''' 画像表示 '''
    new_image = deepcopy(image)
    if type(new_image) != np.ndarray:
        new_image = pil2cv(new_image)
    cv2.namedWindow('window')
    cv2.imshow('window', new_image)
    cv2.waitKey(wait * 1000)
    cv2.destroyWindow('window')


def set_limits(xlim=(None, None), ylim=(None, None)):
    ''' x軸,y軸の範囲を設定 '''
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xmin = min(xmin, xlim[0]) if not xlim[0] is None else xmin
    xmax = max(xmax, xlim[1]) if not xlim[1] is None else xmax
    ymin = min(ymin, ylim[0]) if not ylim[0] is None else ymin
    ymax = max(ymax, ylim[1]) if not ylim[1] is None else ymax
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)


def plot_df(dfs, filename, title='', xlim=(None, None), ylim=(None, None)):
    ''' データフレームからグラフをプロット（一列目が横軸） '''
    plt.figure()
    # リスト化
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]
    # プロット
    for df in dfs:
        cols = df.columns.values
        x_col, y_cols = cols[0], cols[1:]
        for y_col in y_cols:
            plt.plot(df[x_col], df[y_col], marker='.', label=y_col)
    # 各種設定
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.xlabel(x_col)
    plt.ylabel(', '.join(y_cols))
    # グラフ範囲
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xmin = min(xmin, xlim[0]) if not xlim[0] is None else xmin
    xmax = max(xmax, xlim[1]) if not xlim[1] is None else xmax
    ymin = min(ymin, ylim[0]) if not ylim[0] is None else ymin
    ymax = max(ymax, ylim[1]) if not ylim[1] is None else ymax
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    # 保存
    plt.savefig(filename + '.png')


def plot_csv(csvname, savename=None, items=None, title='',
             xlim=(None, None), ylim=(None, None), sep=','):
    ''' csvファイルからグラフをプロット（一列目が横軸）グラフファイル名はcsvと同じ '''
    df = pd.read_csv(csvname, sep=sep, header=0, index_col=None)
    df = df[items] if not items is None else df
    savename = os.path.splitext(csvname)[0] if savename is None else savename
    plot_df(df, savename, title=title, xlim=xlim, ylim=ylim)
