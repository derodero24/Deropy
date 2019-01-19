import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cv2
import Deropy.common as cmn


def pil2cv(image):
    new_image = np.array(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def show_image(image, wait_time=0):
    ''' 画像表示 (縦,横,チャンネル) '''
    if type(image) != np.ndarray:
        image = pil2cv(image)
    cv2.namedWindow('window')
    cv2.imshow('window', image)
    cv2.waitKey(wait_time * 1000)
    cv2.destroyWindow('window')


def plot_df(df, filename, title='', xlim=(None, None), ylim=(None, None)):
    '''データフレームからグラフをプロット（一列目が横軸）'''
    cols = df.columns.values
    x_col, y_cols = cols[0], cols[1:]
    plt.figure()
    # プロット
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
    '''csvファイルからグラフをプロット（一列目が横軸）グラフファイル名はcsvと同じ'''
    df = pd.read_csv(csvname, sep=sep, header=0, index_col=None)
    df = df[items] if not items is None else df
    savename = os.path.splitext(csvname)[0] if savename is None else savename
    plot_df(df, savename, title=title, xlim=xlim, ylim=ylim)