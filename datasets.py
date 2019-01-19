import pickle
import random
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

import Deropy.common as cmn
import Deropy.visual as vsl


def Cifar10(num=50000, seed=None):
    '''Cifar10 データジェネレーター'''
    # データ読み込み
    images, labels = [], []
    for i in range(5):
        with open(cmn.dpath('cifar10/data_batch_' + str(i + 1)), 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            images.extend(d[b'data'])
            labels.extend(d[b'labels'])
    labels = np.array(labels)
    # ランダムシード設定
    if not seed is None:
        old_state = np.random.get_state()
        np.random.seed(seed)
    # インデックス抽出
    indexes = np.array([], dtype='uint8')
    for i in range(10):
        i_index = np.where(labels == i)[0]
        i_index = np.random.choice(i_index, num // 10)
        indexes = np.append(indexes, i_index)
    np.random.shuffle(indexes)
    # ランダムシード復元
    if not seed is None:
        np.random.set_state(old_state)
    # yield
    while True:
        for idx in indexes:
            image = images[idx].reshape(3, 32, 32)
            # (channel, row, column) => (row, column, channel)
            image = image.transpose(1, 2, 0)
            label = np.array([0 for i in range(10)], dtype='uint8')
            label[labels[idx]] = 1
            yield image, label


def Dogs_vs_Cats(size=None, num=25000, seed=None):
    '''Dogs vs Cats データジェネレーター'''
    dir = cmn.dpath('Dogs_vs_Cats/')
    # ランダムシード設定
    if not seed is None:
        old_state = random.getstate()
        random.seed(seed)

    with zipfile.ZipFile(dir + 'train.zip', 'r') as z:
        # ファイル名取得 (0番はディレクトリ名)
        cat_files = random.sample(z.namelist()[1:12501], num // 2)
        dog_files = random.sample(z.namelist()[12501:], num // 2)
        files = cat_files + dog_files
        print(len(z.namelist()))
        # 正解ラベル
        labels = [0 if i < num // 2 else 1 for i in range(num)]
        labels = np.array(labels, dtype='uint8')
        # シャッフル
        files, labels = cmn.shuffle_lists(files, labels, seed=seed)

        # ランダムシード復元
        if not seed is None:
            random.setstate(old_state)

        # yield
        while True:
            for file, label in zip(files, labels):
                with Image.open(z.open(file)) as image:
                    if not size is None:
                        image = image.resize(size)
                    image = np.array(image, dtype='uint8')
                    yield image, label


def Sudoku(num=1000000, seed=None):
    '''数独データジェネレーター'''
    df = pd.read_csv(cmn.dpath('sudoku.csv'))
    # ランダムシード設定
    if not seed is None:
        old_state = random.getstate()
        random.seed(seed)
    indexes = random.sample(list(range(1000000)), num)
    # ランダムシード復元
    if not seed is None:
        random.setstate(old_state)
    # yield
    while True:
        for idx in indexes:
            q = list(df['quizzes'][idx])
            s = list(df['solutions'][idx])
            q = np.array(q, dtype='uint8')
            s = np.array(s, dtype='uint8')
            yield q, s


if __name__ == '__main__':
    # generator = Dogs_vs_Cats(seed=0)
    # generator = Cifar10(seed=0)
    generator = Sudoku(seed=0)
    image, label = next(generator)
    print(image.shape, label)
    # for i in range(10):
    #     image, label = next(generator)
    #     print(image.shape, label)
    #     vsl.show_image(image)
    generator.close()
