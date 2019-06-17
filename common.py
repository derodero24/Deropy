import os
import platform
import random
import re
import unicodedata
from inspect import currentframe
from logging import DEBUG, FileHandler, Formatter, StreamHandler
from urllib.request import urlparse


def path(path):
    ''' ユーザディレクトリからのパス '''
    return os.path.expanduser(path)


def dpath(path):
    ''' データセットディレクトリからのパス '''
    return os.path.expanduser('~/Datasets/' + path)


def get_files(directory):
    ''' ディレクトリ下のファイルを取得 '''
    ignore = ('.DS_Store')
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file in ignore:
                file_list.append(os.path.join(root, file))
    return sorted(file_list)


def filename(url, maxlen=53):
    ''' ファイル名変換 '''
    o = urlparse(url)
    filename, ext = os.path.splitext(o.netloc + o.path)
    filename = re.sub(r'[:*?"<>|/\\]', '_', filename)  # 禁止文字置換
    if filename[-1] == '_':
        filename += 'index'
    if len(filename) > maxlen:
        filename = filename[:maxlen // 2 - 1] + \
            '...' + filename[-maxlen // 2 - 1:]
    return filename, ext


def shuffle_lists(*lists, seed=None):
    ''' リストをまとめてシャッフル '''
    if not seed is None:
        old_state = random.getstate()
        random.seed(seed)
    zipped = list(zip(*lists))
    random.shuffle(zipped)
    if not seed is None:
        random.setstate(old_state)
    return list(zip(*zipped))


def nfd(filename):
    ''' NFD変換 '''
    return unicodedata.normalize('NFD', filename)


def name_index(basename, digits=2, first=0):
    ''' インデックス付きで名前生成 '''
    i = first
    style = '%0' + str(digits) + 'd'
    while True:
        yield basename + '_' + str(style % i)
        i += 1


def system():
    ''' OS判定 '''
    val = {'Darwin': 'mac', 'Windows': 'win', 'Linux': 'lin'}
    name = platform.system()
    if name in val.keys():
        return val[name]
    else:
        return ''


def system_func(mac, win, lin=None, others=None):
    ''' OSに応じて戻り値を変える '''
    val = {'mac': mac, 'win': win, 'lin': lin, '': others}
    return val[system()]


def get_variable_name(var):
    ''' 変数名取得 '''
    names = {id(v): k for k, v in currentframe().f_back.f_locals.items()}
    return names.get(id(var), None)


def init_logger(logger, filename='log'):
    log_fmt = Formatter(
        '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(f'{filename}.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)


if __name__ == '__main__':
    print(system_func(mac='Mac', win='Windows',
                      lin='Linux', others='cannot identify your OS'))
