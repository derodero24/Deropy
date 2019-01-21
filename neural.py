import os
import pickle
from copy import deepcopy
from importlib import import_module, machinery

import numpy as np
import pandas as pd

import cv2
import Deropy.common as cmn


def keras_gpu_options():
    ''' keras gpu設定 '''
    tf = import_module('tensorflow')
    tfb = import_module('keras.backend.tensorflow_backend')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0,1'
    tfb.set_session(tf.Session(config=config))


def save_model(model, filename, framework='keras', args=[], kwargs={}):
    '''モデル・重みの保存'''
    if framework == 'keras':
        with open(filename + '.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(filename + '.h5')
    elif framework == 'pytorch':
        model_cpu = deepcopy(model).cpu()
        inspect = import_module('inspect')
        state = {'module_path': inspect.getmodule(model).__file__,
                 'class_name': model.__class__.__name__,
                 'state_dict': model_cpu.state_dict(),
                 'args': args,
                 'kwargs': kwargs}
        with open(filename + '.pkl', 'wb') as f:  # 一時処置
            pickle.dump(state, f)


def _save_model(model, filename, framework='keras'):
    '''モデル・重みの保存'''
    if framework == 'keras':
        with open(filename + '.json', 'w') as f:
            f.write(model.to_json())
        model.save_weights(filename + '.h5')
    elif framework == 'pytorch':
        if model.__module__ == '__main__':
            module_name = __file__
        else:
            module_name = model.__module__
        import_class = model.__class__.__name__
        state = {'import_path': os.path.abspath(module_name + '.py'),
                 'module_name': module_name,
                 'class_name': import_class,
                 'state_dict': model.state_dict()}
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(state, f)


def load_model(filename, framework='keras'):
    '''モデル・重みの読み込み'''
    if framework == 'keras':
        k_models = import_module('keras.models')
        with open(filename + '.json', 'r') as f:
            model = k_models.model_from_json(f.read())
        model.load_weights(filename + '.h5')
    elif framework == 'pytorch':
        with open(filename + '.pkl', 'rb') as f:
            state = pickle.load(f)
        module = machinery.SourceFileLoader(
            state['module_path'], state['module_path']).load_module()
        args, kwargs = state['args'], state['kwargs']
        model = getattr(module, state['class_name'])(*args, **kwargs)
        model.load_state_dict(state['state_dict'])
    return model


def _load_model(filename, framework='keras', args={}):
    '''モデル・重みの読み込み'''
    from keras.models import model_from_json
    if framework == 'keras':
        with open(filename + '.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(filename + '.h5')
    elif framework == 'pytorch':
        with open(filename + '.pkl', 'rb') as f:
            state = pickle.load(f)
        module = machinery.SourceFileLoader(
            state['module_name'], state['import_path']).load_module()
        model = getattr(module, state['class_name'])(**args)
        model.load_state_dict(state['state_dict'])
    return model


def save_hist(history, filename):
    '''学習履歴を保存'''
    data = {}
    data['epoch'] = list(range(len(history.history['loss'])))
    data['loss'] = history.history['loss']
    data['acc'] = history.history['acc']
    data['val_loss'] = history.history['val_loss']
    data['val_acc'] = history.history['val_acc']
    pd.DataFrame(data).to_csv(filename + '.csv', index=None)


def cal_eval(labels, predict, stride=0.05):
    '''評価指標の保存'''
    sklm = import_module('sklearn.metrics')
    # 閾値
    thresholds = [round(i * stride, 2) for i in range(round(1 / stride) + 1)]
    thresholds[0], thresholds[-1] = 0.01, 0.99
    # ネガティブ基準
    labels_neg = [0 if l else 1 for l in labels]
    predict_neg = [1 - p for p in predict]
    # 計算
    acc_list, recall_list, prec_list = [], [], []
    acc_neg_list, recall_neg_list, prec_neg_list = [], [], []
    for threshold in thresholds:
        tmp_pred = [1 if p > threshold else 0 for p in predict]
        tmp_pred_neg = [1 if p > threshold else 0 for p in predict_neg]
        # 正解率
        acc_list.append(sklm.accuracy_score(labels, tmp_pred))
        acc_neg_list.append(sklm.accuracy_score(labels_neg, tmp_pred_neg))
        # 再現率
        recall_list.append(sklm.recall_score(labels, tmp_pred))
        recall_neg_list.append(sklm.recall_score(labels_neg, tmp_pred_neg))
        # 適合率
        prec_list.append(sklm.precision_score(labels, tmp_pred))
        prec_neg_list.append(sklm.precision_score(labels_neg, tmp_pred_neg))
    # 保存
    df = pd.DataFrame({
        'threshold': thresholds,
        'accuracy': acc_list,
        'accuracy_neg': acc_neg_list,
        'recall': recall_list,
        'recall_neg': recall_neg_list,
        'precision': prec_list,
        'precision_neg': prec_neg_list})
    return df


class ImageDataGenerator:
    ''' keras用DataGenerator '''

    def __init__(self, rescale=None):
        '''rescale=1/255'''
        self.reset()
        self.rescale = rescale

    def reset(self):
        self.images = []
        self.labels = []
        # self.images = np.array([], dtype=np.float32)
        # self.labels = np.array([], dtype=np.float32)

    def flow_from_list(self, files, labels,
                       imgsize, batch_size, shuffle=True, seed=None):
        '''リストからバッチ生成'''
        '''imgsize=(height, width)'''
        self.reset()
        # データ数チェック
        if len(files) != len(labels):
            print("files and labels are different length")
            return
        # シャッフル
        if shuffle:
            files, labels = cmn.shuffle_lists(files, labels, seed)
        # バッチ作成
        while True:
            for i in range(len(files)):
                try:
                    if not os.path.exists(files[i]):
                        files[i] = cmn.nfd(files[i])
                    img = cv2.imread(files[i])  # 読み込み
                    img = cv2.resize(img, imgsize)  # リサイズ
                    # リスケール
                    if not self.rescale is None:
                        img = img.astype(np.float32)
                        img *= self.rescale
                    # リストに追加
                    self.images.append(img)
                    self.labels.append(labels[i])

                    # データが溜まったら
                    if len(self.images) == batch_size:
                        self.images = np.asarray(self.images, dtype=np.float32)
                        self.labels = np.asarray(self.labels, dtype=np.float32)
                        yield self.images, self.labels
                        self.reset()
                except Exception as ex:
                    print(i, files[i], str(ex))
                    exit()
