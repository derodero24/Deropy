import os
import pickle
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
        inspect = import_module('inspect')
        module_path = inspect.getmodule(model).__file__
        class_name = model.__class__.__name__
        state = {'module_path': module_path,
                 'class_name': class_name,
                 'state_dict': model.cpu().state_dict(),
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
