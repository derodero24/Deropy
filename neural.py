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
