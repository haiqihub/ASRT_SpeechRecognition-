#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os
from keras import backend as K
from LanguageModel2 import ModelLanguage
from keras.optimizers import SGD, Adadelta, Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from SpeechModel251 import ModelSpeech

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 进行配置，使用95%的GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
# config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

datapath = ''
modelpath = 'model_speech'

if (not os.path.exists(modelpath)):  # 判断保存模型的目录是否存在
    os.makedirs(modelpath)  # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
if (system_type == 'Windows'):
    datapath = 'C:\\Users\\Dell\\PycharmProjects\\untitled3\\ASRT_SpeechRecognition\\dataset'
    modelpath = modelpath + '\\'
elif (system_type == 'Linux'):
    datapath = 'dataset'
    modelpath = modelpath + '/'
else:
    print('*[Message] Unknown System\n')
    datapath = 'dataset'
    modelpath = modelpath + '/'

# add  迁移训练后的识别结果
ms = ModelSpeech(datapath)
ms.LoadModel(modelpath + 'm251c/speech_model251c_e_0_step_3100.model')
# r = ms.fine_tune(datapath)
# result = r.RecognizeSpeech_FromFile('F:\\ASRT_SpeechRecognition\\A2\\A2_6.wav')
# K.clear_session()
# print('*[提示]语音识别结果： \n', result)
#
# ml = ModelLanguage('model_language')
# ml.LoadModel()
# str_pinyin = result
# result = ml.SpeechToText(str_pinyin)
# print('语音转文字结果：\n', result)
# add~

ms.TrainModel(datapath, epoch=20, batch_size=8, save_step=100)
