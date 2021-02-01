#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
import platform as plat

from SpeechModel251 import ModelSpeech
from LanguageModel2 import ModelLanguage
from keras import backend as K

datapath = ''
modelpath = 'model_speech'

system_type = plat.system()  # 由于不同的系统的文件路径表示不一样，需要进行判断
if (system_type == 'Windows'):
    # datapath = 'dataset'
    datapath = './'
    modelpath = modelpath + '\\'
elif (system_type == 'Linux'):
    datapath = 'dataset'
    modelpath = modelpath + '/'
else:
    print('*[Message] Unknown System\n')
    datapath = 'dataset'
    modelpath = modelpath + '/'


def pickModel(modelType):
    if (modelType == "0"):
        name = 'show/061-speech_model251_e_0_step_625000.model'
        return name
    if (modelType == "1"):
        name = 'show/lh-speech_model251_e_0_step_2900.model'
        return name
    if (modelType == "2"):
        name = 'show/yq-speech_model251_e_0_step_4800.model'
        return name
    if (modelType == "3"):
        name = 'show/hq1speech_model251_e_0_step_410.model'
        return name
    if (modelType == "4"):
        name = 'show/hq2-speech_model251_e_0_step_500.model'
        return name
    else:
        name = '类型输入不正确，请核查。'
        return name


ms = ModelSpeech(datapath)

print('\n*成人语音模型：type=0 \n*直接训练模型：type=1 \n*基于成人模型训练的模型：type=2 \n*迁移学习1.0模型 type=3 \n*迁移学习2.0模型 type=4 \n')
print('-----请输入您想调用的模型类型。-----')
modelType = input("**type**：")  # 0 1 2 3 4

name = pickModel(modelType)
if name == "类型输入不正确，请核查。":
    print("模型的类型输入不符合要求，请核查。")
else:
    ms.LoadModel(modelpath + name)

    r = ms.RecognizeSpeech_FromFile("./dataset/data_child/test/A5_75.wav")
    K.clear_session()
    print('*[提示] 语音识别结果：\n', r)
    ml = ModelLanguage('model_language')
    ml.LoadModel()
    str_pinyin = r
    r = ml.SpeechToText(str_pinyin)
    print('*[提示] 转文字结果：\n', r)

#  测试 模型 wer 代码
# for i in range(0, 8):
#     s = 100 + i * 300
#     ms.LoadModel(modelpath + 'm251d\\speech_model251d_e_0_step_' + str(s) + '.model')
#     print("model:" + str(s))
#     ms.TestModel(datapath, str_dataset='train', data_count=300, out_report=False)
