#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import time
import tensorflow as tf
from itertools import product
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from numpy import array
import logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ## 로깅 사용

# In[4]:


# 로깅 초기화
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s %(message)s')
# StreamHandler
streamingHandler = logging.StreamHandler()
streamingHandler.setFormatter(formatter)
logger.addHandler(streamingHandler)
# FileHandler
file_handler = logging.FileHandler('getLog.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# # 모델의 n_steps만큼의 데이터 넣기...

# ## 최고 성능(accuracy) 모델의 하이퍼파라미터 불러오기

# In[28]:


# Load highest accuracy AI model info
high_acc_model = keras.models.load_model("high_CNN-LSTM_SP500.h5")
model_idx = int(high_acc_model.get_config()['name'].split('_')[1])

df_result = pd.read_csv('result.csv', sep=',')
df_hypParams_Info = df_result[model_idx:model_idx+1]
df_hypParams_Info
#pred_now = high_acc_model.predict(until_now)
#pred_now


# ## 사용할 데이터 불러오기

# In[71]:


df_raw = pd.read_csv('SP_SPX, 1D_until211218.csv'
                ,parse_dates=['time']
                ,index_col=['time'])
df_raw.index=pd.to_datetime(df_raw.index, unit='s')

# raw_data의 마지막 날짜 부터 n_steps 날짜까지의 데이터 구하기
df_close_untilNow = pd.DataFrame(df_raw['close'][-df_hypParams_Info['n_steps'].values[0]:])


# Model 에 들어갔던 데이터와 동일한 data 만들기 (로그차분, scaler 등)

df_processed_data = np.log(df_close_untilNow).diff()*df_hypParams_Info['scaler'].values[0]
df_processed_data = np.array(df_processed_data).reshape(( df_processed_data.shape[1],df_processed_data.shape[0], len(df_close_untilNow.columns)))
df_processed_data[:10]


# ## 예측하기

# In[76]:


pred_tomorrow = high_acc_model.predict(df_processed_data)

if pred_tomorrow>0 :
    print('상승  예측 diff :', pred_tomorrow)
else : print('하락  예측 diff :', pred_tomorrow)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,4))
train_df_with_pred['close'].plot(ax=ax, legend=True)
train_df_with_pred['pred'].plot(ax=ax, legend=True)
ax.set_title("train t:"+str(df_result['test_ratio'][i])+", "+              "n:"+str(int(df_result['n_steps'][i]))+", "+              "s:"+str(df_result['scaler'][i])+", "+              "b:"+str(int(df_result['batch'][i]))
            )

