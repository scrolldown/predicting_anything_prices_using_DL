#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')


import time
import tensorflow as tf
from itertools import product
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from datetime import timezone, datetime, timedelta
from matplotlib import pyplot as plt
from numpy import array
import logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		end_ix = i + n_steps
		if end_ix > len(sequence)-1:
			break
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# ## 로깅 사용
# 로깅 초기화
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s %(message)s')
# StreamHandler
streamingHandler = logging.StreamHandler()
streamingHandler.setFormatter(formatter)
logger.addHandler(streamingHandler)
# FileHandler
file_handler = logging.FileHandler('predictTomorrowLog.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# # 모델의 n_steps만큼의 데이터 넣기...
# ## 최고 성능(accuracy) 모델의 하이퍼파라미터 불러오기
# Load highest accuracy AI model info in "result.csv"
df_result = pd.read_csv('result.csv', sep=',')
df_hypParams_Info = df_result[df_result['accuracy']==max(df_result['accuracy'])]
df_hypParams_Info

# ### 설정 변경 부분 *************************************************
# 모델 하이퍼파라미터 수정
df_hypParams_Info.at[df_hypParams_Info.index[0],'epochs']=200

# 원하는 날짜로 학습하게끔 수정
train_startday = '20091201'
train_endday = '20211220'

# 원본데이터 feature 개수 reshape 위해 넘겨주기
## 만약, 학습에 close price 말고 또 다른 feature 사용한다면 여기가 바뀌어야함
n_features = 1
## *********************************************************************************


# ## 불러온 하이퍼파라미터를 이용하여 동일한 raw_data로 모델 학습
## 원본데이터 생성
df_raw = pd.read_csv('SP_SPX, 1D.csv'
                ,parse_dates=['time']
                ,index_col=['time'])
df_raw.index=pd.to_datetime(df_raw.index, unit='s')

df=df_raw['close'][datetime.strptime(train_startday,'%Y%m%d'):datetime.strptime(train_endday,'%Y%m%d')]

# 최고 acc 갱신 위한 변수는 이미 예측한 조합 중 최고 accuracy로 초기화
global_high_acc = max(df_hypParams_Info['accuracy'])


# ## 모델 생성
for i in df_hypParams_Info.index:    
    # data set-up
    test_cutoff_date = df.index.max() - timedelta(days=int(len(df)*df_hypParams_Info['test_ratio'][i]))
    df_test = df[df.index > test_cutoff_date]
    df_train = df[df.index <= test_cutoff_date]

    # log, diff
    df_norm_train = np.log(df_train).diff()[1:]*df_hypParams_Info['scaler'][i]
    df_norm_test = np.log(df_test).diff()[1:]*df_hypParams_Info['scaler'][i]
    raw_seq = df_norm_train

    X, y = split_sequence(raw_seq, int(int(df_hypParams_Info['n_steps'][i])))

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # 차분값을 양수면 1, 음수면 0으로 바꿔주는 y_label 생성
    ### ==> 1,0을 target으로 학습하는거 아니면 필요 없음
    # y_label=((pd.DataFrame(y)[0]>0)*1).replace(0,df_hypParams_Info['negative_target'][i])

    # optimizer 설정
    opt=keras.optimizers.Adam(learning_rate=df_hypParams_Info['learning_rate'][i])

    # CNN-LSTM ------------------------------
    # CNN 모델을 만드는 부분. 
    # TCN (Temporal Conv. NN)
    model = keras.Sequential()
                        # 요약된 정보의 차원 갯수, 요약을 할 time window의 크기.
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(int(df_hypParams_Info['n_steps'][i]), 1))) # 29, 64
    model.add(layers.MaxPooling1D(pool_size=2)) # 14,64
    model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(int(df_hypParams_Info['n_steps'][i]), 1))) # 13,128
    model.add(layers.MaxPooling1D(pool_size=2)) # 6,128
    model.add(layers.Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(int(df_hypParams_Info['n_steps'][i]), 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu')) # 뉴럴 네트워크
    model.add(layers.Dense(1))
    model.add(layers.RepeatVector(1))
    model.add(layers.LSTM(64, activation=None, return_sequences = True
                ,input_shape=(int(df_hypParams_Info['n_steps'][i]), 1)))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(32, activation=None, return_sequences = True))
    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu'))) # LSTM의 리턴값으로 받은 시퀀스 각각에 대해 수행
    model.add(layers.TimeDistributed(layers.Dense(1)))
    model.add(layers.Reshape((1,)))
            # 그라이언트를 계산
    model.compile(optimizer=opt, loss='mae')
    
    logger.info(model.summary())
    
    start = time.time()
    history = model.fit(X, y, epochs=int(df_hypParams_Info['epochs'][i]), batch_size=int(df_hypParams_Info['batch'][i])
                        ,verbose=1)
    model.save('high_acc_CNN-LSTM_SP500.h5')
    logger.info('fit완료 '+\
                "e:"+str(df_hypParams_Info['epochs'][i])+", "+\
                "lr:"+str(df_hypParams_Info['learning_rate'][i])+", "+\
                "t:"+str(df_hypParams_Info['test_ratio'][i])+", "+\
                "n:"+str(int(df_hypParams_Info['n_steps'][i]))+", "+\
                "s:"+str(df_hypParams_Info['scaler'][i])+", "+\
                "b:"+str(int(df_hypParams_Info['batch'][i]))
            )
    end = time.time()
    # fit 완료 ----------------------------------------------
    # ----------------------------------------------
    # 테스트데이터 생성
    test_raw_seq = df_norm_test

    test_X, test_y = split_sequence(test_raw_seq, int(df_hypParams_Info['n_steps'][i]))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
    y_pred = model.predict(test_X)
    test_mae_loss = np.mean(np.abs(y_pred - test_y), axis=1)
    # ----------------------------------------------
    test_df_with_pred = df_norm_test.to_frame()[int(df_hypParams_Info['n_steps'][i]):]
    test_df_with_pred['pred'] = y_pred
    # 모델, 데이터 생성 끝------------------------------

    # ------------ 성능 기록 ---------------------------
    # confusion matrix 위해 tp,tn,fp,fn 계산
    test_df_with_pred['close_yn']=(test_df_with_pred['close']>0)*1
    test_df_with_pred['pred_yn']=(test_df_with_pred['pred']>0)*1

    test_tp = (test_df_with_pred['close_yn']*test_df_with_pred['pred_yn']).sum()
    test_tn = ((test_df_with_pred['close_yn']+test_df_with_pred['pred_yn'])==0).sum()
    test_fp = test_df_with_pred['pred_yn'].sum() - test_tp
    test_fn = test_df_with_pred['close_yn'].sum() - test_tp

    test_accuracy = (test_tp+test_tn) / len(test_df_with_pred['pred_yn'])
    test_recall = test_tp / (test_df_with_pred['close_yn'].sum())
    test_precision = test_tp / (test_df_with_pred['pred_yn'].sum())
    
    logger.info("test 기준 "+\
                ", acc : "+ str(np.round(test_accuracy,4))+\
                ", rec : "+ str(np.round(test_recall,4))+ \
                ", pre : "+ str(np.round(test_precision,4))+\
                ", time : "+str(timedelta(seconds=end-start))
               )


fig, ax = plt.subplots(figsize=(10,4))
test_df_with_pred['close'].plot(ax=ax, legend=True)
test_df_with_pred['pred'].plot(ax=ax, legend=True)
ax.set_title("train t:"+str(df_result['test_ratio'][i])+", "+\
                "n:"+str(int(df_result['n_steps'][i]))+", "+\
                "s:"+str(df_result['scaler'][i])+", "+\
                "b:"+str(int(df_result['batch'][i]))
            )
fig.savefig('high_acc_model.jpg')

# ## 예측에 사용 할 데이터 불러오기

# ### yfinance를 사용해서 오늘까지의 데이터를 불러옴
import pytz
import yfinance as yf

## 원본데이터를 n_steps길이 만큼 생성
### 영업일 기준 n_steps길이만큼 생성해야 하므로,
### /5를하고 8을 곱해서 (5영업일로 나누고 *8일의 길이)
### yfinance에서 넉넉히 데이터 추출하고
### 그중에 n_steps만큼 데이터 가져오기
now = datetime.now(timezone.utc)

starttime = now - timedelta(days=(df_hypParams_Info['n_steps'][df_hypParams_Info.index[0]]//5*8))
today = str(now.year)+"-"+str(now.month)+"-"+str(now.day)
startday = str(starttime.year)+"-"+str(starttime.month)+"-"+str(starttime.day)
df_raw_untilNow = yf.download('^GSPC', start=startday, end=today)[-int( df_hypParams_Info['n_steps'][df_hypParams_Info.index[0]] ):]


df_close_untilNow = df_raw_untilNow['Close']


# Model 에 들어갔던 데이터와 동일한 data 만들기 (로그차분, scaler 등)

df_processed_data = np.log(df_close_untilNow).diff()*df_hypParams_Info['scaler'][df_hypParams_Info.index[0]]
df_processed_data = np.array(df_processed_data).reshape(( n_features, df_processed_data.shape[0], n_features))


# ## 예측하기
pred_tomorrow = model.predict(df_processed_data)

if pred_tomorrow>0 :
        logger.info(str(df_raw_untilNow.index[-1])+" 기준 예측 : "+\
        "상승. (diff : "+str(pred_tomorrow)+")\t"+\
        " acc : "+ str(np.round(test_accuracy,4))+\
        ", rec : "+ str(np.round(test_recall,4))+ \
        ", pre : "+ str(np.round(test_precision,4))+\
        ", time : "+str(timedelta(seconds=end-start))
        )
else : logger.info(str(datetime.now())+" 기준 예측 : "+\
        "하락. (diff : "+str(pred_tomorrow)+")\t"+\
        " acc : "+ str(np.round(test_accuracy,4))+\
        ", rec : "+ str(np.round(test_recall,4))+ \
        ", pre : "+ str(np.round(test_precision,4))+\
        ", time : "+str(timedelta(seconds=end-start))
        )



