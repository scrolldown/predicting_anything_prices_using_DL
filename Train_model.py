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


# # 학습변수

# ## GRID_SEARCH 하이퍼파라미터 설정

# In[5]:


# 고정
epochs=[50]
learning_rate=[0.001]

# gird_search 대상 변수
test_ratio =[0.2, 0.3]
n_steps = [30,60,90]
scaler=[10,100,1000]
batch=[64,128,256,512]


# epoch 말고 또 loop만큼 돌려보기
## ==> loop 1초과로 하려면 평균내서 df_result 저장하기?
loop=5


# In[6]:


# grid_search dataframe 생성
items=[epochs, learning_rate, test_ratio, n_steps, scaler, batch]

search_df=pd.DataFrame(list(product(*items)),columns=['epochs', 'learning_rate', 'test_ratio', 'n_steps', 'scaler', 'batch'])


# In[7]:


search_df['is_predict']='N'
search_df['tp']=0
search_df['tn']=0
search_df['fp']=0
search_df['fn']=0
search_df['accuracy']=0.0
search_df['recall']=0.0
search_df['precision']=0.0
search_df['loss_history_1']=0.0
search_df['loss_history_2']=0.0
search_df['loss_history_3']=0.0
search_df['loss_history_4']=0.0
search_df['time']=''
search_df


# In[8]:


# 이미 예측한 하이퍼파라미터 조합은 제외하도록 기존에 있던 result 와 drop_duplicates
df_readFile = pd.read_csv('result.csv', sep=',')
df_predict_y = df_readFile[df_readFile['is_predict']=='Y']
# is_predict가 N인것만 남기고 search_df랑 concat
df_merge = pd.concat([df_predict_y, search_df], ignore_index=True)
df_merge = df_merge.drop_duplicates(subset=['epochs','learning_rate','test_ratio'
                                            ,'n_steps','scaler','batch']).reset_index(drop=True)
df_merge = df_merge[df_merge['is_predict']=='N']
df_merge['index']=df_merge.index.values


## model setting 중 오류나지않도록 int로 변환
df_merge['epochs']=df_merge['epochs'].astype(int)
df_merge['n_steps']=df_merge['n_steps'].astype(int)
df_merge['scaler']=df_merge['scaler'].astype(int)
df_merge['batch']=df_merge['batch'].astype(int)


# In[9]:


df_predict_y


# In[10]:


df_result = df_merge
df_result


# ## 데이터 n_steps 단위로 잘라줄 함수 생성

# In[11]:


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


# ## 원본데이터 생성

# In[12]:


df_raw = pd.read_csv('SP_SPX, 1D.csv'
                ,parse_dates=['time']
                ,index_col=['time'])
df_raw.index=pd.to_datetime(df_raw.index, unit='s')

#06년 1월~ 21년 11월까지
df=df_raw['close'][14123:-5]
df


# In[13]:


# ai 모델 객체 저장해 둘 model_list 생성
model_list=[]

# 원본데이터 feature 개수 reshape 위해 넘겨주기
n_features = 1

# 최고 acc 갱신 위한 변수는 이미 예측한 조합 중 최고 accuracy로 초기화
#  --> 예측한 데이터가 없었으면 최고 acc 는 0
if len(df_predict_y)!=0 :
    global_high_acc = max(df_predict_y['accuracy'])
else : global_high_acc = 0


# ## 모델 생성 및 학습 및 성능측정

# In[ ]:


for i in df_result.index:
    
    # grid_search 조합 별 평균 변수 초기화
    local_test_tp_sum = 0
    local_test_tn_sum = 0
    local_test_fp_sum = 0
    local_test_fn_sum = 0
    local_test_accuracy_sum = 0
    local_test_recall_sum = 0
    local_test_precision_sum = 0
    local_test_time_sum = 0
    local_test_loss_history_1_sum = 0
    local_test_loss_history_2_sum = 0
    local_test_loss_history_3_sum = 0
    local_test_loss_history_4_sum = 0
        
    # data set-up
    test_cutoff_date = df.index.max() - timedelta(days=int(len(df)*df_result['test_ratio'][i]))
    df_test = df[df.index > test_cutoff_date]
    df_train = df[df.index <= test_cutoff_date]

    # log, diff
    df_norm_train = np.log(df_train).diff()[1:]*df_result['scaler'][i]
    df_norm_test = np.log(df_test).diff()[1:]*df_result['scaler'][i]
    raw_seq = df_norm_train

    X, y = split_sequence(raw_seq, int(int(df_result['n_steps'][i])))

    X = X.reshape((X.shape[0], X.shape[1], n_features))
    
    # 차분값을 양수면 1, 음수면 0으로 바꿔주는 y_label 생성
    ### ==> 1,0을 target으로 학습하는거 아니면 필요 없음
    # y_label=((pd.DataFrame(y)[0]>0)*1).replace(0,df_result['negative_target'][i])

    # optimizer 설정
    opt=keras.optimizers.Adam(learning_rate=df_result['learning_rate'][i])

    # CNN-LSTM ------------------------------
    # CNN 모델을 만드는 부분. 
    # TCN (Temporal Conv. NN)
    model = keras.Sequential()
                           # 요약된 정보의 차원 갯수, 요약을 할 time window의 크기.
    model.add(layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(int(df_result['n_steps'][i]), 1))) # 29, 64
    model.add(layers.AveragePooling1D(pool_size=2)) # 14,64
    model.add(layers.Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(int(df_result['n_steps'][i]), 1))) # 13,128
    model.add(layers.MaxPooling1D(pool_size=2)) # 6,128
    model.add(layers.Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(int(df_result['n_steps'][i]), 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu')) # 뉴럴 네트워크
    model.add(layers.Dense(1))
    model.add(layers.RepeatVector(1))
    model.add(layers.LSTM(64, activation=None, return_sequences = True
                  ,input_shape=(int(df_result['n_steps'][i]), 1)))
    model.add(layers.Dropout(0.5))
    model.add(layers.LSTM(32, activation=None, return_sequences = True))
    model.add(layers.TimeDistributed(layers.Dense(100, activation='relu'))) # LSTM의 리턴값으로 받은 시퀀스 각각에 대해 수행
    model.add(layers.TimeDistributed(layers.Dense(1)))
    model.add(layers.Reshape((1,)))
            # 그라이언트를 계산
    model.compile(optimizer=opt, loss='mae')

    if i==df_result.index[0]:
        model.summary()
        
    
    logger.info('\t'+str(i)+" t:"+str(df_result['test_ratio'][i])+", "+                "n:"+str(int(df_result['n_steps'][i]))+", "+                "s:"+str(df_result['scaler'][i])+", "+                "b:"+str(int(df_result['batch'][i]))
                 )
    
    # loop 번째 시도하는 for 반복문
    for j in range(loop):
        start = time.time()
        history = model.fit(X, y, epochs=int(df_result['epochs'][i]), batch_size=int(df_result['batch'][i])
                            ,verbose=0) # epoch 표시 끄기        
        # ----------------------------------------------
        train_pred = model.predict(X)
        end = time.time()
        # ----------------------------------------------
        train_df_with_pred = df_norm_train.to_frame()[int(df_result['n_steps'][i]):]
        train_df_with_pred['pred'] = train_pred
        
        # ----------------------------------------------
        # 테스트데이터 생성
        test_raw_seq = df_norm_test

        test_X, test_y = split_sequence(test_raw_seq, int(df_result['n_steps'][i]))
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
        y_pred = model.predict(test_X)
        test_mae_loss = np.mean(np.abs(y_pred - test_y), axis=1)
        # ----------------------------------------------
        test_df_with_pred = df_norm_test.to_frame()[int(df_result['n_steps'][i]):]
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
        
        quartile_len_loss_history=len(history.history['loss'])//4
        
        local_test_tp_sum = local_test_tp_sum + test_tp
        local_test_tn_sum = local_test_tn_sum + test_tn
        local_test_fp_sum = local_test_fp_sum + test_fp
        local_test_fn_sum = local_test_fn_sum + test_fn
        local_test_accuracy_sum = local_test_accuracy_sum + test_accuracy
        local_test_recall_sum = local_test_recall_sum + test_recall
        local_test_precision_sum = local_test_precision_sum + test_precision
        local_test_time_sum = local_test_time_sum + (end-start)
        local_test_loss_history_1_sum = local_test_loss_history_1_sum + history.history['loss'][quartile_len_loss_history*1-1]
        local_test_loss_history_2_sum = local_test_loss_history_2_sum + history.history['loss'][quartile_len_loss_history*2-1]
        local_test_loss_history_3_sum = local_test_loss_history_3_sum + history.history['loss'][quartile_len_loss_history*3-1]
        local_test_loss_history_4_sum = local_test_loss_history_4_sum + history.history['loss'][quartile_len_loss_history*4-1]
        
        logger.info(str(j+1)+" 번째시도 test"+                   ", acc : "+ str(np.round(test_accuracy,4))+                   ", rec : "+ str(np.round(test_recall,4))+                   ", pre : "+ str(np.round(test_precision,4))+                   ", time : "+str(timedelta(seconds=end-start))
                   )
            
        if j==(loop-1):
            
            df_result.at[i,'is_predict']='Y'
            
            df_result.at[i,'tp'] = local_test_tp_sum // loop
            df_result.at[i,'tn'] = local_test_tn_sum // loop
            df_result.at[i,'fp'] = local_test_fp_sum // loop
            df_result.at[i,'fn'] = local_test_fn_sum // loop
            df_result.at[i,'accuracy']=local_test_accuracy_sum / loop
            df_result.at[i,'recall']=local_test_recall_sum / loop
            df_result.at[i,'precision']=local_test_precision_sum / loop
            df_result.at[i,'time']=timedelta( seconds=(local_test_time_sum//loop) )
            
            df_result.at[i,'loss_history_1'] = history.history['loss'][quartile_len_loss_history*1-1] / loop
            df_result.at[i,'loss_history_2'] = history.history['loss'][quartile_len_loss_history*2-1] / loop
            df_result.at[i,'loss_history_3'] = history.history['loss'][quartile_len_loss_history*3-1] / loop
            df_result.at[i,'loss_history_4'] = history.history['loss'][quartile_len_loss_history*4-1] / loop  
            
            df_result.at[i,'index']=i
            
            df_predict_y = (pd.concat([ df_predict_y, df_result[i:i+1] ], ignore_index=True)).reset_index(drop=True)
            df_predict_y.to_csv('result.csv', sep=',',index=False)
            
            
            # loop만큼의 반복학습 후 평균 accuracy가 가장 높은 하이퍼파라미터 조합이면
            # 해당 모델의 가장 마지막 학습 WEIGHT와 test plot을 저장
            if (local_test_accuracy_sum/loop) > global_high_acc:
                global_high_acc = (local_test_accuracy_sum/loop)
                model.save('high_CNN-LSTM_SP500.h5')
                fig, ax = plt.subplots(figsize=(10,4))
                train_df_with_pred['close'].plot(ax=ax, legend=True)
                train_df_with_pred['pred'].plot(ax=ax, legend=True)
                ax.set_title("train t:"+str(df_result['test_ratio'][i])+", "+                              "n:"+str(int(df_result['n_steps'][i]))+", "+                              "s:"+str(df_result['scaler'][i])+", "+                              "b:"+str(int(df_result['batch'][i]))
                            )
                fig.savefig('high_accruacy_model.jpg')
                logger.info("  ** 최고성능 모델 갱신, acc : "+                            str(local_test_accuracy_sum/loop)+                            " 저장 완료 ** ")
                
        

        
fig, ax = plt.subplots(figsize=(10,4))
df_result[['accuracy','recall','precision']].plot(legend=True)
fig.savefig('Performance.jpg')




# In[ ]:




