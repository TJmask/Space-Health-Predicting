import os             
import glob
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from datetime import timedelta
import matplotlib.pyplot as plt
import warnings
from pandas.core.common import SettingWithCopyWarning

# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# %matplotlib inline

from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

def MyModel(time_scale_train,
        time_ahead):
    name1 = 'data_' + str(int(time_ahead*1000))+ 'ms' ## data_300ms
    name2 = 'data_' + str(int(time_scale_train*1000))+ 'ms' ## data_500ms
    name3 = str(int(time_ahead*1000))+ 'ms' ## 300ms
    name4 = str(int(time_scale_train*1000))+ 'ms' ## 500ms
    #### read all the positive samples
    crash_feature_label_300ms_500ms = pd.read_pickle('data/'+ name2 + '/crash_feature_label_' + name3 + '_' + name4 +'_test')
   
    ### read all the negative samples
    noncrash_feature_label_300ms_500ms = pd.read_pickle('data/'+ name2 + '/noncrash_feature_label_' + name3 + '_' + name4 +'_test')



    records = pd.concat([crash_feature_label_300ms_500ms, noncrash_feature_label_300ms_500ms])
    # records.index = np.arange(0, len(records) )
    # records.head()
    inputs = records.features_org_vel
    targets = records.label

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=22)

    K_train_index = []
    K_test_index =[]
    for train, test in kfold.split(inputs, targets):
        K_train_index.append(train)
        K_test_index.append(test)

    class_weights = [
            {0:1,1:1},
            {0:1,1:10},
            {0:1,1:20},
            {0:1,1:50}
            ]

    results_all = {}

    for w in range(len(class_weights)):
        print('wwwwwwwwww', w)
        auc_score = []
        accuracy = []
        precision = []
        recall = []
        for i in range(10):
            print('kkkkkkkkkkkk', i)

            X_train = np.array([np.vstack(i) for i in inputs.iloc[K_train_index[i]]])
            X_test = np.array([np.vstack(i) for i in inputs.iloc[K_test_index[i]]])

            y_train = targets.iloc[K_train_index[i]]
            y_test = targets.iloc[K_test_index[i]]

            X_train = sequence.pad_sequences(X_train, maxlen=50, padding='post', dtype='float', truncating='post')
            y_train = np.array(y_train).reshape(len(y_train),1)

            X_test = sequence.pad_sequences(X_test, maxlen=50, padding='post', dtype='float', truncating='post')
            y_test = np.array(y_test).reshape(len(y_test),1)

            
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            enc = enc.fit(y_train)
            y_train = enc.transform(y_train)
            y_test = enc.transform(y_test)

            ### train model
            
            model = keras.Sequential()
            model.add(
                keras.layers.LSTM(
                    units=128,
                    input_shape=[X_train.shape[1], X_train.shape[2]]
                )
                )

            model.add(keras.layers.Dropout(rate=0.5))
            model.add(keras.layers.Dense(units=128, activation='relu'))
            model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
            model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.Recall(),tf.keras.metrics.Precision()]
            #   metrics=['categorical_accuracy']
            )


            history = model.fit(
                X_train, y_train,
                epochs=30,
                batch_size=32,
                validation_split=0.1,
                shuffle=True,
                class_weight = class_weights[w]
            )


            results =model.evaluate(X_test, y_test)
            y_pred = model.predict(X_test)

            auc_score.append(round(results[1]*100, 2))

            predictions = np.argmin(y_pred, axis=1)
            testing = y_test[:,0]
            cf_array = confusion_matrix(testing, predictions)

            aa = pd.DataFrame(cf_array)
            accuracy.append( (aa.iloc[0,0] + aa.iloc[1,1])/np.sum(cf_array) )
            precision.append(aa.iloc[0,0]/(aa.iloc[0,0] + aa.iloc[1,0]))
            recall.append(aa.iloc[0,0]/(aa.iloc[0,0] + aa.iloc[0,1]))

            print(cf_array)
            
        
        results_all[w] = [round(np.nanmean(auc_score),2), round(np.nanstd(auc_score),2),
                            round(np.nanmean(accuracy),2), round(np.nanstd(accuracy),2),
                            round(np.nanmean(precision),2), round(np.nanstd(precision),2),
                            round(np.nanmean(recall),2), round(np.nanstd(recall),2)]

        
                
        pd.DataFrame(results_all).to_csv('results/results_' + str(int(time_scale_train*1000)) + '/' + name3 +'_' + name4+ '_' + 'results_all.csv')
    
if __name__ == "__main__":
    # time_early = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    time_early = [0.5]
    for i in time_early:
        MyModel(time_scale_train = 2.0,
        time_ahead = i)

