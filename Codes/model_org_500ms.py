import os             
import glob
import pandas as pd
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from datetime import timedelta
# import matplotlib.pyplot as plt
# import warnings
import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


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

    #### merge both positive and negative together
    data_final = pd.concat([crash_feature_label_300ms_500ms, noncrash_feature_label_300ms_500ms])
    data_final = data_final[['features_cal_vel','features_org_vel','label']]

    # records_try = data_final.iloc[:20000, :]
    inputs = data_final.features_org_vel
    targets = data_final.label

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)

    # K-fold Cross Validation model evaluation
    class_weights =[{0:1,1:1}, 
            {0:1,1:10},
            {0:1,1:20},
            {0:1,1:50}]

    results_all = {}
    for i in range(len(class_weights)):
        fold_no = 1
        print(f'Training for weights {i} ...')
        
        auc_score = []
        accuracy = []
        precision = []
        recall = []
        
        for train_index, test_index in kfold.split(inputs):
            x_train = inputs.iloc[train_index]
            y_train = targets.iloc[train_index]
            
            x_test = inputs.iloc[test_index]
            y_test = targets.iloc[test_index]

            ## data for training
            x_train = sequence.pad_sequences(x_train, maxlen=50, padding='post', dtype='float', truncating='post')
            y_train = np.array(y_train).reshape(len(y_train),1)
            
            ## data for testing
            x_test = sequence.pad_sequences(x_test, maxlen=50, padding='post', dtype='float', truncating='post')
            y_test = np.array(y_test).reshape(len(y_test),1)

            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            enc = enc.fit(y_train)
            y_train = enc.transform(y_train)
            y_test = enc.transform(y_test)


            model = keras.Sequential()
            model.add(
                keras.layers.LSTM(
                    units=128,
                    input_shape=[x_train.shape[1], x_train.shape[2]]
                )
                )

            model.add(keras.layers.Dropout(rate=0.5))
            model.add(keras.layers.Dense(units=128, activation='relu'))
            model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
            
            model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=[tf.keras.metrics.AUC()]
            #   metrics=['categorical_accuracy']
            )

            print('------------------------------------------------------------------------')
            print(f'Training for fold {fold_no} ...')


            history = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            shuffle=False,
            class_weight = class_weights[i]
            )
            
            auc = model.evaluate(x_test, y_test)
            y_pred = model.predict(x_test)
            auc_score.append(round(auc[1]*100, 2))
            # Increase fold number
            fold_no = fold_no + 1
            
            ## results 
            predictions = y_pred[:,0]
            predictions[predictions>=0.5] = 1
            predictions[predictions<0.5] = 0
            testing = y_test[:,0]
            
            cf_array = confusion_matrix(testing, predictions)
            aa = pd.DataFrame(cf_array)
            accuracy.append( (aa.iloc[0,0] + aa.iloc[1,1])/np.sum(cf_array) )
            precision.append(aa.iloc[0,0]/(aa.iloc[0,0] + aa.iloc[1,0]))
            recall.append(aa.iloc[0,0]/(aa.iloc[0,0] + aa.iloc[0,1]))

            print('0000000000000000', aa)
            # print('1111111111111',precision)
            # print('2222222222222',recall)
            
        results_all[i] = [round(np.nanmean(auc_score),2), round(np.nanstd(auc_score),2),
                        round(np.nanmean(accuracy),2), round(np.nanstd(accuracy),2),
                        round(np.nanmean(precision),2), round(np.nanstd(precision),2),
                        round(np.nanmean(recall),2), round(np.nanstd(recall),2)]
            
    pd.DataFrame(results_all).to_csv('results/results_' + str(int(time_scale_train*1000)) + '/' + name3 +'_' + name4+ '_' + 'results_all.csv')


if __name__ == "__main__":
    time_early = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    for i in time_early:
        MyModel(time_scale_train = 0.5,
        time_ahead = i)


        

        # df_resutls.to_csv('results/results_1000/'+ str(i)+'_500ms_1000ms_original_vel'+'.csv')

        # print(predictions_cal.shape, testing_cal.shape)

