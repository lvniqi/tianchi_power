#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:34:33 2017

@author: boweiy
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd
from preprocess import get_day_all_df,get_month_all_df
from preprocess import load_filtered_day_model,load_day_model
from preprocess import load_prophet_model,load_filtered_prophet_model
from preprocess import load_filtered_prophet_exp_model,load_prophet_exp_model

from preprocess import load_extera_holi_prophet_model,load_extera_holi_prophet_filtered_model
from preprocess import load_extera_holi_prophet_exp_model,load_extera_holi_prophet_exp_filtered_model
from preprocess import load_extera_holi_prophet_f2_model,load_extera_holi_prophet_exp_f2_model

from preprocess import load_filtered_pre_predict_model
from preprocess import load_user_exp_model,load_user_model
from preprocess import load_predict_model
from preprocess import get_month_by_id,get_full_user_ids
from preprocess import exp_power
from preprocess import load_prophet_14day_model,\
    load_prophet_14day_filtered_model,\
    load_prophet_14day_f2_model,\
    load_prophet_14day_exp_model,\
    load_prophet_14day_exp_filtered_model,\
    load_prophet_14day_exp_f2_model
    
from preprocess import load_prophet_7day_model,\
    load_prophet_7day_filtered_model,\
    load_prophet_7day_f2_model,\
    load_prophet_7day_exp_model,\
    load_prophet_7day_exp_filtered_model,\
    load_prophet_7day_exp_f2_model
    
from preprocess import load_tiny_7_model,\
    load_tiny_7_filtered_model,\
    load_tiny_7_f2_model,\
    load_tiny_7_exp_model,\
    load_tiny_7_exp_filtered_model,\
    load_tiny_7_exp_f2_model
    
from preprocess import load_np_tiny_7_model,\
    load_np_tiny_7_filtered_model,\
    load_np_tiny_7_f2_model,\
    load_np_tiny_7_exp_model,\
    load_np_tiny_7_exp_filtered_model,\
    load_np_tiny_7_exp_f2_model

from multiprocessing import Pool as m_Pool
from preprocess import get_feature_cloumn,get_feature_cloumn_tiny
from preprocess import filter_spring_festval,filter_sept
import sys,os
from pathos.multiprocessing import ProcessingPool as Pool
import time

_save_paths = [
             #%% extera holi 28 days
             './features/tensorflow_model/extera_holi_month_model/',
             #useless
             './features/tensorflow_model/extera_holi_month_exp_model/',
             
             './features/tensorflow_model/extera_holi_month_filtered_model/',
             #useless
             './features/tensorflow_model/extera_holi_month_filtered_exp_model/',
             
             './features/tensorflow_model/extera_holi_month_f2_model/',
             './features/tensorflow_model/extera_holi_month_f2_exp_model/',
             #%% extera holi 14 days
             #useless
             #'./features/tensorflow_model/month_14_model/',
             #'./features/tensorflow_model/month_14_exp_model/',
             #useless
             #'./features/tensorflow_model/month_14_filtered_model/',
             #'./features/tensorflow_model/month_14_filtered_exp_model/',
             #useless
             #'./features/tensorflow_model/month_14_f2_model/',
             #'./features/tensorflow_model/month_14_f2_exp_model/',
             #%% 7 days
             #'./features/tensorflow_model/month_7_model/',
             #'./features/tensorflow_model/month_7_exp_model/',
             
             './features/tensorflow_model/month_7_filtered_model/',
             './features/tensorflow_model/month_7_filtered_exp_model/',
             
             './features/tensorflow_model/month_7_f2_model/',
             './features/tensorflow_model/month_7_f2_exp_model/',
             #%% tiny 7 days
             #'./features/tensorflow_model/tiny_7_model/',
             #'./features/tensorflow_model/tiny_7_exp_model/',
             
             './features/tensorflow_model/tiny_7_filtered_model/',
             './features/tensorflow_model/tiny_7_filtered_exp_model/',
             
             './features/tensorflow_model/tiny_7_f2_model/',
             './features/tensorflow_model/tiny_7_f2_exp_model/',
             #%% np_tiny 7 days
             #'./features/tensorflow_model/np_tiny_7_model/',
             #'./features/tensorflow_model/np_tiny_7_exp_model/',
             
             #'./features/tensorflow_model/np_tiny_7_filtered_model/',
             #'./features/tensorflow_model/np_tiny_7_filtered_exp_model/',
             
             #'./features/tensorflow_model/np_tiny_7_f2_model/',
             #'./features/tensorflow_model/np_tiny_7_f2_exp_model/',
             #%%
             ]

_create_feature_funcs = [
                       #%% extera holi 28 days                                   
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_extera_holi_month_feature),
                        #useless
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_extera_holi_month_feature),
                                                           
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_filtered_extera_holi_month_feature),
                        ##useless
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_filtered_extera_holi_month_feature),
                                                           
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_f2_extera_holi_month_feature),
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_f2_extera_holi_month_feature),
                                                           
                        #%% extera holi 14 days
                        #useless
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_f2_14day_month_feature_unexp),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_14day_month_feature),
                        #useless                               
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_filtered_14day_month_feature_unexp),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_filtered_14day_month_feature),
                        #useless                                   
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_f2_14day_month_feature_unexp),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_f2_14day_month_feature),
                        #%% extera holi 7 days
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_7day_month_feature),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_7day_month_feature),
                                                           
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_filtered_7day_month_feature),
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_filtered_7day_month_feature),
                                                           
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_f2_7day_month_feature),
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_f2_7day_month_feature),
                        #%% tiny 7 days
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_tiny7_feature),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_tiny7_feature),
                                                           
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_filtered_tiny7_feature),
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_filtered_tiny7_feature),
                                                           
                        lambda :create_predict_feature_all(is_exp=False,
                                     feature_func=create_f2_tiny7_feature),
                        lambda :create_predict_feature_all(is_exp=True,
                                     feature_func=create_f2_tiny7_feature),
                        #%% np_tiny 7 days
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_np_tiny_7feature),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_np_tiny_7feature),
                                                           
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_filtered_np_tiny_7feature),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_filtered_np_tiny_7feature),
                                                           
                        #lambda :create_predict_feature_all(is_exp=False,
                        #             feature_func=create_f2_np_tiny_7feature),
                        #lambda :create_predict_feature_all(is_exp=True,
                        #             feature_func=create_f2_np_tiny_7feature),
                        #%%
                        ]
assert len(_save_paths) == len(_create_feature_funcs)
_feature_length = len(_create_feature_funcs)
def create_features(user_id,is_exp,
                         feature_cloumn_func = lambda day:get_feature_cloumn(None,day,has_user_type=False),
                         load_exp_func = load_user_exp_model,
                         load_func = load_user_model,
                         is_exp_power = False
                        ):
    print user_id
    dataset = get_month_by_id(user_id)
    result = []
    for day in range(1,32):
        feature_column = feature_cloumn_func(day)
        x_ = dataset[feature_column]
        trainer = xgb.XGBRegressor()
        if is_exp:
            if is_exp_power:
                x_ = exp_power(x_)
            load_exp_func(trainer,day,user_id)
        else:
            load_func(trainer,day,user_id)
        y_p = trainer.predict(x_)
        y_p = pd.Series(y_p,name='y_p#%d'%(day-1))
        if not is_exp:
            y_p = np.exp(y_p)
        result.append(y_p)
    result = pd.DataFrame(result).T
    result.index = dataset.index
    for day in range(31):
        result['real#%d'%day] = dataset['y#%d'%day].apply(np.exp)
    sys.stdout.flush()
    return result


    
create_users_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,has_user_type=False),
    load_exp_func = load_user_exp_model,
    load_func = load_user_model)

create_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day),
    load_exp_func = lambda trainer,day,user_id:load_prophet_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_model(trainer,day))

create_filtered_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day),
    load_exp_func = lambda trainer,day,user_id:load_filtered_prophet_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_filtered_prophet_model(trainer,day))
#%% 28 days 
create_extera_holi_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,holiday_range=9,has_extera_holiday=True),
    load_exp_func = lambda trainer,day,user_id:load_extera_holi_prophet_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_extera_holi_prophet_model(trainer,day))

create_filtered_extera_holi_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,holiday_range=9,has_extera_holiday=True),
    load_exp_func = lambda trainer,day,user_id:load_extera_holi_prophet_exp_filtered_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_extera_holi_prophet_filtered_model(trainer,day))

create_f2_extera_holi_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,holiday_range=9,has_extera_holiday=True),
    load_exp_func = lambda trainer,day,user_id:load_extera_holi_prophet_exp_f2_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_extera_holi_prophet_f2_model(trainer,day))
#%% 14days
create_14day_month_feature_unexp = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=14,holiday_range=9,has_extera_holiday=True,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_14day_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_14day_model(trainer,day))

create_14day_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=14,has_prophet=False,has_extera_holiday=True,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_14day_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_14day_model(trainer,day),
    is_exp_power = True)

create_filtered_14day_month_feature_unexp = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=14,holiday_range=9,has_extera_holiday=True,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_14day_exp_filtered_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_14day_filtered_model(trainer,day))

create_filtered_14day_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=14,has_prophet=False,has_extera_holiday=True,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_14day_exp_filtered_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_14day_filtered_model(trainer,day),
    is_exp_power = True)

create_f2_14day_month_feature_unexp = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=14,holiday_range=9,has_extera_holiday=True,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_14day_exp_f2_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_14day_f2_model(trainer,day))

create_f2_14day_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=14,has_prophet=False,has_extera_holiday=True,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_14day_exp_f2_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_14day_f2_model(trainer,day),
    is_exp_power = True)
#%% 7days
create_7day_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=7,has_prophet=False,has_extera_holiday=False,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_7day_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_7day_model(trainer,day),
    is_exp_power = True)

create_filtered_7day_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=7,has_prophet=False,has_extera_holiday=False,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_7day_exp_filtered_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_7day_filtered_model(trainer,day),
    is_exp_power = True)

create_f2_7day_month_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn(None,day,feature_range=7,has_prophet=False,has_extera_holiday=False,has_extera_weather=True),
    load_exp_func = lambda trainer,day,user_id:load_prophet_7day_exp_f2_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_prophet_7day_f2_model(trainer,day),
    is_exp_power = True)
#%% tiny 7days
create_tiny7_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn_tiny(None,day),
    load_exp_func = lambda trainer,day,user_id:load_tiny_7_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_tiny_7_model(trainer,day),)

create_filtered_tiny7_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn_tiny(None,day),
    load_exp_func = lambda trainer,day,user_id:load_tiny_7_exp_filtered_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_tiny_7_filtered_model(trainer,day),)

create_f2_tiny7_feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn_tiny(None,day),
    load_exp_func = lambda trainer,day,user_id:load_tiny_7_exp_f2_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_tiny_7_f2_model(trainer,day),)
#%% np tiny 7days
create_np_tiny_7feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn_tiny(None,day,has_prophet = False),
    load_exp_func = lambda trainer,day,user_id:load_np_tiny_7_exp_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_np_tiny_7_model(trainer,day),)

create_filtered_np_tiny_7feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn_tiny(None,day,has_prophet = False),
    load_exp_func = lambda trainer,day,user_id:load_np_tiny_7_exp_filtered_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_np_tiny_7_filtered_model(trainer,day),)

create_f2_np_tiny_7feature = lambda user_id,is_exp:create_features(user_id,is_exp,
    feature_cloumn_func = lambda day:get_feature_cloumn_tiny(None,day,has_prophet = False),
    load_exp_func = lambda trainer,day,user_id:load_np_tiny_7_exp_f2_model(trainer,day),
    load_func = lambda trainer,day,user_id:load_np_tiny_7_f2_model(trainer,day),)
#%%
def create_predict_feature_all(is_exp=False,feature_func = create_users_feature):
    users_list = []
    for user_id in get_full_user_ids():
        predict_df = feature_func(user_id,is_exp)
        users_list.append((user_id,predict_df))
    return users_list
    
    
def rebuild_predict_feature_all(create_feature_func = lambda :create_predict_feature_all(False),
                save_path = './features/tensorflow_model/user_model/'):
    users_list = create_feature_func()
    for day in range(1,32):
        new_df = pd.DataFrame(index = users_list[0][1].index)
        for user_id,user_df in users_list:
            new_df['y_p#%d'%user_id] = user_df['y_p#%d'%(day-1)]
            new_df['y_#%d'%user_id] = user_df['real#%d'%(day-1)]
        new_df.to_csv(save_path+'%d.csv'%day)
    return users_list

def get_dataset(day):
    all_model_df = map(lambda path:pd.DataFrame.from_csv(path+'%d.csv'%day),_save_paths)
    return all_model_df


def get_batch(data_id_list,length):
    return np.random.choice(data_id_list,length,replace=False)
    
def resample_x_y_(all_dataset,length=900):
    user_id_list = get_batch(get_full_user_ids(),length)
    def get_real(dataset):
        return dataset[map(lambda user_id:'y_#%d'%user_id,user_id_list)].sum(axis=1)
    def get_predict(dataset):
        return dataset[map(lambda user_id:'y_p#%d'%user_id,user_id_list)].sum(axis=1)
    real = get_real(all_dataset[0])
    predict = pd.DataFrame(map(get_predict,all_dataset)).T

    real = np.array(real).reshape(-1,1)
    predict = np.array(predict)
    return (predict,real)

def get_other_features(day,filter_fuc):
    other_features = pd.DataFrame.from_csv('./features/tensorflow_model/other_feature.csv')
    other_features = filter_fuc(other_features)
    ssd = other_features[['ssd#%d'%day_t for day_t in range(day-1-1,day-1+2)]]
    holiday = other_features[['holiday#%d'%day_t for day_t in range(day-1-1,day-1+2)]]
    festday = other_features[['festday#%d'%day_t for day_t in range(day-1-1,day-1+2)]]
    other_feature = (ssd,holiday,festday)
    other_feature = map(np.array,other_feature)
    return other_feature

def get_x_y_(all_dataset,day):
    (predict,real) = resample_x_y_(all_dataset,1369)
    other_feature = get_other_features(day,lambda x:x.ix[:len(real),:])
    return (predict,real,other_feature)
    
  
class tf_model:
    def __init__(self,day,learning_rate = 1e-2):
        self.graph = tf.Graph()
        self.model_path = './models/tf_model/'+'%d/tf.tfmodel'%day
        with self.graph.as_default():
            self.x_predict = tf.placeholder("float", [None,_feature_length])
            self.y_ = tf.placeholder("float", [None,1])
            self.x_ssd = tf.placeholder("float", [None,3])
            self.x_hoilday = tf.placeholder("float", [None,3])
            self.x_festday = tf.placeholder("float", [None,3])
            o2 = self.get_conv_layer(self.x_ssd,'ssd')
            o3 = self.get_conv_layer(self.x_hoilday,'hoilday')
            o4 = self.get_conv_layer(self.x_festday,'festday')
            o_c = tf.concat([o2,o3,o4],1)
            #layer fc 1
            w_fc1 = tf.get_variable('all/w_fc1', [3,4],
                                      initializer=tf.random_normal_initializer())
            b_fc1 = tf.get_variable('all/b_fc1', [4,],
                                    initializer=tf.random_normal_initializer())
            
            fco_1 = tf.nn.sigmoid(tf.matmul(o_c, w_fc1) + b_fc1)
            #layer fc 2
            w_fc2 = tf.get_variable('all/w_fc2', [4,_feature_length],
                                      initializer=tf.random_normal_initializer())
            b_fc2 = tf.get_variable('all/b_fc2', [_feature_length,],
                                    initializer=tf.random_normal_initializer())
            #zoom
            w_zoom = tf.get_variable('all/w_zoom', [3,1],
                                      initializer=tf.random_normal_initializer())
            b_zoom = tf.get_variable('all/b_zoom', [1,],
                                    initializer=tf.random_normal_initializer())
            #0.95~1.05
            self.zoom = tf.nn.sigmoid(tf.matmul(o_c, w_zoom) + b_zoom)*0.1+0.95
            
            self.percent = tf.nn.softmax(tf.matmul(fco_1, w_fc2) + b_fc2)
            self.y_p = tf.reduce_sum(self.x_predict*self.percent*self.zoom,1)
            self.y_p = tf.reshape(self.y_p,[-1,1])
            self.regularizers = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.error_rate = tf.reduce_mean(tf.abs(self.y_-self.y_p))
            self.mse = self.error_rate + 1e-5*self.regularizers
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.mse)
            self.sess = tf.Session(graph = self.graph)
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
    def save_model(self):
        return self.saver.save(self.sess,self.model_path)
        
    def load_model(self):
        tf.reset_default_graph()
        self.saver.restore(self.sess,self.model_path)
        
    def train(self,x_,y_,other_features):
        (ssd,hoilday,festday) = other_features
        return self.sess.run(self.train_step,feed_dict={
                self.x_predict:x_,self.y_:y_,
                self.x_ssd:ssd,self.x_hoilday:hoilday,
                self.x_festday:festday})
        
    def get_accuracy(self,x_,y_,other_features):
        (ssd,hoilday,festday) = other_features
        return self.sess.run(self.error_rate,feed_dict={
                self.x_predict:x_,self.y_:y_,
                self.x_ssd:ssd,self.x_hoilday:hoilday,
                self.x_festday:festday})
        
    def get_percent(self,x_,other_features):
        (ssd,hoilday,festday) = other_features
        return self.sess.run(self.percent,feed_dict={
                self.x_predict:x_,
                self.x_ssd:ssd,self.x_hoilday:hoilday,
                self.x_festday:festday})
        
    def predict(self,x_,other_features):
        (ssd,hoilday,festday) = other_features
        return self.sess.run(self.y_p,feed_dict={
                self.x_predict:x_,
                self.x_ssd:ssd,self.x_hoilday:hoilday,
                self.x_festday:festday})
    @staticmethod
    def parametric_relu(names,_x):
      alphas = tf.get_variable(names+'/alpha', _x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32)
      pos = tf.nn.relu(_x)
      neg = alphas * (_x - abs(_x)) * 0.5
      return pos + neg
    @staticmethod
    def get_conv_layer(input_layer,names = 'bad_weather'):
        # data shape is "[batch, in_width, in_channels]"
        # filter shape is "[filter_width, in_channels, out_channels]"
        x_other_features_r = tf.reshape(input_layer,[-1,3,1])
        #layer 1
        w_1 = tf.get_variable(names+'/w1', [3,1,1],
                                  initializer=tf.random_normal_initializer())
        b_1 = tf.get_variable(names+'/b1', [1,],
                                  initializer=tf.random_normal_initializer())
        conv_1 = tf.nn.conv1d(x_other_features_r, w_1, stride=1, padding='SAME') + b_1
        prelu_1 = tf_model.parametric_relu(names+'1/',conv_1)
        #layer 2
        w_2 = tf.get_variable(names+'/w2', [3,1,1],
                                  initializer=tf.random_normal_initializer())
        b_2 = tf.get_variable(names+'/b2', [1,],
                                  initializer=tf.random_normal_initializer())
        conv_2 = tf.nn.conv1d(prelu_1, w_2, stride=1, padding='SAME') + b_2
        prelu_2 = tf_model.parametric_relu(names+'2/',conv_2)
        prelu_2_l = tf.reshape(prelu_2,[-1,3*1])
        #layer fc 1
        w_fc1 = tf.get_variable(names+'/w_fc1', [3*1,1],
                                  initializer=tf.random_normal_initializer())
        b_fc1 = tf.get_variable(names+'/b_fc1', [1,],
                                initializer=tf.random_normal_initializer())
        fco_1 = tf.nn.sigmoid(tf.matmul(prelu_2_l, w_fc1) + b_fc1)
        
        return fco_1

def train_tf_once(day):
    all_dataset = get_dataset(day)
    #all_dataset = map(filter_spring_festval,all_dataset)
    #all_dataset = map(filter_sept,all_dataset)
    all_dataset = map(lambda dataset:dataset.dropna(),all_dataset)
    trainer = tf_model(day,learning_rate=1e-3)
    if os.path.exists('./models/tf_model/'+'%d/tf.tfmodel.meta'%day):
        print 'load_model!'
        trainer.load_model()
    (x_,y_,other_features) = get_x_y_(all_dataset,day)
    for time_t in range(9001):
        x_,y_ = resample_x_y_(all_dataset,1300)
        if time_t %100 == 0:
            print day,' ',time_t,':', trainer.get_accuracy(x_,y_,other_features)
            trainer.save_model()
            sys.stdout.flush()
        trainer.train(x_,y_,other_features)
    
class tf_percent_model:
    def __init__(self,day,learning_rate = 1e-2):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x_predict = tf.placeholder("float", [None,_feature_length])
            self.y_ = tf.placeholder("float", [None,1])
            #layer fc 1
            w_1 = tf.get_variable('all/w_1', [_feature_length,],
                                      initializer=tf.random_normal_initializer())
            #zoom layer
            w_zoom = tf.get_variable('all/w_zoom', [1,],
                                      initializer=tf.random_normal_initializer())
            #0.8~1.2
            self.zoom = tf.nn.sigmoid(w_zoom)*0.4+0.8
            self.percent = tf.nn.softmax(w_1)*self.zoom
            self.y_p = tf.reduce_sum(self.x_predict*self.percent,1)
            self.y_p = tf.reshape(self.y_p,[-1,1])
            self.error_rate = tf.reduce_mean(tf.abs(self.y_-self.y_p)/self.y_)
            self.mse = tf.reduce_mean(tf.abs(self.y_-self.y_p))
            #self.mse = self.error_rate
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = self.optimizer.minimize(self.mse)
            self.sess = tf.Session(graph = self.graph)
            self.sess.run(tf.global_variables_initializer())
        
    def re_init(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        
    def train(self,x_,y_):
        return self.sess.run(self.train_step,feed_dict={
                self.x_predict:x_,self.y_:y_})
        
    def get_accuracy(self,x_,y_):
        return self.sess.run(self.error_rate,feed_dict={
                self.x_predict:x_,self.y_:y_})
    
    def get_mse(self,x_,y_):
        return self.sess.run(self.mse,feed_dict={
                self.x_predict:x_,self.y_:y_})
    
    def get_percent(self):
        return self.sess.run(self.percent)
    
    def predict(self,x_):
        return self.sess.run(self.y_p,feed_dict={
                self.x_predict:x_})
    @staticmethod
    def resample_x_y_(all_dataset,user_id):
        real = all_dataset[0]['y_#%d'%user_id].dropna()
        predict = pd.DataFrame(map(lambda dataset:dataset['y_p#%d'%user_id],all_dataset)).T.dropna()
    
        real = np.array(real).reshape(-1,1)
        predict = np.array(predict)
        return (predict,real)
        
def train_tf_once_percent(day):
    all_dataset = get_dataset(day)
    all_dataset_sept = map(filter_sept,all_dataset)
    #all_dataset = map(filter_spring_festval,all_dataset)
    #all_dataset = map(filter_sept,all_dataset)
    #all_dataset = map(lambda dataset:dataset.dropna(),all_dataset)
    rst = pd.DataFrame()
    error_rate = []
    trainer = tf_percent_model(day,learning_rate=1e-1)
    for user_id in get_full_user_ids():
        trainer.re_init()
        (x_,y_) = trainer.resample_x_y_(all_dataset_sept,user_id)
        x_ = x_[:-day]
        #(x_,y_) = trainer.resample_x_y_(all_dataset,user_id)
        #x_ = x_[:-day]
        #remove error point
        #mask = (y_ > y_[-30:].min()/4)&(y_ < y_[-30:].max()*4)
        #make sure the final 30 point is in the dataset
        #mask[-30:,:] = True
        #y_ = y_[mask].reshape(-1,1)
        #x_ = x_[mask.repeat(_feature_length).reshape(-1,_feature_length)].reshape(-1,_feature_length)
        round_count = 501
        for time_t in range(round_count):
            if time_t == round_count-1:
                print (day,user_id),' ',time_t,':', trainer.get_mse(x_,y_)
                sys.stdout.flush()
            trainer.train(x_,y_)
        mse = trainer.get_mse(x_,y_)
        '''
        #%% retrain
        if mse >1e3:
            x_,y_ = trainer.resample_x_y_(all_dataset_sept,user_id)
            x_ = x_[:-day]
            #x_ = x_[-30:]
            #y_ = y_[-30:]
            trainer.re_init()
            round_count = 501
            for time_t in range(round_count):
                if time_t == round_count-1:
                    print 'retrain:',(day,user_id),' ',time_t,':', trainer.get_mse(x_,y_)
                    sys.stdout.flush()
                trainer.train(x_,y_)
            mse = trainer.get_mse(x_,y_)
        '''
        #%% retrain
        rst[user_id] = trainer.get_percent()
        error_rate.append(mse)
    rst = rst.T
    rst.columns = map(lambda x:'percent#%d'%x,range(_feature_length))
    rst['mse'] = error_rate
    rst.to_csv('./features/tensorflow_model/percent_model/%d.csv'%day)
    return  rst
    #return (x_,y_,trainer.predict(x_),trainer.get_percent(),)
    
def predict_tf_one_shop(day,shop_id,start_date = '2016-10-1'):
    all_dataset = get_dataset(day)
    all_dataset = map(lambda x:x.ix[start_date:start_date],all_dataset)
    y_p_features = map(lambda user_id:tf_percent_model.resample_x_y_(all_dataset,user_id)[0].reshape(-1),[shop_id,])
    percent = pd.DataFrame.from_csv('./features/tensorflow_model/percent_model/%d.csv'%day)

    percent = percent.ix[shop_id]
    percent = percent[map(lambda x:'percent#%d'%x,range(_feature_length))]
   
    val =  (percent*y_p_features[0]).sum()
    print (day,shop_id,val)
    return val
def predict_tf_one_shop_all(shop_id,start_date = '2016-10-1'):
    p = m_Pool(30)
    for day in range(1,32):
        p.apply_async(predict_tf_one_shop,(day,1416,start_date))
    p.close()
    p.join()
                      
def predict_tf_once(day,start_date = '2016-10-1'):
    all_dataset = get_dataset(day)
    all_dataset = map(lambda x:x.ix[start_date:start_date],all_dataset)
    y_p_features = map(lambda user_id:tf_percent_model.resample_x_y_(all_dataset,user_id)[0].reshape(-1),get_full_user_ids())
    y_p_features_df = pd.DataFrame(y_p_features,index = get_full_user_ids())
    percent = pd.DataFrame.from_csv('./features/tensorflow_model/percent_model/%d.csv'%day)
    #percent = pd.DataFrame.from_csv('./features/tensorflow_model/percent_model/%d.csv'%2)
    #%%
    percent = percent[map(lambda x:'percent#%d'%x,range(_feature_length))]
    t = pd.DataFrame(index = percent.index)
    t[pd.Timestamp(start_date)+pd.Timedelta('%dd'%(day-1))] = (np.array(y_p_features_df)*percent).sum(axis=1)
    t = t.T
    t.to_csv('./result/predict_part/%d.csv'%day)
    real = int(np.round((np.array(y_p_features_df)*percent).sum().sum()))
    print (day,real)
    return (day,real)
def predict_tf_all(path = None):
    result_list = []
    p = m_Pool(31)
    result_list = p.map(predict_tf_once,range(1,32))
    p.close()
    p.join()
    print 'writing...'
    result_df = pd.DataFrame(index = range(1))
    for day,result in result_list:
        day_s = str(day)
        if len(day_s)<=1:
            day_s = '0'+day_s
        result_df['201610'+day_s] = result
    result_df = result_df.T
    result_df.columns = ['predict_power_consumption']
    if path == None:
        date = str(pd.Timestamp(time.ctime())).replace(' ','_').replace(':','_')
        path = './result/'+date+'.csv'
    result_df.to_csv(path,index_label='predict_date')
    
    l = map(lambda day:pd.DataFrame.from_csv('./result/predict_part/%d.csv'%day),range(1,32))
    t = pd.concat(l)
    t.to_csv('./result/predict_part/'+date+'.csv')

def get_7d_all(user_id):
    data_paths = [
                     './features/tensorflow_model/np_tiny_7_model/',
                     './features/tensorflow_model/np_tiny_7_exp_model/',
                     
                     './features/tensorflow_model/np_tiny_7_filtered_model/',
                     './features/tensorflow_model/np_tiny_7_filtered_exp_model/',
                     
                     './features/tensorflow_model/np_tiny_7_f2_model/',
                     './features/tensorflow_model/np_tiny_7_f2_exp_model/',
                 ]
    def get_predict_val(dataset):
        return dataset['y_p#%d'%user_id][-1]
    def get_mid_val(day):
        all_dataset = map(lambda path:pd.DataFrame.from_csv(path+'%d.csv'%day),data_paths)
        val_list = map(get_predict_val,all_dataset)
        val = np.median(val_list)
        print (user_id,day,val)
        return val
    return map(get_mid_val,range(1,32))
def fix_shop(path,fix_range = 2):
    df  = pd.DataFrame.from_csv(path)
    shop_df = pd.DataFrame.from_csv('./t_shop.csv')
    his_max_last_year = shop_df.ix["2015-10-1":"2015-10-31"].max()
    his_min_last_year = shop_df.ix["2015-10-1":"2015-10-31"].min()
    his_max = pd.DataFrame.from_csv('./t_shop.csv')[-40:].max()
    his_max = pd.DataFrame([his_max_last_year,his_max]).max()
    his_min = pd.DataFrame.from_csv('./t_shop.csv')[-40:].min()
    his_min = pd.DataFrame([his_min_last_year,his_min]).min()
    now_mid = df.median()
    for user_id in get_full_user_ids():
        user_his_max = his_max['%d'%user_id]
        user_his_min = his_min['%d'%user_id]
        user_now_mid = now_mid['%d'%user_id]
        if user_now_mid>user_his_max*fix_range \
            or user_now_mid < user_his_min/fix_range:
            print 'id:%d now:%d max:%d min :%d'%\
                (user_id,user_now_mid,user_his_max,user_his_min)
            t = get_7d_all(user_id)
            t = map(lambda x:user_his_max*2 if x> user_his_max*2 else x,t)
            t = map(lambda x:1 if x<1 else x,t)
            print t
            df['%d'%user_id] = t
            df.to_csv(path)
        else:
            print '%d pass!'%user_id
    #return df
if __name__ == "__main__":
    '''
    for day in range(1,31):
        train_tf_once(int(day))
    '''
    '''
    p = m_Pool(5)
    for day in range(1,32):
        print day
        p.apply_async(train_tf_once,args=(day,))
        time.sleep(10)
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    '''
    
    def create_path():
        for path in _save_paths:
            if not os.path.exists(path):
                os.mkdir(path)
    create_path()
    def rebuild_predict_feature_all_mt(pos):
        func,path = zip(_create_feature_funcs,_save_paths)[pos]
        print path+':'
        rebuild_predict_feature_all(create_feature_func = func,save_path = path)
    p = m_Pool(3)
    for pos in range(_feature_length):
        print pos
        p.apply_async(rebuild_predict_feature_all_mt,args=(pos,))
        #time.sleep(10)
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    
    
    p = m_Pool(7)
    for day in range(1,32):
        print day
        p.apply_async(train_tf_once_percent,args=(day,))
        #time.sleep(10)
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    
    predict_tf_all('./1.csv')
