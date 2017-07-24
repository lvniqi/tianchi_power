#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:17:04 2017

@author: boweiy
"""
import pandas as pd
import numpy as np
from multiprocessing import Pool as m_Pool
from preprocess import get_dataset,get_user_id_list,increase_index
from preprocess import save_month_df,save_history_df
from preprocess import check_empty,filter_user_id

def make_features(user_id,user_df):
    """
    构造单天特征
    """
    print 'user_id:', user_id
    power = user_df.power_consumption
    assert power.index[0] == user_df.index[0]
    assert len(user_df.index) == 639
    new_df = pd.DataFrame(index=user_df.index.union(pd.date_range('2016-9-1','2016-9-30')))
    pw_new = power.copy()
    #predict 30 days and 30days for features
    for d in range(60):
        pw_new.index += pd.Timedelta('1D')
        new_df['power#-%d'%(d+1)] = pw_new
    #create 30 models
    for d in range(30):
        #30 days features
        x_ = new_df[new_df.columns[d:30+d]]
        x_['y'] = power
        x_.to_csv('./features/day_model/%d/%d.csv'%(d+1,user_id))
        
    #return x_
def make_month_features(user_id,user_df):
    """
    构造单天特征
    """
    print 'user_id:', user_id
    power = user_df.power_consumption.copy()
    assert power.index[0] == user_df.index[0]
    new_df = pd.DataFrame(index=user_df.index.union(pd.date_range('2016-10-1','2016-10-31')))
    pw_new = power.copy()
    #predict 30 days and 30days for features
    for d in range(30):
        pw_new.index += pd.Timedelta('1D')
        new_df['power#-%d'%(d+1)] = pw_new
    #create 30 models
    for d in range(31):
        #30 days features
        new_df['y#%d'%d] = power
        power.index -= pd.Timedelta('1D')
    save_month_df(new_df,user_id)
    return new_df
    
def make_month_features_all():
    pw_df_list = []
    dataset = get_dataset()
    dataset.power_consumption = dataset.power_consumption.apply(np.log)
    for user_id in get_user_id_list():
        print user_id
        if not check_empty(user_id):
            user_df = filter_user_id(dataset,user_id).resample('1D').mean().fillna(0)
            #add to list
            pw_df_list.append((user_id,user_df))
            #make_features(user_id,user_df)
    
    p = m_Pool(64)
    for arg in pw_df_list:
        #p.apply_async(make_features,args=(arg))
        p.apply_async(make_month_features,args=(arg))
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()

def history_feature(user_df):
    user_pow_last_year = user_df.ix[user_df.index <'2016-1-1']
    user_pow_last_year = increase_index(user_pow_last_year)
    #weekly average
    weekly_roll = user_pow_last_year.rolling(7,center=True)
    weekly_median = weekly_roll.median()
    weekly_max = weekly_roll.max()
    weekly_min = weekly_roll.min()
    #weekly2 average
    weekly2_roll = user_pow_last_year.rolling(14,center=True)
    weekly2_median = weekly2_roll.median()
    #monthly average
    monthly_roll = user_pow_last_year.rolling(30,center=True)
    monthly_median = monthly_roll.median()
    feature_index = user_pow_last_year.index[user_pow_last_year.index<'2016-10-30']
    feature_df = pd.DataFrame(index = feature_index)
    feature_df['weekly_median'] = weekly_median
    feature_df['weekly_max'] = weekly_max
    feature_df['weekly_min'] = weekly_min
    feature_df['weekly2_median'] = weekly2_median
    feature_df['monthly_median'] = monthly_median
    feature_df = feature_df.dropna()
    feature_df = feature_df.apply(np.log)
    return feature_df

def make_history_month_features(user_id,user_df):
    """
    构造单天特征
    """
    print 'user_id:', user_id
    power = user_df.power_consumption.copy()
    feature_df = history_feature(power)
    new_df = pd.DataFrame(index = feature_df.index)
    #create 30 models
    for d in range(30):
        for cols in feature_df:
            #30 days features
            new_df[cols+'#%d'%d] = feature_df[cols]
        feature_df.index -= pd.Timedelta('1D')
    new_df = new_df.dropna()
    save_history_df(new_df.dropna(),user_id)
    return new_df

def make_history_month_features_all():
    pw_df_list = []
    dataset = get_dataset()
    dataset.power_consumption = dataset.power_consumption
    for user_id in get_user_id_list():
        print user_id
        if not check_empty(user_id):
            user_df = filter_user_id(dataset,user_id).resample('1D').mean().fillna(1)
            #add to list
            pw_df_list.append((user_id,user_df))
            #make_features(user_id,user_df)
    
    p = m_Pool(64)
    for arg in pw_df_list:
        p.apply_async(make_history_month_features,args=(arg))
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    
if __name__ == '__main__':
    make_month_features_all()
    #make_history_month_features_all()

    