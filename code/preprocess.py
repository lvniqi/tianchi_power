#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:36:18 2017

@author: boweiy
"""
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from multiprocessing import Pool as m_Pool

from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
DATASET_PATH = './dataset/Tianchi_power.csv'
DATASET_SEPT_PATH = './dataset/Tianchi_power_sept.csv'

FEATURE_PATH = './features/'
DAY_FEATURE_PATH = FEATURE_PATH+'day_model/'
DAY_FEATURE_ALL_PATH = FEATURE_PATH+'day_model_all/'

MONTH_FEATURE_PATH = FEATURE_PATH+'month_model/'
HOLIDAY_MONTH_FEATURE_PATH = FEATURE_PATH+'holiday_month_model/'
PROPHET_HOLIDAY_MONTH_FEATURE_PATH = FEATURE_PATH+'prophet_holiday_month_model/'

HOLIDAY_PATH = FEATURE_PATH+'holiday/holi_feat_10.csv'
FEST_PATH = FEATURE_PATH+'holiday/fest_feat_10.csv'
WEATHER_PATH = FEATURE_PATH+'weather/weather.csv'
PROPHET_PATH = FEATURE_PATH+'prophet/'
USER_TYPE_PATH = FEATURE_PATH+'user_type/user_type.csv'
HISTORY_PATH = FEATURE_PATH+'history/'

WEATHER_COLUMNS = ['Temp','bad_weather','ssd']
#PROPHET_COLUMNS = ['trend','weekly','yearly','seasonal']
#PROPHET_COLUMNS = ['weekly','yearly']
PROPHET_COLUMNS = ['trend','yearly']
HISTORY_COLUMNS = ['weekly_median','weekly_max','weekly_min',\
                   'weekly2_median','monthly_median']

MODELS_PATH = './models/'
DAY_MODELS_PATH = MODELS_PATH +'day_model/'
DAY_FILTERED_MODELS_PATH = MODELS_PATH +'day_filtered_model/'

PROPHET_MODELS_PATH = MODELS_PATH +'prophet_model/'
PROPHET_FILTERED_MODELS_PATH = MODELS_PATH +'prophet_filtered_model/'

PROPHET_EXP_MODELS_PATH = MODELS_PATH +'prophet_exp_model/'
PROPHET_EXP__FILTERED_MODELS_PATH = MODELS_PATH +'prophet_exp_filtered_model/'

EXTERA_HOLI_PROPHET_MODELS_PATH = MODELS_PATH +'extera_holi_prophet_model/'
EXTERA_HOLI_PROPHET_FILTERED_MODELS_PATH = MODELS_PATH +'extera_holi_prophet_filtered_model/'
EXTERA_HOLI_PROPHET_F2_MODELS_PATH = MODELS_PATH +'extera_holi_prophet_f2_model/'

EXTERA_HOLI_PROPHET_EXP_MODELS_PATH = MODELS_PATH +'extera_holi_prophet_exp_model/'
EXTERA_HOLI_PROPHET_EXP_FILTERED_MODELS_PATH = MODELS_PATH +'extera_holi_prophet_exp_filtered_model/'
EXTERA_HOLI_PROPHET_EXP_F2_MODELS_PATH = MODELS_PATH +'extera_holi_prophet_exp_f2_model/'

PROPHET_14DAY_MODELS_PATH = MODELS_PATH +'prophet_14_model/'
PROPHET_14DAY_FILTERED_MODELS_PATH = MODELS_PATH +'prophet_14_filtered_model/'
PROPHET_14DAY_F2_MODELS_PATH = MODELS_PATH +'prophet_14_f2_model/'

PROPHET_14DAY_EXP_MODELS_PATH = MODELS_PATH +'prophet_14_exp_model/'
PROPHET_14DAY_EXP_FILTERED_MODELS_PATH = MODELS_PATH +'prophet_14_exp_filtered_model/'
PROPHET_14DAY_EXP_F2_MODELS_PATH = MODELS_PATH +'prophet_14_exp_f2_model/'

PROPHET_7DAY_MODELS_PATH = MODELS_PATH +'prophet_7_model/'
PROPHET_7DAY_FILTERED_MODELS_PATH = MODELS_PATH +'prophet_7_filtered_model/'
PROPHET_7DAY_F2_MODELS_PATH = MODELS_PATH +'prophet_7_f2_model/'

PROPHET_7DAY_EXP_MODELS_PATH = MODELS_PATH +'prophet_7_exp_model/'
PROPHET_7DAY_EXP_FILTERED_MODELS_PATH = MODELS_PATH +'prophet_7_exp_filtered_model/'
PROPHET_7DAY_EXP_F2_MODELS_PATH = MODELS_PATH +'prophet_7_exp_f2_model/'

TINY_7_MODELS_PATH = MODELS_PATH +'tiny_7_model/'
TINY_7_FILTERED_MODELS_PATH = MODELS_PATH +'tiny_7_filtered_model/'
TINY_7_F2_MODELS_PATH = MODELS_PATH +'tiny_7_f2_model/'

TINY_7_EXP_MODELS_PATH = MODELS_PATH +'tiny_7_exp_model/'
TINY_7_EXP_FILTERED_MODELS_PATH = MODELS_PATH +'tiny_7_exp_filtered_model/'
TINY_7_EXP_F2_MODELS_PATH = MODELS_PATH +'tiny_7_exp_f2_model/'
#%% np_tiny7
NP_TINY_7_MODELS_PATH = MODELS_PATH +'np_tiny_7_model/'
NP_TINY_7_FILTERED_MODELS_PATH = MODELS_PATH +'np_tiny_7_filtered_model/'
NP_TINY_7_F2_MODELS_PATH = MODELS_PATH +'np_tiny_7_f2_model/'

NP_TINY_7_EXP_MODELS_PATH = MODELS_PATH +'np_tiny_7_exp_model/'
NP_TINY_7_EXP_FILTERED_MODELS_PATH = MODELS_PATH +'np_tiny_7_exp_filtered_model/'
NP_TINY_7_EXP_F2_MODELS_PATH = MODELS_PATH +'np_tiny_7_exp_f2_model/'
#%% np_tiny7

PRE_PREDICT_MODELS_PATH =  MODELS_PATH +'pre_predict_model/'
PRE_PREDICT_FILTERED_MODELS_PATH =  MODELS_PATH +'pre_predict_filtered_model/'
PREDICT_MODELS_PATH =  MODELS_PATH +'predict_model/'

USER_MODELS_PATH = MODELS_PATH +'user_model/'
USER_EXP_MODELS_PATH = MODELS_PATH +'user_exp_model/'

_empty_user_df = pd.Series.from_csv('./dataset/empty_user.csv')
_user_type_df = pd.Series.from_csv(USER_TYPE_PATH)
_power_range_df = pd.Series.from_csv('./dataset/power_range.csv').apply(np.log)
_exp_exp_ratio_df = pd.Series.from_csv('./dataset/exp_noexp_ratio.csv')
#%% get method
def get_dataset():
    train_set = pd.DataFrame.from_csv(DATASET_PATH)
    eval_set = pd.DataFrame.from_csv(DATASET_SEPT_PATH)
    return pd.concat([train_set,eval_set])

def get_user_id_list():
    return list(_empty_user_df.index)
    
def get_empty_user_ids():
    return list(_empty_user_df[_empty_user_df ==True].index)
    
def get_full_user_ids():
    return list(_empty_user_df[_empty_user_df ==False].index)
    
def get_big_user_ids(th = 7):
    return list(_power_range_df[get_full_user_ids()][_power_range_df>th].index)

def get_small_user_ids(th = 7):
    return list(_power_range_df[get_full_user_ids()][_power_range_df<=th].index)
    
def get_exp_ratio_df():
    return _exp_exp_ratio_df
    
def get_user_id_by_path(df_path):
    return int(df_path[df_path.rindex('/')+1:df_path.rindex('.')])
    
def get_file_path(floder_path):
    return sorted(map(lambda dir_t:floder_path+dir_t,os.listdir(floder_path)))

def get_day_df_path(day):
    floder_path = DAY_FEATURE_PATH+'%d/'%day
    return get_file_path(floder_path)
def get_day_df(day):    
    df_path = get_day_df_path(day)
    all_model =  pd.concat(map(pd.DataFrame.from_csv,df_path)).dropna()
    #all_model.to_csv(DAY_FEATURE_ALL_PATH+'%d'%day)
    return all_model

    
def get_month_df_path():
    return get_file_path(MONTH_FEATURE_PATH)

def get_holiday_month_df_path():
    return get_file_path(HOLIDAY_MONTH_FEATURE_PATH)
    
def get_prophet_holiday_month_df_path():
    return get_file_path(PROPHET_HOLIDAY_MONTH_FEATURE_PATH)
    
def get_history_path():
    return get_file_path(HISTORY_PATH)

def get_prophet_columns():
    return PROPHET_COLUMNS

def get_weather_columns():
    return WEATHER_COLUMNS

def get_scaled_user():
    dataset = get_dataset()
    new_df = pd.DataFrame(index=set(dataset.index))
    new_df = new_df.sort_index()
    for user_id in get_user_id_list():
        #print user_id
        if not check_empty(user_id):
            new_df[user_id] = dataset[dataset.user_id == user_id].power_consumption
    new_df_log = new_df.apply(np.log)
    new_df_log_scaled = preprocessing.MinMaxScaler().fit_transform(new_df_log.ix[60:,:].dropna())
    return pd.DataFrame(new_df_log_scaled,columns = new_df_log.columns)
    
def classify_user():
    new_df_log_scaled = get_scaled_user()
    c = DBSCAN(eps=90,min_samples=50,metric='manhattan').fit(new_df_log_scaled.T)
    pd.value_counts(c.labels_)
    d = c.labels_
    types = pd.DataFrame(d,index=new_df_log_scaled.columns)[0]
    types[types == -1] = 2
    return types
    
def get_holiday_df(day):
    import datetime
    holiday_df = pd.DataFrame.from_csv(HOLIDAY_PATH)
    index_t = holiday_df.init_date.apply(lambda x: datetime.datetime.strptime(x[:10], '%Y/%m/%d'))
    holiday_df.pop('init_date')
    holiday_df = holiday_df.set_index(index_t)
    holiday_df.index += pd.Timedelta('%dD'%(30+(day-1)))
    #holiday_df = holiday_df.ix[:,day:30+day]
    holiday_df.columns = map(lambda x:'festday#%d'%x,range(-30-(day-1),31-(day-1)+5))
    return holiday_df

def get_festday_df(day):
    import datetime
    holiday_df = pd.DataFrame.from_csv(FEST_PATH)
    index_t = holiday_df.init_date.apply(lambda x: datetime.datetime.strptime(x[:10], '%Y/%m/%d'))
    holiday_df.pop('init_date')
    holiday_df = holiday_df.set_index(index_t)
    holiday_df.index += pd.Timedelta('%dD'%(30+(day-1)))
    #holiday_df = holiday_df.ix[:,day:30+day]
    holiday_df.columns = map(lambda x:'holiday#%d'%x,range(-30-(day-1),31-(day-1)+5))
    return holiday_df
    
def get_prophet_df(user_id):
    prophet_df = pd.DataFrame.from_csv(PROPHET_PATH+'%d.csv'%user_id)
    prophet_df.index = pd.to_datetime(prophet_df.ds)
    prophet_df = prophet_df[get_prophet_columns()]
    #predict 31 days
    new_df = pd.DataFrame(index = prophet_df.index[31:-3])
    for col in prophet_df.columns:
        t_col = prophet_df[col].copy()
        t_col.index += pd.Timedelta('3D')
        #feature 3 days
        #predict 33 days
        for day in range(-3,31+3):
            new_df[col+'#%d'%day] = t_col
            t_col.index -= pd.Timedelta('1D')
    return new_df.dropna()

def get_history_df(user_id):
    return pd.DataFrame.from_csv(HISTORY_PATH+'%d.csv'%user_id)
    
def get_weather_df():
    weather_df = pd.DataFrame.from_csv(WEATHER_PATH)
    weather_df = weather_df[get_weather_columns()]
    #predict 30 days
    new_df = pd.DataFrame(index = weather_df.index[30:-88-3])
    for col in weather_df.columns:
        t_col = weather_df[col].copy()
        t_col.index += pd.Timedelta('3D')
        #feature 7 days
        #predict 30 days
        for day in range(-30,31+3):
            new_df[col+'#%d'%day] = t_col
            t_col.index -= pd.Timedelta('1D')
    return new_df.dropna()

    
def get_day_all_df(day):
    df = pd.DataFrame.from_csv(DAY_FEATURE_ALL_PATH+'%d.csv'%day)
    return df

def get_month_by_path(path):
    return pd.DataFrame.from_csv(path)

def get_month_by_id(user_id):
    return pd.DataFrame.from_csv(PROPHET_HOLIDAY_MONTH_FEATURE_PATH+'%d.csv'%user_id)
def get_month_all_df():
    
    p = m_Pool(64)
    path_list = get_prophet_holiday_month_df_path()
    all_df_list = p.map(get_month_by_path,path_list)
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    return pd.concat(all_df_list)
    '''
    df = pd.DataFrame.from_csv(FEATURE_PATH+'all.csv')
    return df
    '''
def get_month_big_all_df(th = 7):
    p = m_Pool(64)
    ids = get_big_user_ids(th)
    all_df_list = p.map(get_month_by_id,ids)
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    return pd.concat(all_df_list)
def get_month_small_all_df(th = 7):
    p = m_Pool(64)
    ids = get_small_user_ids(th)
    all_df_list = p.map(get_month_by_id,ids)
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    return pd.concat(all_df_list)
#%% drop na method
def drop_na_df(old_df):
    old_df['y'] = old_df.y.fillna(-1)
    return old_df.dropna()

def drop_na_month_df(old_df):
    new_df = old_df.copy()
    for col in new_df.columns:
        if 'y#' in col and len(col) <= 4:
            new_df[col] = new_df[col].fillna(-1)
    return new_df.dropna().replace(-1,np.nan)
#%%   mearge method
def mearge_holiday_day_df(day):
    all_df_list = []
    holiday_df = get_holiday_df(day)
    for df_path in get_day_df_path(day):
        print df_path
        old_df = pd.DataFrame.from_csv(df_path)
        old_df = drop_na_df(old_df)
        new_df = pd.merge(holiday_df,old_df,left_index=True,right_index=True)
        new_df.insert(0,'user_id',get_user_id_by_path(df_path))
        all_df_list.append(new_df)
    all_df = pd.concat(all_df_list)
    all_df.to_csv(DAY_FEATURE_ALL_PATH+'%d.csv'%day)

def mearge_holiday_month_df(holiday_df,festday_df,f_id,df_path):
    print "pos %d:file %s"%(f_id,df_path)
    user_id = get_user_id_by_path(df_path)
    old_df = pd.DataFrame.from_csv(df_path)
    old_df = drop_na_month_df(old_df)
    new_df = pd.merge(festday_df,old_df,left_index=True,right_index=True)
    new_df = pd.merge(holiday_df,new_df,left_index=True,right_index=True)
    new_df.insert(0,'user_id',user_id)
    new_df.to_csv(HOLIDAY_MONTH_FEATURE_PATH+'%d.csv'%user_id)
def mearge_holiday_month_df_all():
    holiday_df = get_holiday_df(1)
    festday_df = get_festday_df(1)
    p = m_Pool(64)
    for f_id,df_path in enumerate(get_month_df_path()):
        p.apply_async(mearge_holiday_month_df,(holiday_df,festday_df,f_id,df_path,))
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()

def mearge_prophet_holiday_month_df(df_path):
    print "file %s"%df_path
    user_id = get_user_id_by_path(df_path)
    old_df = pd.DataFrame.from_csv(df_path)
    old_df = drop_na_month_df(old_df)
    prophet_df = get_prophet_df(user_id)
    weather_df = get_weather_df()
    #history_df = get_history_df(user_id)
    #new_df = pd.merge(history_df,old_df,left_index=True,right_index=True)
    new_df = old_df
    new_df = pd.merge(prophet_df,new_df,left_index=True,right_index=True)
    new_df = pd.merge(weather_df,new_df,left_index=True,right_index=True)
    new_df.pop('user_id')
    #add his
    for his_day in [-28,-21,-14,-7]:
        power_column = ['power#%d'%(his_day+day_t) for day_t in range(0,7)]
        power_t = new_df[power_column]
        new_df.insert(0,'max7_power#%d'%(his_day),power_t.max(axis=1))
        new_df.insert(0,'min7_power#%d'%(his_day),power_t.min(axis=1))
        new_df.insert(0,'std7_power#%d'%(his_day),power_t.std(axis=1))
        new_df.insert(0,'mean7_power#%d'%(his_day),power_t.mean(axis=1))
        holi_t = new_df[['festday#%d'%(his_day+day_t) for day_t in range(0,7)]]
        new_df.insert(0,'mean7_holiday#%d'%(his_day),holi_t.mean(axis=1))
    new_df.insert(0,'user_id',user_id)
    my_type = _user_type_df[user_id]
    for user_type in  sorted(set(_user_type_df)):
        if my_type == user_type:
            new_df.insert(user_type+1,'user_type#%d'%user_type,1)
        else:
            new_df.insert(user_type+1,'user_type#%d'%user_type,0)
    new_df.to_csv(PROPHET_HOLIDAY_MONTH_FEATURE_PATH+'%d.csv'%user_id)
    return new_df
def mearge_prophet_holiday_month_df_all():
        
    all_df_list = []
    p = m_Pool(64)
    path_list = get_holiday_month_df_path()
    all_df_list = p.map(mearge_prophet_holiday_month_df,path_list)
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    '''
    all_df = pd.concat(all_df_list)
    all_df.to_csv(FEATURE_PATH+'all.csv')
    '''
    
def mearge_holiday_day_df_all():
    p = m_Pool(64)
    for day in range(1,31):
        p.apply_async(mearge_holiday_day_df,args=(day,))
        #p.apply_async(predict_using_prophet, args=(arg,))
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
#%%   save model method
def save_model(xgb_regressor,day,folder_path):
    xgb_regressor._Booster.save_model(folder_path+'%d.xgbmodel'%day)

save_day_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,DAY_MODELS_PATH)
save_filtered_day_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,DAY_FILTERED_MODELS_PATH)

save_prophet_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_MODELS_PATH)
save_filtered_prophet_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_FILTERED_MODELS_PATH)
save_prophet_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_EXP_MODELS_PATH)
save_filtered_prophet_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_EXP__FILTERED_MODELS_PATH)

save_extera_holi_prophet_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_MODELS_PATH)
save_extera_holi_prophet_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_FILTERED_MODELS_PATH)
save_extera_holi_prophet_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_F2_MODELS_PATH)
save_extera_holi_prophet_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_EXP_MODELS_PATH)
save_extera_holi_prophet_exp_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_EXP_FILTERED_MODELS_PATH)
save_extera_holi_prophet_exp_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_EXP_F2_MODELS_PATH)

save_prophet_14day_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_14DAY_MODELS_PATH)
save_prophet_14day_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_14DAY_FILTERED_MODELS_PATH)
save_prophet_14day_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_14DAY_F2_MODELS_PATH)
save_prophet_14day_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_14DAY_EXP_MODELS_PATH)
save_prophet_14day_exp_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_14DAY_EXP_FILTERED_MODELS_PATH)
save_prophet_14day_exp_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_14DAY_EXP_F2_MODELS_PATH)

save_prophet_7day_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_7DAY_MODELS_PATH)
save_prophet_7day_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_7DAY_FILTERED_MODELS_PATH)
save_prophet_7day_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_7DAY_F2_MODELS_PATH)
save_prophet_7day_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_7DAY_EXP_MODELS_PATH)
save_prophet_7day_exp_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_7DAY_EXP_FILTERED_MODELS_PATH)
save_prophet_7day_exp_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PROPHET_7DAY_EXP_F2_MODELS_PATH)

save_tiny_7_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,TINY_7_MODELS_PATH)
save_tiny_7_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,TINY_7_FILTERED_MODELS_PATH)
save_tiny_7_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,TINY_7_F2_MODELS_PATH)
save_tiny_7_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,TINY_7_EXP_MODELS_PATH)
save_tiny_7_exp_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,TINY_7_EXP_FILTERED_MODELS_PATH)
save_tiny_7_exp_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,TINY_7_EXP_F2_MODELS_PATH)

#%% np_tiny7
save_np_tiny_7_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,NP_TINY_7_MODELS_PATH)
save_np_tiny_7_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,NP_TINY_7_FILTERED_MODELS_PATH)
save_np_tiny_7_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,NP_TINY_7_F2_MODELS_PATH)
save_np_tiny_7_exp_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,NP_TINY_7_EXP_MODELS_PATH)
save_np_tiny_7_exp_filtered_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,NP_TINY_7_EXP_FILTERED_MODELS_PATH)
save_np_tiny_7_exp_f2_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,NP_TINY_7_EXP_F2_MODELS_PATH)

#%% end np_tiny7

save_user_model = lambda xgb_regressor,day,user_id:save_model(xgb_regressor,day,USER_MODELS_PATH+'%d/'%user_id)
save_user_exp_model = lambda xgb_regressor,day,user_id:save_model(xgb_regressor,day,USER_EXP_MODELS_PATH+'%d/'%user_id)


save_pre_predict_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PRE_PREDICT_MODELS_PATH)
save_filtered_pre_predict_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PRE_PREDICT_FILTERED_MODELS_PATH)
save_predict_model = lambda xgb_regressor,day:save_model(xgb_regressor,day,PREDICT_MODELS_PATH)
#%%   load model method
def load_model(xgb_regressor,day,folder_path):
    booster = xgb.Booster()
    booster.load_model(folder_path+'%d.xgbmodel'%day)
    xgb_regressor._Booster = booster

load_day_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,DAY_MODELS_PATH)
load_filtered_day_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,DAY_FILTERED_MODELS_PATH)
load_prophet_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_MODELS_PATH)
load_filtered_prophet_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_FILTERED_MODELS_PATH)
load_prophet_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_EXP_MODELS_PATH)
load_filtered_prophet_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_EXP__FILTERED_MODELS_PATH)

load_extera_holi_prophet_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_MODELS_PATH)
load_extera_holi_prophet_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_FILTERED_MODELS_PATH)
load_extera_holi_prophet_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_F2_MODELS_PATH)
load_extera_holi_prophet_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_EXP_MODELS_PATH)
load_extera_holi_prophet_exp_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_EXP_FILTERED_MODELS_PATH)
load_extera_holi_prophet_exp_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,EXTERA_HOLI_PROPHET_EXP_F2_MODELS_PATH)

load_prophet_14day_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_14DAY_MODELS_PATH)
load_prophet_14day_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_14DAY_FILTERED_MODELS_PATH)
load_prophet_14day_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_14DAY_F2_MODELS_PATH)
load_prophet_14day_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_14DAY_EXP_MODELS_PATH)
load_prophet_14day_exp_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_14DAY_EXP_FILTERED_MODELS_PATH)
load_prophet_14day_exp_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_14DAY_EXP_F2_MODELS_PATH)

load_prophet_7day_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_7DAY_MODELS_PATH)
load_prophet_7day_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_7DAY_FILTERED_MODELS_PATH)
load_prophet_7day_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_7DAY_F2_MODELS_PATH)
load_prophet_7day_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_7DAY_EXP_MODELS_PATH)
load_prophet_7day_exp_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_7DAY_EXP_FILTERED_MODELS_PATH)
load_prophet_7day_exp_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PROPHET_7DAY_EXP_F2_MODELS_PATH)

load_tiny_7_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,TINY_7_MODELS_PATH)
load_tiny_7_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,TINY_7_FILTERED_MODELS_PATH)
load_tiny_7_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,TINY_7_F2_MODELS_PATH)
load_tiny_7_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,TINY_7_EXP_MODELS_PATH)
load_tiny_7_exp_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,TINY_7_EXP_FILTERED_MODELS_PATH)
load_tiny_7_exp_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,TINY_7_EXP_F2_MODELS_PATH)

#%% np_tiny7
load_np_tiny_7_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,NP_TINY_7_MODELS_PATH)
load_np_tiny_7_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,NP_TINY_7_FILTERED_MODELS_PATH)
load_np_tiny_7_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,NP_TINY_7_F2_MODELS_PATH)
load_np_tiny_7_exp_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,NP_TINY_7_EXP_MODELS_PATH)
load_np_tiny_7_exp_filtered_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,NP_TINY_7_EXP_FILTERED_MODELS_PATH)
load_np_tiny_7_exp_f2_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,NP_TINY_7_EXP_F2_MODELS_PATH)
#%% end np_tiny7

load_user_model = lambda xgb_regressor,day,user_id:load_model(xgb_regressor,day,USER_MODELS_PATH+'%d/'%user_id)
load_user_exp_model = lambda xgb_regressor,day,user_id:load_model(xgb_regressor,day,USER_EXP_MODELS_PATH+'%d/'%user_id)

load_pre_predict_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PRE_PREDICT_MODELS_PATH)
load_filtered_pre_predict_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PRE_PREDICT_FILTERED_MODELS_PATH)
load_predict_model = lambda xgb_regressor,day:load_model(xgb_regressor,day,PREDICT_MODELS_PATH)
#%% others
def filter_empty_user(dataset):
    user_index = sorted(set(dataset.user_id))
    empty_series = pd.Series(index=user_index,dtype='bool')
    for user_id in empty_series.index:
        filtered_power = filter_user_id(dataset,user_id).power_consumption
        power_end = filtered_power[-14:]
        if power_end.mean()<50:
            print 'warming ! user:%d,average:%f'%(user_id,power_end.mean())
            empty_series[user_id] = True
        else:
            empty_series[user_id] = False
    return empty_series
def filter_user_power_range(dataset):
    user_index = sorted(set(dataset.user_id))
    power_series = pd.Series(index=user_index)
    for user_id in power_series.index:
        filtered_power = filter_user_id(dataset,user_id).power_consumption
        power_series[user_id] = filtered_power.median()
    return power_series
def filter_user_id(dataset,user_id):
    return dataset[dataset.user_id == user_id]

def filter_spring_festval(dataset):
    old_index = dataset.index
    pos_mask = ((old_index<'2015-1-1')|(old_index>'2015-3-25'))&\
            ((old_index<'2016-1-1')|(old_index>'2016-3-25'))
    return dataset.ix[pos_mask]

def filter_sept(dataset):
    old_index = dataset.index
    pos_mask = ((old_index>'2015-6-1')&(old_index<'2015-11-30'))|\
            (old_index>'2016-6-1')
    return dataset.ix[pos_mask]
def check_empty(user_id):
    """
    return true if the power of user_id is 1
    """
    return  _empty_user_df[user_id]

def increase_index(last_year):
    """
    将2015年的index替换为2016年的
    """
    last_year.index += pd.Timedelta('365D')
    new_index = pd.Series(last_year.index)
    new_index[new_index >= '2016-2-29'] += pd.Timedelta('1D')
    last_year.index = new_index
    return last_year

def exp_power(x_):
    new_x_ = pd.DataFrame()
    for col in x_:
        if 'power' in col:
            new_x_[col] = x_[col].apply(np.exp)
        else:
            new_x_[col] = x_[col]
    return new_x_
    
def save_month_df(df,user_id):
    df.to_csv(MONTH_FEATURE_PATH+'%d.csv'%user_id)
    
def save_history_df(df,user_id):
    df.to_csv(HISTORY_PATH+'%d.csv'%user_id)
    
def save_day_df(df,day,user_id):
    df.to_csv(DAY_FEATURE_PATH+'%d/%d.csv'%(day+1,user_id))
    
def get_feature_cloumn(columns,day,
                       feature_range = 28,
                       holiday_range = 5,
                       has_extera_holiday = False,
                       has_extera_weather = False,
                       has_holiday = True,
                       has_weather = True,
                       has_prophet = True,
                       has_power = True,
                       has_user_type = True,
                       has_history = False,
                       has_user = False,):
    assert holiday_range%2 == 1
    feature_column = []
    if has_user_type:
        feature_column += ['user_type#%d'%user_t for user_t in sorted(set(_user_type_df))]
    if has_prophet:
        for prophet_column in get_prophet_columns():
            feature_column += [prophet_column+'#%d'%day_t for day_t in range(day-1-2,day-1+3)]
            
    if has_extera_weather:
        end_pos = day-1-2 if day-1-2<0 else 0
        for weather_column in get_weather_columns():
            feature_column += [weather_column+'#%d'%day_t for day_t in range(-feature_range,end_pos)]
    if has_weather:
        for weather_column in get_weather_columns():
            feature_column += [weather_column+'#%d'%day_t for day_t in range(day-1-2,day-1+3)]
    if has_extera_holiday:
        end_pos = day-1-holiday_range/2 if day-1-holiday_range/2<0 else 0
        feature_column += ['holiday#%d'%day_t for day_t in range(-feature_range,end_pos)]
        feature_column += ['festday#%d'%day_t for day_t in range(-feature_range,end_pos)]
                               
    if has_holiday:
        feature_column += ['holiday#%d'%day_t for day_t in range(day-1-holiday_range/2,day-1+holiday_range/2+1)]
        feature_column += ['festday#%d'%day_t for day_t in range(day-1-holiday_range/2,day-1+holiday_range/2+1)]
    if has_power:        
        feature_column += ['power#%d'%day_t for day_t in range(-1,-feature_range-1,-1)]    
    if has_user:
        full_user = get_full_user_ids()
        feature_column += ['user#%d'%user_t for user_t in full_user]
    if has_history:
        feature_column += [his_columns+'#%d'%(day-1) for his_columns in HISTORY_COLUMNS]
            
    return feature_column
    
def get_feature_cloumn_tiny(columns,day,
                       feature_range = 7,
                       holiday_range = 9,
                       has_extera_holiday = False,
                       has_extera_weather = False,
                       has_holiday = True,
                       has_weather = True,
                       has_prophet = True,
                       has_power = True,
                       has_user_type = False,
                       has_history = True,
                       has_user = False,):
    assert holiday_range%2 == 1
    def get_day_his_list():
        day_his_list = []
        day_t = day - 1
        while day_t > -30:
                if(day_t) < -feature_range:
                    day_his_list.append(day_t)
                day_t -= 7
        return day_his_list
    feature_column = []
    day_his_list = get_day_his_list()
    
    
    
    if has_user_type:
        feature_column += ['user_type#%d'%user_t for user_t in sorted(set(_user_type_df))]
    if has_prophet:
        for prophet_column in get_prophet_columns():
            feature_column += [prophet_column+'#%d'%day_t for day_t in range(day-1-2,day-1+3)]
            
    if has_extera_weather:
        end_pos = day-1-2 if day-1-2<0 else 0
        feature_column += ['Temp'+'#%d'%day_t for day_t in range(-feature_range,end_pos)]
    if has_weather:
            feature_column += ['Temp'+'#%d'%day_t for day_t in range(day-1-2,day-1+3)]
    if has_extera_holiday:
        end_pos = day-1-holiday_range/2 if day-1-holiday_range/2<0 else 0
        feature_column += ['holiday#%d'%day_t for day_t in range(-feature_range,end_pos)]
        feature_column += ['festday#%d'%day_t for day_t in range(-feature_range,end_pos)]
                               
    if has_holiday:
        feature_column += ['holiday#%d'%day_t for day_t in range(day-1-holiday_range/2,day-1+holiday_range/2+1)]
        feature_column += ['festday#%d'%day_t for day_t in range(day-1-holiday_range/2,day-1+holiday_range/2+1)]
    if has_power:        
        feature_column += ['power#%d'%day_t for day_t in range(-1,-feature_range-1,-1)]
        feature_column += ['power#%d'%day_t for day_t in day_his_list]
    if has_user:
        full_user = get_full_user_ids()
        feature_column += ['user#%d'%user_t for user_t in full_user]
    if has_history:
        for his_columns in ['mean7','max7','min7','std7']:
            feature_column += [his_columns+'_power#%d'%his_day for his_day in [-28,-21,-14,-7]]
        feature_column += ['mean7_holiday#%d'%his_day for his_day in[-28,-21,-14,-7]]
    return feature_column
    
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    pred_exp = np.exp(preds)
    pred_2_exp = pred_exp*pred_exp
    labels_exp = np.exp(labels)
    grad = (pred_exp-labels_exp)*pred_exp
    hess = (2*pred_2_exp - pred_exp*labels_exp)
    '''
    a = 2*labels
    b = (preds+labels)**2
    grad = 122*np.where((preds - labels)>0,a,-a)/b
    grad = np.where(np.abs(preds - labels)<1e-10,0,grad)
    hess = grad*(-2)/(preds+labels)
    '''
    return grad, hess
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    '''
    pred_exp = np.exp(preds)
    labels_exp = np.exp(labels)
    return 'error', np.sum(np.abs(pred_exp-labels_exp))/np.sum(labels_exp)
    '''
    return 'error', np.sum(np.abs(preds-labels))/np.sum(labels)
    
def crate_pre_train_model(x_,y_):
    (x_train,x_test) = train_test_split(x_,test_size=0.1,random_state=1)
    (y_train,y_test) = train_test_split(y_,test_size=0.1,random_state=1)
    dtrain = xgb.DMatrix( x_train, label=y_train)
    dtest = xgb.DMatrix( x_test, label=y_test)
    evallist  = [(dtrain,'train'),(dtest,'eval')]
    param = {'objective':'reg:linear','max_depth':3 }
    param['nthread'] = 64
    #param['min_child_weight'] = 15
    #param['subsample'] = 1
    #param['num_class'] = 7
    plst = param.items()
    num_round = 5000
    bst = xgb.train( plst, dtrain, num_round,
                    evallist,early_stopping_rounds=100,
                    #obj=logregobj,
                    feval=evalerror
                    )
    return bst

# %% main
if __name__ == '__main__':
    mearge_holiday_month_df_all()
    mearge_prophet_holiday_month_df_all()