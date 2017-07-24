#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:58:48 2017

@author: boweiy
"""

import pandas as pd
import numpy as np
from fbprophet import Prophet
from multiprocessing import Pool as m_Pool
from preprocess import get_dataset
import os

def predict_using_prophet(user_df):
    user_df = user_df.resample('!D').mean().fillna(1).astype('int')
    user_id = user_df.user_id[0]
    print user_id
    df = pd.DataFrame()
    df['ds'] = user_df.index
    df['y'] = np.array(user_df[['power_consumption']])
    df['y'] = np.log(df['y'])
    df.columns = ['ds','y']
    m = Prophet()
    #m = Prophet(interval_width=0.95,mcmc_samples=500)
    
    m.fit(df)
    future = m.make_future_dataframe(periods=31+3)
    forecast = m.predict(future)
    img_f = m.plot_components(forecast)
    img_f.savefig('./fig/forecast/%d.png'%user_df.user_id[0])
    img = m.plot(forecast)
    img.savefig('./fig/%d.png'%user_df.user_id[0])
    forecast.to_csv('./features/prophet/%d.csv'%user_df.user_id[0])
    return forecast
def check(dataset,user_id):
    print user_id
    if not os.path.exists('./features/%d.csv'%user_id):
        user_df = dataset[dataset.user_id == user_id]
        assert user_df.power_consumption.sum() == len(user_df)
        return False
    a = pd.DataFrame.from_csv('./features/%d.csv'%user_id)
    assert a.ds.iloc[-1] == '2016-09-30'
    return True
    #return a.ds.iloc[-1] == '2016-09-30'
    
if __name__ == '__main__':
    dataset = get_dataset()
    '''for user_id in set(dataset.user_id):
        predict_using_prophet(dataset[dataset.user_id == user_id])
    '''    
    
    p = m_Pool(64)
    for arg in set(dataset.user_id):
        arg_df = dataset[dataset.user_id == arg]
        p.apply_async(predict_using_prophet,args=(arg_df,))
        #p.apply_async(predict_using_prophet, args=(arg,))
    
    print 'Waiting for all subprocesses done...'
    p.close()
    p.join()
    
    '''
    all_one_list = []
    for user_id in set(dataset.user_id):
        if not check(dataset,user_id):
            all_one_list.append(user_id)
    '''
    