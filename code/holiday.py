# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:40:34 2017

@author: WZ5040
"""

import json
import urllib2
import pandas as pd
import calendar
def get_holiday_mask(month = '201610'):
    mask_url = urllib2.urlopen('http://www.easybots.cn/api/holiday.php?m=%s'%month)
    holiday_mask = json.loads(mask_url.read())
    holiday_mask = holiday_mask[holiday_mask.keys()[0]]
    holiday_mask = dict(map(lambda key:(int(key),int(holiday_mask[key])),holiday_mask))
    return holiday_mask
def get_holiday_df(year,month):
    monthRange = calendar.monthrange(year,month)[-1]
    mask_month = "%s%s"%(year,month)
    if month < 10:
        mask_month = "%s0%s"%(year,month)
    mask = get_holiday_mask(mask_month)
    a = pd.DataFrame(index = pd.date_range('%s-%s-1'%(year,month), periods=monthRange, freq='D'))
    index = pd.Series(a.index)
    mask_df = index.apply(lambda x:mask[x.day] if x.day in mask else 0)
    mask_df.index = index
    a['holiday'] = (mask_df == 1).astype('int')
    a['festday'] = (mask_df == 2).astype('int')
    return a
def make_month_features(holiday_df):
    df_list = []
    for cols in ['holiday','festday']:
        new_df = pd.DataFrame(index = holiday_df.index)
        holi = holiday_df[cols].copy()
        holi_new = holi.copy()
        #predict 30 days and 30days for features
        for d in range(30):
            holi_new.index += pd.Timedelta('1D')
            new_df['%s#-%d'%(cols,d+1)] = holi_new
        #create 31 models
        for d in range(31+3):
            #predict 31 days + 3days
            new_df['%s#%d'%(cols,d)] = holi
            holi.index -= pd.Timedelta('1D')
        new_df = new_df[map(lambda day:'%s#%d'%(cols,day),range(-30,30+3))]
        new_df = new_df.ix['2015-1-1':'2016-12-31']
        df_list.append(new_df.dropna())
    return df_list
    
if __name__ == '__main_1_':
    holiday_list = []
    holiday_list.append(get_holiday_df(2014,12))
    for year in range(2015,2017):
        for month in range(1,13):
            print '%d-%d'%(year,month)
            holiday_list.append(get_holiday_df(year,month))
    holiday_list.append(get_holiday_df(2017,1))
    holiday_list.append(get_holiday_df(2017,2))
    holiday_df = pd.concat(holiday_list)
    holiday_df.to_csv('holiday_mask.csv')
    holiday_df_all = make_month_features(holiday_df)
    holiday_df_all[0].to_csv('holi_11.csv')
    holiday_df_all[1].to_csv('feat_11.csv')