# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:25:31 2017

@author: WZ5040
"""
import os  
from bs4 import BeautifulSoup  
import re  
import urllib2  
import requests
import xlwt
import pandas as pd 
def html(url = 'http://www.weather.com.cn/weather/101010100.shtml'  ):
    headers = {'User-Agent':'Mozilla/5.0 (X11; U; Linux i686)Gecko/20071127 Firefox/2.0.0.11'}  
    req = urllib2.Request(url,headers=headers)  
    response = urllib2.urlopen(req).read()  
    soup = BeautifulSoup(response, 'html.parser')    
    city=[]  
    city.append(str(soup.title))  
    link= str(city).decode('string_escape')  
    pattern = re.findall(r'<title>(.*?)</title>',link)  
    title = str(pattern)  
    title_list = title[2:146]  
    print (str(title_list).decode('string_escape').decode("utf-8"))     
    href_list =[]  
    for link in soup.find_all(id="hidden_title"):            
        href_list.append(str(link))  
    pattern2 = re.compile(r'<.*? value="(.*?)"/>')  
    for href in href_list:       
        result = re.match(pattern2,href)  
        if result:  
            link_href=result.groups()[0]             
            print "%s " %link_href.decode("utf-8")  
    href_lis=[]  
    for link in soup.find_all('p'):  
        href_lis.append(str(link))  
    pattern3 = re.compile(r'<p class=".*?" title="(.*?)">.*?</p>')  
    result = re.match(pattern3,href_lis[0])  
    if result:  
            link_href=result.groups()[0]             
            print "%s " %link_href.decode("utf-8")  
    href_wind=[]  
    for link in soup.find_all('p'):  
        href_wind.append(str(link))  

#获得某一个月的天气数据
def getListByUrl(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")
    weathers = soup.select("#tool_site")
    title = weathers[1].select("h3")[0].text
    weatherInfors = weathers[1].select("ul")
    weatherList = list()
    for weatherInfor in weatherInfors:
        singleWeather = list()
        for li in weatherInfor.select('li'):
            singleWeather.append(li.text)
        weatherList.append(singleWeather)
    #print(title)
    return weatherList,title

#@par:addressUrl 获得某地区的数据
#@par:excelSavePath  数据的保存地址
def getListByAddress(addressUrl,excelSavePath):
    # url = "http://lishi.tianqi.com/beijing/index.html"
    url = addressUrl
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")
    dates = soup.select(".tqtongji1 ul li a")
    #workbook = xlwt.Workbook(encoding='utf-8')
    data_frame_array = []
    for d in dates[:30]:
        weatherList,title = getListByUrl(d["href"])
        a = pd.DataFrame(weatherList[1:])
        a.columns = weatherList[0]
        data_frame_array.append(a)
        '''booksheet = workbook.add_sheet(title,cell_overwrite_ok=True)
        for i,row in enumerate(weatherList):
            for j,col in enumerate(row):
                booksheet.write(i,j,col)'''
    #workbook.save(excelSavePath)
	all = pd.concat(data_frame_array).sort_values(by=u'日期')
    all.to_csv(excelSavePath,encoding="utf-8")

if __name__ == "__main__":
    citys = tuple(pd.read_csv('./city_name.csv')['0'])
    for city in citys:
        addressName = city
        addresses = BeautifulSoup(requests.get('http://lishi.tianqi.com/').text,"html.parser")
        queryAddress = addresses.find_all('a',text=addressName)
        if len(queryAddress):
            if not os.path.exists('c:/weather'):
                os.makedirs('c:/weather')
            #savePath = "c:/weather/"+unicode(addressName,'utf-8')+".xls"
            savePath = "c:/weather/"+unicode(addressName,'utf-8')+".csv"
            for q in queryAddress:
                getListByAddress(q["href"],savePath)
                print(u"已经天气数据保存到:"+savePath)
        else:
            print("不存在该城市的数据:",city)
if __name__ == "__main__":
    html()