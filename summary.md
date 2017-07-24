电量预测AI大赛 总结
====
![title](https://work.alibaba-inc.com/aliwork_tfs/g01_alibaba-inc_com/tfscom/TB1oAMfQFXXXXX1XVXXXXXXXXXX.tfsprivate.jpg)

## 赛题

---
### 赛题背景
此次比赛赛题为企业用电需求预测。主办方提供扬中市高新区1000多家企业的脱敏历史用电量数据，要求参赛者通过模型算法精准预测该地区下一个月的每日总用电量。
### 赛题数据
主办方提供的数据非常简洁(~~*少*~~)，只有日期、企业id及用电量数据。

|record_date|user_id|power_consumption|
|:-------------:|:-------------:|:-----:|
| 20150101     |    1          |   1031|
| ...          | ...           |...    |
| 20161130     |1454           |100000 |

### 评估指标
主办方的具体评分公式不完全公开，总得分为相对误差的函数。
<img src="https://work.alibaba-inc.com/aliwork_tfs/g01_alibaba-inc_com/tfscom/TB1vGo4QFXXXXcuaXXXXXXXXXXX.tfsprivate.png" width = "450" height = "50" alt="score" align=center />
### 结果提交
这次比赛提交的是所有企业的结果总和。

|predict_date|power_consumption|
|:-------------:|:-----:|
| 2016/12/1     |10310000|
| ...          | ...    |
| 201612/31     |10000000 |

说实话我对这样的安排非常的不解，为什么不对每家企业分开提交预测结果呢?
现在这样的提交方式总共只需要28~31个条目，给测答案、作弊或者用玄学脑补数据提供了过多的机会。
以至于[excel大神(~~*真假难辨*~~)](https://tianchi.aliyun.com/competition/new_articleDetail.html?raceId=231602&postsId=2005)轻轻松松秒杀一众模型党。

## 解法介绍
---
### 外部数据处理

#### 节假日数据
感谢[easybots](http://www.easybots.cn/)，我们从其网站上爬取了2014年末至2017年初的节假日及法定假日数据，并在线下做了滑动窗口。代码在[holiday.py](https://github.com/lvniqi/tianchi_power/blob/master/code/holiday.py)中。

#### 天气数据
初赛时使用的是[wunderground](http://www.wunderground.com/)提供的南京禄口国际机场每小时气温及湿度数据，以方便计算人体舒适度。
天气状况使用[weather](http://www.weather.com.cn/)，对坏天气(如暴雨、大雪、暴雪等)进行特别标记后使用。

复赛时官方提供天气数据，遂切换至官方数据。数据包含最低、最高温度、天气，天气状况经过脑补的[weather2val变换](https://github.com/lvniqi/tianchi_power/blob/master/code/weather2val_t.csv)后使用。

### 特征工程(线下部分)

#### 数据清洗
我们把数据清洗一部分放在模型训练之前，一部分放在欠拟合模型中(稍后介绍)。

首先，去掉了前14天均值小于50的商店，因为这些店对于最终预测结果没什么太大影响(见[filter_empty_user](https://github.com/lvniqi/tianchi_power/blob/master/code/preprocess.py#L493))。

而后强制去掉了春节那部分的数据[filter_spring_festval](https://github.com/lvniqi/tianchi_power/blob/master/code/preprocess.py#L515)。因为比较懒，一开始没想到做这个步骤，所以这个是在特征做完，训练之前做的。
#### 特征提取

在线下比赛中，我们尝试使用了各种各样奇奇怪怪的特征，如下所示。
生成代码可见[get_feature_cloumn](https://github.com/lvniqi/tianchi_power/blob/master/code/preprocess.py#L560)。
|特征|解释|
|:-------------:|:-----:|
|user_type#n|对店家使用DBSCAN进行分类后进行onehot编码|
|trend#n|以当天为center，窗口大小为5的趋势数据，使用prophet获得|
|yearly#n|以当天为center，窗口大小为5的年趋势数据，使用prophet获得|
|temp#n|以当天为center，窗口大小为5的温度数据|
|bad_weather#n|以当天为center，窗口大小为5的坏天气数据|
|ssd#n|以当天为center，窗口大小为5的人体舒适度数据|
|holiday#n|以当天为center，窗口大小为5的周末假日数据|
|festday#n|以当天为center，窗口大小为5的法定假日数据|
|power#n|前第n天的电量值，包含前28天数据|

其中，[Prophet](https://github.com/facebookincubator/prophet)要强力推荐下，一个facebook提出的智能化预测工具，能自动检测趋势变化，按年周期组件使用傅里叶级数建模，按周的周期组件使用虚拟变量建模。线下版本的特征提取中使用了Prophet的yearly和trend特征，虽然有可能过拟合的问题，但是这种玄学实在是诱人。

此外，对于一部分模型使用的特征，我们对power和预测值进行了log变换，以减小数据的变化范围。

我们还观察了特征重要性，搞了tiny版本的特征，删去了冗余特征，添加了单周统计特征(min、max、std、mean)，这儿不赘言了，有兴趣可见[get_feature_cloumn_tiny](https://github.com/lvniqi/tianchi_power/blob/master/code/preprocess.py#L605)。

### 模型设计(线下部分)
最终版本的线下模型会对31天分开训练，每天使用了六组特征不同的模型，分别是做过log变换和未做过log变换的28天特征、14天特征、以及tiny版本7天特征(共2*3=6)。
每组模型首先使用了1个3层500棵树的xgboost做清洗。
而后使用2个种不同比例抽取最优秀的样本作为清洗后训练集，再训练2个5至6层1000至1600棵树的xgboost模型。
每组模型的训练方式如下所示，所以共计产生了31*(6*3)个xgboost模型。

<div align=center>
<img src="https://github.com/lvniqi/tianchi_power/blob/master/image/train_xgb_down.png" width = "403" height = "322" alt="train-xgb" align=center />
</div>

### 模型融合(线下部分)
模型融合部分我们使用tensorflow设计了一个线性回归的模型(详细可见[tf_percent_model]())。
``` python

```

**这边模型融合其实是有些问题的，stacking原来是划分数据集的，我们为了尽可能使用数据集并减少计算量，采用的是划分特征的方式。**

**另一方面，我们所有的模型都没有做交叉验证！！！是的，我们的做法非常不靠谱，千万不要学。没有交叉验证意味着我们这么做有极大的可能会过拟合，当然单个模型的棵树和层数事先测试过，但是这个LR显然是不准的....**

### 特征工程(线上部分)
#### 数据清洗
线上的模型仅仅去掉了[最近一周总电量小于100的店家](https://github.com/lvniqi/tianchi_power/blob/master/blob/master/code/get_feature_column_sql.py#L199)，其他清洗放在欠拟合模型中。虽然这个欠拟合模型已经不那么欠拟合了。
#### 特征选择
线上部分由于SQL的限制和对阿里PAI平台不太熟悉的原因，进一步简化了特征，去掉了使用onehot线性回归的那些个特征，甚至连onehot都用得极少。

|特征|解释|
|:-------------:|:-----:|
|temperature_low_n|n天前最低温度|
|temperature_high_n|n天前最高温度|
|weather_val_n|n天前天气值|
|power_n|前第n天的电量值，包含前7天数据及预测当天前四周相关日期电量值|
|mean7_power_n|以n天前为start的一周电量均值|
|max7_power_n|以n天前为start的一周电量最大值|
|min7_power_n|以n天前为start的一周电量最小值|
|std7_power_n|以n天前为start的一周电量标准差|
|dayofweek|周几|
|monthofyear|月份|

### 模型设计+模型融合(线上部分)


