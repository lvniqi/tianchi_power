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

## 外部数据处理

#### 节假日数据
感谢[easybots](http://www.easybots.cn/)，我们从其网站上爬取了2014年末至2017年初的节假日及法定假日数据，并在线下做了滑动窗口。代码在[holiday.py](https://github.com/lvniqi/tianchi_power/blob/master/code/holiday.py)中。

#### 天气数据
初赛时使用的是[wunderground](http://www.wunderground.com/)提供的南京禄口国际机场每小时气温及湿度数据，以方便计算人体舒适度。
天气状况使用[weather](http://www.weather.com.cn/)，对坏天气(如暴雨、大雪、暴雪等)进行特别标记后使用。

复赛时官方提供天气数据，遂切换至官方数据。数据包含最低、最高温度、天气。天气数据是文本的类别型数据，我们根据晴天体感温度偏高，下雪天体感温度偏低，将天气状况映射成数值型数据，映射规则为经过脑补的[weather2val变换](https://github.com/lvniqi/tianchi_power/blob/master/code/weather2val_t.csv)。

## 线下解法介绍
---
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
|temp#n|温度数据|
|bad_weather#n|坏天气数据|
|ssd#n|人体舒适度数据|
|holiday#n|以当天为center，窗口大小为5的周末假日数据|
|festday#n|以当天为center，窗口大小为5的法定假日数据|
|power#n|前第n天的电量值，包含前28天数据|

其中，[Prophet](https://github.com/facebookincubator/prophet)要强力推荐下，一个facebook提出的智能化预测工具，能自动检测趋势变化，按年周期组件使用傅里叶级数建模，按周的周期组件使用虚拟变量建模。线下版本的特征提取中使用了Prophet的yearly和trend特征，虽然有可能过拟合的问题，但是这种玄学实在是诱人。

此外，对于一部分模型使用的特征，我们对power和预测值进行了log变换，以减小数据的变化范围。

我们还观察了特征重要性，搞了tiny版本的特征，删去了冗余特征，添加了单周统计特征(min、max、std、mean)，这儿不赘言了，有兴趣可见[get_feature_cloumn_tiny](https://github.com/lvniqi/tianchi_power/blob/master/code/preprocess.py#L605)。

### 模型设计(线下部分)
最终版本的线下模型会对31天分开训练，每天使用了六组特征不同的模型，分别是做过log变换和未做过log变换的28天特征、7天特征、以及tiny版特征(共2\*3=6)。

每组模型首先使用了1个3层500棵树的xgboost做清洗。

而后使用2个种不同比例抽取最优秀的样本作为清洗后训练集，再训练2个5至6层1000至1600棵树的xgboost模型。

每组模型的训练方式如下所示，所以共计产生了31\*(6\*3)个xgboost模型。

<div align=center>
<img src="https://github.com/lvniqi/tianchi_power/blob/master/image/train_xgb_down.png" width = "403" height = "322" alt="train-xgb" align=center />
</div>

### 模型融合(线下部分)
模型融合部分我们使用tensorflow设计了一个线性回归的模型(详细可见类[tf_percent_model](https://github.com/lvniqi/tianchi_power/blob/master/code/train_tensorflow.py#L532))，
对每个商家单独做模型融合。原因是考虑到不同的店可能适合不同的模型，而且600+个条目应该足够线性回归训练了，另外使用xgboost的线性回归模型能控制结果缩放比例(zoom)，使得结果可控。
``` python
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
```

### 反思(线下部分)
**模型融合存在问题，没有划分训练集和验证集，导致无法做交叉验证。虽然单个模型的棵树和层数事先测试过，但是融合之后说不清楚，也没法验证。**

## 线上解法介绍
---

### 特征工程(线上部分)
#### 数据清洗
线上的模型仅仅去掉了[最近一周总电量小于100的店家](https://github.com/lvniqi/tianchi_power/blob/master/code/get_feature_column_sql.py#L199)，其他清洗放在欠拟合模型中。虽然这个欠拟合模型已经不那么欠拟合了。
对于1416以及1414这两家店使用预测值进行了异常值填补(因为这两家店在11月底的时候停产了,由于我们做了滑窗，去除这两个异常值意味着去除一整个11月的训练数据)。
#### 特征选择
线上部分由于SQL的限制和对阿里PAI平台不太熟悉的原因，我们实在无力像线下那样暴力做特征了，所以只能进一步简化特征。
最终版本的特征是在线下的tiny版本的基础上做完的，参考了线上GBDT的特征重要性，去掉了facebook prophet特征，加入了星期几以及月份数特征。这些特征做了one-hot编码但并没有使用，因为考虑到做one-hot编码过于稀疏。

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

### 模型设计及模型融合(线上部分)

线上部分的模型设计和模型融合是在一起做的。首先贴上整个实验图。

<div align=center>
<img src="https://github.com/lvniqi/tianchi_power/blob/master/image/%E6%A8%A1%E5%9E%8B%E5%9B%BE.PNG" width = "496" height = "510" alt="test" align=center />
</div>

首先将数据集划分10%作为测试集，作为线性回归模型融合的数据来源。剩下90%作为GBDT及PS-SMART训练使用。

#### 模型设计
##### PS-SMART+GBDT
由于我们没有发现PAI平台能在IDE上敲建模命令这个隐藏功能，所以只能大幅压缩模型。
最终版本的线上模型用了1个4层1000棵树的PS-SMART做清洗，而后训练集以三种不同比例抽取最优秀的样本作为清洗后训练集，再训练1个5层1000棵树的PS-SMART+2个6层1000棵树的GBDT。
为了加大各个模型间的差异，我们将特征进行采样，使每个模型得到大约2/3数量的原始特征(类似随机森林中特征提取)(见[split_features](https://github.com/lvniqi/tianchi_power/blob/master/code/preprocess.py#L790))。
大致的流程图如下图所示。

</div>
<div align=center>
<img src="https://github.com/lvniqi/tianchi_power/blob/master/image/train_xgb_up.png" width = "567" height = "321" alt="train-xgb" align=center />
</div>

##### ARIMA

考虑到11月以及12月的节假日较少，用电量相对比较平稳(通过观察15年11,12月电量)，比较适合使用时间序列模型进行建模。
所以我们使用了ARIMA模型对数据进行了时序建模。
话说PAI平台的AUTO-ARIMA超好用啊，不用怎么调参即可达到不错的效果，点赞。

#### 模型融合
##### 线性回归融合
第一种方式即是之前的实验图展示的那样，使用线性回归融合PS-SMART及GBDT预测结果。
而后再将线性回归结果与ARIMA加权得到最终结果。
##### 模型预测取中值
最后一天提交时，
考虑到线上的线性回归无法对每家店单独操作，
我们索性直接将PS-SMART及GBDT所得到的4个预测结果取中值，
以期为每个店自动选择合适的模型进行预测，
并剔除不合适的异常模型对结果造成的影响。
SQL代码是这样的:
```sql
INSERT INTO TABLE tianchi_power_answer_gbdt_avg
SELECT '1'
	, SUM((ordinal(2, prediction_result_1, prediction_result_2, prediction_result_3, prediction_result_4) + ordinal(3, prediction_result_1, prediction_result_2, prediction_result_3, prediction_result_4)) / 2)
FROM gbdt_predict_day_1
GROUP BY day_num;
```
使用模型预测结果对取中值和LR回归以SAE为代价作比较，如下图所示。

</div>
<div align=center>
<img src="https://github.com/lvniqi/tianchi_power/blob/master/image/median_lr.png" alt="median-lr" align=center />
</div>

可见取中值基本上比线性回归要稍好一些，
而且还不考虑这个是没有做交叉验证的，
在真实情况下取中值效果相比较而言肯定会更优秀一点。

所以最后一天提交结果基本上用了中值结果，但考虑到保险起见，毕竟只有最后一次机会了，仍然加权了ARIMA及线性回归融合模型结果，现在看来不加权可能会更好一点。

## 关于人工调整
除了上述模型训练结果，我们还对模型训练结果进行过手动的微调。
但是这些个微调是不是合理的呢？这就要靠撞大运了。

起初在线下时，我们常将模型融合完的计算结果乘以一个尺度系数来逼近答案。
在10月的预测中，10月1日至7日使用模型怎么也预测不好。
最后使用去年的数据总体乘了个系数使得10月7日和10月8日能连续接上，这才看起来正常点。

此后在线上的比赛中，我们觉得不能胡乱乘系数了，开始使用总体电量的ARIMA预测均值作为基准计算系数。
但是使用ARIMA的结果是天气或者季节特征无法被利用。

在最后一次的评测中，我们想起了信号处理里面的中值滤波，改掉了模型融合中的LR，直接以4个模型的预测中值作为融合，这样减少了单个异常模型对单家店的影响，为保险起见又加权了ARIMA和LR模型融合结果。
观察最后结果，考虑到31日为元旦假期首日，14日(哎.. 27号忘记看了)最低温度过低，加入了人工规则进行修正。然而这次就没这么幸运啦。
看到天渡发的图(目测黑色是基准数据)，眼泪掉下来...简直是越改越差了嘛...

所以各位，我奉劝一句，别乱加人工，还是相信模型的为好。

## 其他脑洞
还有些其他脑洞线下还来不及做，在这儿记录下。

#### 企业聚类
在线下的时候尝试过对企业用电情况的一个聚类，由于当时只是单纯的对电量数据使用DBSCAN划分，聚类的效果并不好，因此后来就没有考虑了。但是后来想了一下，其实可以利用温度和节假日等特征进行聚类，或者使用相关性分析的方法筛选出不同的企业，在保证每种企业训练样本充足的情况下，使用不同的模型进行训练，或者将分类好的分类信息作为onehot变量加入之前的模型。

#### onehot
onehot特征(如假期等)太过稀疏了，直接拿来用tree based model训练，在节点分裂的时候不一定会被看上。所以可以在下次比赛时试试先过个线性回归试试。

<div align=center>
<img src="https://github.com/lvniqi/tianchi_power/blob/master/image/onehot_lr.png" width = "393" height = "328" alt="onehot-lr" align=center />
</div>

#### Other
由于我们分别对31天划分了31个模型，并且31个模型的预测情况都不太一样，最后对模型修正的时候是对整体修正的，应该使用其他方法对每个模型都单独修正。

