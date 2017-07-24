
#WEATHER_COLUMN = ['temp','bad_weather','dew']
WEATHER_COLUMN = ['temperature_low','temperature_high','weather_val']
def get_dayofweek_one_hot(day):
    return map(lambda x:'d%d_%d'%(day,x),range(7))
def get_monthofyear_one_hot(day):
    return map(lambda x:'m%d_%d'%(day,x),range(12))
def day2str(day):
    day_str = '%d'%day
    return day_str.replace('-','_')

def get_cols(col,day):
    return col+"%s"%day2str(day)

def get_power(day):
    return "power%s"%day2str(day)
    
def get_holiday(day):
    return "holiday%s"%day2str(day)

def get_festday(day):
    return "festday%s"%day2str(day)
    
def get_weather(col,day):
    return get_cols(col,day)
    
def get_yearly(day):
    return "yearly%s"%day2str(day)
    
def dropna_line(col_str):
    return col_str + ' IS NOT NULL'
    
def dropna_val(col_str,val=10):
    return col_str + ' > %d'%val

def select_columns(dataset,col_list):
    col_list_str = reduce(lambda x,y:x+','+y,col_list)
    return 'select %s from %s'%(col_list_str,dataset)

def get_feature_cloumn_tiny(day,
                       feature_range = 7,
                       holiday_range = 5,
                       prophet_range = 3,
                       has_extera_holiday = False,
                       has_extera_weather = False,
                       has_holiday = True,
                       has_weather = True,
                       has_prophet = False,
                       has_power = True,
                       has_history = True,
                       has_y = False,
                       has_user_id = False,
                       has_day_of_week = True,
                       has_day_of_month = True,
                       
                       ):
    assert holiday_range%2 == 1
    assert feature_range%2 == 1
    assert prophet_range%2 == 1
    def get_day_his_list():
        day_his_list = []
        day_t = day - 1
        while day_t > -30:
                if(day_t) < -feature_range:
                    day_his_list.append(day_t)
                if(day_t-1) < -feature_range and (day_t-1)>-30:
                    day_his_list.append((day_t-1))
                if(day_t+1) < -feature_range and (day_t+1)>-30:
                    day_his_list.append((day_t+1))
                day_t -= 7
        return day_his_list
    feature_column = []
    day_his_list = get_day_his_list()
    
    if has_user_id:
        feature_column += ['date_time','day_num','user_id',]
    
    if has_prophet:
        end_pos = day-1-prophet_range/2 if day-1-prophet_range/2<0 else 0
        feature_column += map(get_yearly,
              range(-feature_range,end_pos))
        feature_column += map(get_yearly,
              range(day-1-prophet_range/2,day-1+prophet_range/2+1))
            
    if has_extera_weather:
        end_pos = day-1-2 if day-1-2<0 else 0
        for weather_col in WEATHER_COLUMN:
            feature_column += map(lambda day:get_weather(weather_col,day),range(-feature_range,end_pos))
    if has_weather:
        for weather_col in WEATHER_COLUMN:
            feature_column += map(lambda day:get_weather(weather_col,day),range(day-1-2,day))
    if has_extera_holiday:
        end_pos = day-1-holiday_range/2 if day-1-holiday_range/2<0 else 0
        feature_column += map(get_holiday,range(-feature_range,end_pos))
        feature_column += map(get_festday,range(-feature_range,end_pos))
                               
    if has_holiday:
        feature_column += map(get_holiday,range(day-1-holiday_range/2,day-1+holiday_range/2+1))
        feature_column += map(get_festday,range(day-1-holiday_range/2,day-1+holiday_range/2+1))
    if has_power:
        feature_column += map(get_power,range(-1,-feature_range-1,-1))
        feature_column += map(get_power,day_his_list)
    if has_history:
        for his_columns in ['mean7','max7','min7','std7']:
            feature_column += map(lambda day:get_cols(his_columns+'_power',day),[-28,-21,-14,-7])
    if has_day_of_week:
        feature_column += [get_cols('dayofweek',(day-1)),]
    if has_day_of_month:
        feature_column += [get_cols('monthofyear',(day-1)),]
    if has_y:
        feature_column += [get_power(day-1),]
    return feature_column 

def get_feature_mm(feature_list,operator,rst_name):
    rst_str_list = []
    for pos,feature_col in  enumerate(feature_list[:-1]):
        head = '  Case\n'
        str_t_list = []
        for feature_col_t in feature_list[pos+1:]:
            str_t = feature_col+operator+feature_col_t
            str_t_list.append(str_t)
        str_line = reduce(lambda x,y:x+' AND '+y,str_t_list)
        str_line = '    \tWhen ' + str_line + ' Then '+feature_col
        rst_str_list.append(str_line)
    rst_str = head+reduce(lambda x,y:x+'\n'+y,rst_str_list)
    rst_str += '\n    Else '+feature_list[-1]
    rst_str += '\n  End As %s'%rst_name
    return rst_str
    
def get_feature_min(feature_list,rst_name = 'min7_power'):
    col_list_str = reduce(lambda x,y:x+','+y,feature_list)
    return 'least(%s) as %s'%(col_list_str,rst_name)
    #return get_feature_mm(feature_list,'<=',rst_name)
    
def get_feature_max(feature_list,rst_name = 'max7_power'):
    col_list_str = reduce(lambda x,y:x+','+y,feature_list)
    return 'greatest(%s) as %s'%(col_list_str,rst_name)
    #return get_feature_mm(feature_list,'>=',rst_name)

def get_feature_mean(feature_list,rst_name = 'mean7_power'):
    col_list_str = reduce(lambda x,y:x+'+'+y,feature_list)
    return "("+col_list_str+")/7 as %s"%rst_name

def get_feature_std(feature_list,mean_name='mean7_power_7',rst_name='std7_power_7',dataset_name='t'):
    def pow_cut_col(col):
        return 'pow(%s - %s, 2)'%(dataset_name+'.'+col,dataset_name+'.'+mean_name)
    pow_list = map(pow_cut_col,feature_list)
    pow_str = reduce(lambda x,y:x+' + '+y,pow_list)
    sqrt_str = 'sqrt(('+pow_str+')/7)'
    
    return sqrt_str+'as %s'%rst_name

def get_feature_statistic(dataset,cols,pre_name = '7_power',week_name='_7'):
    pre_all_name = pre_name+week_name
    std_str = get_feature_std(cols,'mean'+pre_all_name,'std'+pre_all_name,'t')
    min_max_mean_str = '( select day_num, user_id,'+'\n'+\
        get_feature_max(cols,'max'+pre_all_name)+'\n,'\
        +get_feature_min(cols,'min'+pre_all_name)+'\n,'\
        +get_feature_mean(cols,'mean'+pre_all_name)+'\n,'\
        +reduce(lambda x,y:x+','+y,cols)+'\n'\
        +'from %s'%dataset+'\n) t;'
    
    all_str = 'select t.day_num, t.user_id,'+'\n'+\
        't.'+'max7_power'+week_name+','+\
        't.'+'min7_power'+week_name+','+\
        't.'+'mean7_power'+week_name+','+\
        std_str+' from '+min_max_mean_str
                         
    return all_str
    
def create_31_day_model(day_range = range(1,32),
                        is_write = True):
    rst_list = []
    for day in day_range:
        rst = ''
        day_dataset = "feat_slide_day_%d"%day
        rst += 'DROP TABLE IF EXISTS %s;\n'%day_dataset
        feature_column = get_feature_cloumn_tiny(day,has_user_id=True,has_y = True)
        rst += 'create table %s as '%day_dataset
        #rst += select_columns("pure_feat_slide_all_wd_dom",feature_column)
        rst += select_columns("pure_feat_slide_all_onehot",feature_column)
        rst += ';\n'
        rst_list.append(rst)
    #-----get na feature-----''
    for day in day_range:
        rst = ''
        day_dataset = "feat_slide_day_%d"%day
        rst += 'DROP TABLE IF EXISTS %s_na;\n'%day_dataset
        rst += 'create table %s_na as select * '%day_dataset
        rst += 'from %s where '%day_dataset
        feature_column = get_feature_cloumn_tiny(day,has_user_id=True,has_y = True)
        feature_column_dropna = map(dropna_line,feature_column)
        rst += reduce(lambda x,y:x+'\n'+'AND '+y,feature_column_dropna)
        #power_feature_column = \
        #    reduce(lambda x,y:x+[y,] if y.startswith('pow') else x,feature_column,[])
        power_feature_column = map(lambda day:'power%s'%day2str(day),range(-7,0))
        power_feature_column += ['power%d'%(day-1)]
        power_feature_column_n1 = map(dropna_val,power_feature_column)
        rst += '\nAND ' + reduce(lambda x,y:x+'+ '+y,power_feature_column)+' > 100'
        rst += ';\n'
        rst_list.append(rst)
    if is_write:
        sql_file = open('./test_sql.sql','w')
        rst_all_str = reduce(lambda x,y:x+'\n'+y,rst_list)
        sql_file.write(rst_all_str)
        sql_file.close()
    return rst_list
    
def create_31_day_predict(day_range = range(1,32),day_num = 701,
                        is_write = True):
    rst_list = []
    for day in day_range:
        rst = ''
        day_dataset = "feat_slide_day_%d"%day
        day_dataset_predict = "feat_slide_day_%d_predict"%day
        rst += 'DROP TABLE IF EXISTS %s;\n'%day_dataset_predict
        feature_column = get_feature_cloumn_tiny(day,has_user_id=True,has_y = False)
        #power_feature_column = \
        #    reduce(lambda x,y:x+[y,] if y.startswith('pow') else x,feature_column,[])
        power_feature_column = map(lambda day:'power%s'%day2str(day),range(-7,0))
        rst += 'create table %s as '%day_dataset_predict
        rst += select_columns(day_dataset,feature_column)
        rst += '\n where  day_num=%d and '%day_num
        feature_column_dropna = map(dropna_line,feature_column)
        rst += reduce(lambda x,y:x+'\n'+'AND '+y,feature_column_dropna)
        power_feature_column_n1 = map(dropna_val,power_feature_column)
        rst += '\nAND ' + reduce(lambda x,y:x+'+ '+y,power_feature_column)+' > 100'
        rst += ';'
        rst_list.append(rst)
    if is_write:
        sql_file = open('./test_sql_predict.sql','w')
        rst_all_str = reduce(lambda x,y:x+'\n'+y,rst_list)
        sql_file.write(rst_all_str)
        sql_file.close()
    return rst_list
    
def split_feature_columns(columns):
    columns = list(columns)
    c_p1 = columns[::3]
    c_p2 = columns[1::3]
    c_p3 = columns[2::3]
    c_1 = c_p1+c_p2
    c_2 = c_p2+c_p3
    c_3 = c_p1+c_p3
    return (c_1,c_2,c_3)
    
def onthot_monthofyear():
    rst = ''
    for i in range(31):
        for j in range(12):
            rst+= "if(a.monthofyear"+str(i)+"="+str(j+1)+",1,0) as m"+str(i)+"_"+str(j)+","
    return rst
def onthot_dayofweek():
    rst = ''
    for i in range(31):
        for j in range(7):
            rst+= "if(a.dayofweek"+str(i)+"="+str(j)+",1,0) as d"+str(i)+"_"+str(j)+","
    return rst
    
def feature_columns2str(feature_column):
    return reduce(lambda x,y:x+' , '+y,feature_column)

def extera_feature(day):
    return ['day_num', 'user_id', 'date_time','power%d'%(day-1) ]
            
def get_splited_feature_columns(day,has_extera_feature = False):
    feature_columns = split_feature_columns(get_feature_cloumn_tiny(day))
    if has_extera_feature:
        feature_columns = map(lambda column : extera_feature(day)+column,feature_columns)
    return map(feature_columns2str,feature_columns)
    
def get_nosplited_feature_columns(day,has_extera_feature = False):
    feature_column = get_feature_cloumn_tiny(day)
    if has_extera_feature:
        feature_column = extera_feature(day)+feature_column
    return feature_columns2str(feature_column)
if __name__ == '__main__':
    print '------test dropna------'
    a = []
    for weather_col in WEATHER_COLUMN:
            a += map(lambda day:get_weather(weather_col,day),range(-30,33))
    a += map(get_festday,range(-30,33))
    a += map(get_holiday,range(-30,33))
    a += map(get_yearly,range(-7,33))
    a += map(get_power,range(-29,30))
    a = map(dropna_line,a)
    print reduce(lambda x,y:x+'\n'+'AND '+y,a)
    print '------test get_feature_cloumn_tiny------'
    print get_feature_cloumn_tiny(1)
    print '------if final dataset is "temp"------'
    feature_column = get_feature_cloumn_tiny(1)
    print select_columns("temp",feature_column)
    print '------get statistic of power -14~-8------'
    cols = map(get_power,range(-14,-7))
    print get_feature_statistic('slide_power_consumption_60',\
                                cols,pre_name = '7_power',week_name='_14')
    create_31_day_model()
    create_31_day_predict()