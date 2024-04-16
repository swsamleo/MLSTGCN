# 用来确定不同area对应功能的数量

import pandas as pd
from collections import Counter
import random

df = pd.read_csv('datasets\map\Business_establishment_trading_name_and_industry_classification_2019.csv', encoding="utf-8", low_memory=False)
df2019 = df[df['Census year'] == 2019]  # 仅选取2019年部分
print(df2019)
Areas = list(df['CLUE small area'].drop_duplicates(keep="first"))
Areas2fun_num = {}  # 区域对应功能数量字典
for area in Areas:
    area_fun_df = df2019[df2019['CLUE small area'] == area]  # 区域对应所有功能dataframe
    area_fun_df.sort_values(by=['Industry (ANZSIC4) code'],inplace=True)
    area_fun_list = list(area_fun_df['Industry (ANZSIC4) code'])    # 区域对应所有功能列表
    areafunnum = dict(Counter(area_fun_list))
    Areas2fun_num[area] = areafunnum
df3 = pd.DataFrame.from_dict(Areas2fun_num, orient='index')
df3.fillna(0, inplace=True) # 用0填充空值
df4 = df3.T # 转置
df4.sort_index(axis=1,inplace=True)
print(df4)
print(df4.corr())
df4.to_csv('processed_data\\map\\area2funnum.csv')
df4.corr().to_csv('processed_data\\map\\areafuncorr.csv')
# segid = list(df['SegID'])
# markernum = dict(Counter(segid))
# df3 = pd.DataFrame.from_dict(markernum, orient='index')
# df3.to_csv('processed_data\map\segnum.csv')