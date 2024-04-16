from numpy.lib.function_base import kaiser
from numpy.lib.histograms import histogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit        #导入leastsq模块

# def func1(x,a,b):
#     return a * np.exp(-b * x)
# P0 = [0.2, 0.2]                         # 第一次猜测的函数拟合参数
# 停车事件
df = pd.read_csv("processed_data\\parking_events\\small_area_parking_two_months.csv")
# 划分的区域的列表
arealist = pd.read_csv("processed_data\map\\areanum copy.csv")["AreaName"].values

timeline_max = 240
timeline = np.arange(0,int(timeline_max+5),5)
xlabels = np.arange(0,int(timeline_max+5),timeline_max/4)


histogram_time_num = np.zeros((len(arealist), len(timeline)-1))
print(histogram_time_num.shape)
duration_list = []
# for i in range(len(arealist)):
#     df1 = df[df["SmallAreas"] == arealist[i]]
#     for j in range(len(timeline)):
#         if (j< (len(timeline)-1)):
#             df2 = df1[(df1['DurationMinutes'] <= timeline[j+1]) & (df1['DurationMinutes'] > timeline[j])]
#             histogram_time_num[i][j] = len(df2)
#         else:
#             df2 = df1[(df1['DurationMinutes'] >= timeline[j])]
#             histogram_time_num[i][j-1] = histogram_time_num[i][j-1] + len(df2)

# df3 = pd.DataFrame(histogram_time_num)
# df3.T.to_csv('processed_data\parking_events\\area_duration_num.csv',index=0,header=None)

for i in range(len(arealist)):
    df1 = df[df["SmallAreas"] == arealist[i]]   # 某区域的停车事件
    duration_list.append(df1['DurationMinutes'].values)


plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

color_list = ['#E2E2DF', '#D2D2CF', '#E2CFC4', '#F7D9C4', '#FAEDCB', '#C9E4DE', '#C6DEF1','#DBCDF0' ,'#F2C6DE', '#F9C6C9',\
'#E2E2DF', '#D2D2CF', '#E2CFC4', '#F7D9C4', '#FAEDCB', '#C9E4DE', '#C6DEF1','#DBCDF0' ,'#F2C6DE', '#F9C6C9',\
'#E2E2DF', '#D2D2CF', '#E2CFC4', '#F7D9C4', '#FAEDCB', '#C9E4DE', '#C6DEF1','#DBCDF0' ,'#F2C6DE', '#F9C6C9',\
'#E2E2DF', '#D2D2CF', '#E2CFC4', '#F7D9C4', '#FAEDCB', '#C9E4DE', '#C6DEF1','#DBCDF0' ,'#F2C6DE', '#F9C6C9']
fig = plt.figure( figsize = (16,9))
fig.subplots_adjust(hspace=0.8, wspace=0.4)

for i in range(1,len(arealist)+1):
    ax = fig.add_subplot(6,7,i)
    a = duration_list[i-1]
    n, bins_limits, patches = ax.hist(a, bins = timeline, color = color_list[i-1])
    print(bins_limits[:(len(timeline)-1)]+2.5, n)
    if (i==1):
        datas = np.array(bins_limits[:(len(timeline)-1)]+2.5)
        datas = np.row_stack((datas,n/(np.sum(n)*5)))
    else:
        datas = np.row_stack((datas,n/(np.sum(n)*5)))
    # ax.plot(bins_limits[:(len(timeline)-1)]+2.5, func1(bins_limits[:(len(timeline)-1)]+2.5, popt[0],popt[1]), label="拟合数据", linewidth = 0.3, color ='r', ls ='--')
    ax.plot(bins_limits[:(len(timeline)-1)]+2.5,n,linewidth = 0.3, color ='k', ls ='-.')
    ax.set_title(arealist[i-1])
    ax.set_xticks(xlabels)
    # ax.set_xlabel('Durations Time (minutes)')
    # ax.set_ylabel('Number of Parking Events')
print(datas)
# plt.savefig(f"results\\figures\\durations_{timeline_max}min_test.svg", format="svg")
data = pd.DataFrame(datas)
data.to_csv('processed_data\\duration_distribution\\distribution_240min.csv', header=None, index=0)
plt.show()