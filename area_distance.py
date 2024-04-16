import math
import pandas as pd
import numpy as np
import re

# def distance_of_stations(point1, point2):
#     long1, lat1 = point1
#     long2, lat2 = point2
#     delta_long = math.radians(long2 - long1)
#     delta_lat  = math.radians(lat2 - lat1)
#     s = 2 * math.asin(math.sqrt(math.pow(math.sin(delta_lat / 2), 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.pow(math.sin(delta_long / 2), 2)))
#     s = s * 6378.2    
#     return s
def distance_of_stations(long1, lat1, long2, lat2):
    delta_long = math.radians(long2 - long1)
    delta_lat  = math.radians(lat2 - lat1)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(delta_lat / 2), 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.pow(math.sin(delta_long / 2), 2)))
    s = s * 6378.2 * 1000    
    return s # 单位米

df= pd.read_csv('processed_data\\map\\small_areas2center.csv',encoding="utf-8", low_memory=False)
# df = df.iloc[0:,:]
print(df.head(5))


segnums = pd.read_csv('processed_data\\map\\small_areas2center.csv',encoding="utf-8", low_memory=False)
segnums.sort_values(by=['Smallarea'], inplace=True)

areanums = pd.read_csv('processed_data\\map\\areanum_valid.csv',encoding="utf-8", low_memory=False)
areanums_list = areanums['Smallareas'].values
print(areanums_list)

segnums = segnums[segnums['Smallarea'].isin(areanums_list)]
segnums.reset_index(inplace=True)

Lons = segnums['Lon']
Lats = segnums['Lat']



# sampled_seg2geo = [] # 选取的路段坐标列表

# for i in range(len(df)):
#     if df['Areaname'][i] in segnums_list:
#         sampled_seg2geo.append(df['Loc'][i])



# seg_num = len(sampled_seg2geo) # 路段数量
# locs = sampled_seg2geo
# # locs = df1['Loc']

distance_matrix = np.zeros((len(Lons),len(Lons)))
# # neighbour_matrix = np.array([[0,0,1,0,1,0,1,1,0,0,0,0,1], [0,0,0,0,1,0,0,0,1,0,1,1,1], [1,0,0,0,1,1,0,0,0,0,0,0,0], \
# #     [0,0,0,0,0,0,1,0,0,0,0,1,0], [1,1,1,0,0,1,1,0,0,0,1,0,1], [0,0,1,0,1,0,0,0,0,1,1,0,0],\
# #         [1,0,0,1,1,0,0,1,0,0,0,1,1], [1,0,0,0,0,0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,1,0],\
# #             [0,0,0,0,0,1,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,0,0,0,0,0],[0,1,0,1,0,0,1,0,1,0,0,0,1],\
# #                 [0,1,0,0,1,0,1,0,0,0,0,1,0]])
# # print(neighbour_matrix)
# disrelation_matrix = np.zeros((seg_num,seg_num)) #距离关系矩阵
for i in range(len(Lons)):
    # p1 = locs[i]
    # lon1=float(re.findall(r"\((.+?),",p1)[0])
    # lat1=float(re.findall(r", (.+?)\)",p1)[0])
    lon1 = Lons[i]
    lat1 = Lats[i]
    for j in range(len(Lons)):
        # p2 = locs[j]
        # lon2=float(re.findall(r"\((.+?),",p2)[0])
        # lat2=float(re.findall(r", (.+?)\)",p2)[0])
        lon2 = Lons[j]
        lat2 = Lats[j]
        distance_matrix[i][j] = distance_of_stations(lon1, lat1, lon2, lat2)

# for i in range(len(Lons)):
#     for j in range(len(Lons)):
#         if(i != j ):
#             # if neighbour_matrix[i][j] != 1:# 不相邻
#             #disrelation_matrix[i][j] = 1/distance_matrix[i][j]
# #print(disrelation_matrix)

# df2 = pd.DataFrame(neighbour_matrix)
# df2.to_csv('processed_data\map\W1_area_test.csv',index = 0,header=None)
df3 = pd.DataFrame(distance_matrix)
df3.to_csv('processed_data\map\D1_area_test.csv',index = 0,header=None)
# df4 = pd.DataFrame(disrelation_matrix)
# df4.to_csv('processed_data\map\W2_area_test.csv',index = 0,header=None)





