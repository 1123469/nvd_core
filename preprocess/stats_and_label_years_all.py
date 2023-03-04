import os
print(os.getcwd())

import pandas as pd
# 核心代码，设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
import operator
from sklearn.preprocessing import LabelEncoder
# year = '2020'
# year = '2021'
# years = ['2020','2021']
# years = ['2020','2021','2022']
# years = ['2017','2018','2019','2020','2021','2022']
# years = ['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
years = ['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
infix = str(years[0])+"to"+str(years[len(years)-1])
path = '..\\data\\clean\\nvdcve-1.1-'+infix+'.csv'
dataset=pd.read_csv(path,header = None)
dataset.columns=['cve_id','cwe_id','descript']
cwe_list=list(dataset['cwe_id'])
cwe_set=set(cwe_list)
cwe_dict={}

for cwe in cwe_set:
    cwe_dict[cwe]=cwe_list.count(cwe)

cwe_dict_sort=sorted(cwe_dict.items(),key=operator.itemgetter(1),reverse=True)

# cwe_min_count = 2000
cwe_min_count = 2500
# cwe_min_count = 3000
#获取前n个漏洞类型的数据
df=dataset.copy()
for cwe in cwe_set:
    if cwe_dict[cwe] < cwe_min_count or cwe=='NVD-CWE-Other':
    # if cwe_dict[cwe] < cwe_min_count:
        df = df[~df['cwe_id'].isin([cwe])]
# 标记
infix+='_no'
print(len(df))

print(cwe_dict_sort)
print(len(cwe_dict_sort))
print(set(list(df['cwe_id'])))
print(len(set(list(df['cwe_id']))))

#设置类别标签并写入文件
encoder=LabelEncoder()
df['labels']=encoder.fit_transform(df['cwe_id'])
cwe_labels = list(df['labels'])
print(len(cwe_labels))
cwe_labels = set(cwe_labels)
print(len(cwe_labels))

decode_list = []
for i in range(len((cwe_labels))):
    decode_list.append(encoder.inverse_transform([i])[0])
print(decode_list)
decode_list_save_path = '..\\data\\decode_label\\nvdcve-1.1-'+infix+'_'+str(cwe_min_count)+'_decode_list'


label_save_path = '..\\data\\clean\\nvdcve-1.1-'+infix+'_'+str(cwe_min_count)+'_labels.csv'
# df.to_csv(label_save_path,header = None,index=False)
# with open(decode_list_save_path, 'w') as f:
#     f.write(str(decode_list))

