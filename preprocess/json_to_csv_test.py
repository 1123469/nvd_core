import os
print(os.getcwd())
"""
encapsulate the method of json_to_csv
"""
import pandas as pd
import json

def json_to_csv_append(year,path):
    with open('..\\data\\raw\\nvdcve_json\\nvdcve-1.1-'+year+'.json','r',encoding='utf-8') as f:
        data = json.load(f)
    list_y = data['CVE_Items']
    cve_cwe_list = []
    cve_id_list = []
    cve_des_list = []
    for cve in list_y:
        cwe_value_pre = cve['cve']['problemtype']['problemtype_data'][0]['description']
        # if len(cwe_value_pre)==0:
        #     continue
        if cwe_value_pre == []:
            continue
        cve_id_list.append(cve['cve']['CVE_data_meta']['ID'])
        for value in cwe_value_pre:
            cwe_value = value
        cve_cwe_list.append(cwe_value['value'])
        cve_des_list.append(cve['cve']['description']['description_data'][0]['value'])
    cve_dic = {'cve_id':cve_id_list,'cwe_id':cve_cwe_list,'descript':cve_des_list}
    df = pd.DataFrame(cve_dic)
    df = df[~df['cwe_id'].str.contains('noinfo')]
    df.to_csv(path,header=False,index = False,mode='a')

def json_to_csv_override(year,path):
    with open('..\\data\\raw\\nvdcve_json\\nvdcve-1.1-'+year+'.json','r',encoding='utf-8') as f:
        data = json.load(f)
    list_y = data['CVE_Items']
    cve_cwe_list = []
    cve_id_list = []
    cve_des_list = []
    for cve in list_y:
        cwe_value_pre = cve['cve']['problemtype']['problemtype_data'][0]['description']
        if len(cwe_value_pre)==0:
            continue
        cve_id_list.append(cve['cve']['CVE_data_meta']['ID'])
        for value in cwe_value_pre:
            cwe_value = value
        cve_cwe_list.append(cwe_value['value'])
        cve_des_list.append(cve['cve']['description']['description_data'][0]['value'])
    cve_dic = {'cve_id':cve_id_list,'cwe_id':cve_cwe_list,'descript':cve_des_list}
    df = pd.DataFrame(cve_dic)
    df = df[~df['cwe_id'].str.contains('noinfo')]
    df.to_csv(path,header=False,index = False)
