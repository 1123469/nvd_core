from json_to_csv import json_to_csv_override
# year = '2020'
# year = '2021'
# year = '2022'
year = '2016'
path = '..\\data\\clean\\nvdcve-1.1-'+year+'.csv'
if __name__ == '__main__':
    json_to_csv_override(year,path)
