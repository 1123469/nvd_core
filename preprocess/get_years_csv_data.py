from json_to_csv import json_to_csv_append

# years = ['2020','2021']
years = ['2020','2021','2022']
infix = str(years[0])
for i in range(1,len(years)):
    infix += '-'+str(years[i])
path = '..\\data\\clean\\nvdcve-1.1-'+infix+'.csv'
if __name__ == '__main__':
    for year in years:
        json_to_csv_append(year,path)

