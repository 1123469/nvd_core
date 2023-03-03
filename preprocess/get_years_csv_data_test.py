from json_to_csv import json_to_csv_append

# years = ['2020','2021']
# years = ['2020','2021','2022']
years = ['2013','2014','2015','2017','2018','2019','2020','2021','2022']
infix = str(years[0])+"to"+str(years[len(years)-1])
path = '..\\data\\clean\\nvdcve-1.1-'+infix+'.csv'
if __name__ == '__main__':
    for year in years:
        json_to_csv_append(year,path)

