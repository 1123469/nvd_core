from train_wv import train_wv

# years = ['2020','2021']
# years = ['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
years = ['2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022']
infix = ''
infix = str(years[0])+"to"+str(years[len(years)-1])
# 标记
infix+='_no'
# cwe_min_count = 500
# cwe_min_count = 700
# cwe_min_count = 2000
# cwe_min_count = 2500
cwe_min_count = 3000
infix+='_'+str(cwe_min_count)
if __name__ == '__main__':
    # train_wv(infix,100,1,5)
    # train_wv(infix, 200, 1, 5)
    train_wv(infix, 300, 1, 5)
