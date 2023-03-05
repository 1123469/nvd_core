from train_fast import train_fast

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
    train_fast(infix,vec_len=100,count=1,window_len=5)
    # train_fast(infix, vec_len=200, count=1, window_len=5)
    # train_fast(infix, vec_len=300, count=1, window_len=5)
