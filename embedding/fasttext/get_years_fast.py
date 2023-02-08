from train_fast import train_fast

years = ['2020','2021']
infix = ''
infix = str(years[0])
for i in range(1,len(years)):
    infix += '-'+str(years[i])
if __name__ == '__main__':
    train_fast(infix,vec_len=100,count=1,window_len=5)
