import pandas as pd
from preprocess.clean_text import clean_text

def train_wv(year,vec_len,count,window_len):
    label_path = '..\\..\\data\\clean\\nvdcve-1.1-'+year+'_labels.csv'
    dataset = pd.read_csv(label_path, header=None)
    dataset.columns = ['cve_id', 'cwe_id', 'descript', 'label']
    dataset['contents'] = dataset['descript'].apply(clean_text)
    print(dataset['contents'])
    from gensim.models import word2vec  # 导入gensim包
    model = word2vec.Word2Vec(dataset['contents'], vector_size=vec_len, min_count=count, window=window_len)  # 设置训练参数
    save_path = '..//..//models//wv//'+year+"_"+str(vec_len)+"_"+str(count)+"_"+str(window_len)+'.pkl'
    model.save(save_path)


