import pandas as pd
from preprocess.clean_text import clean_text

def train_fast(year,vec_len,count,window_len,skip_gram=1,hierarchical_softmax=1,worker_num=10):
    label_path = '..\\..\\data\\clean\\nvdcve-1.1-' + year + '_labels.csv'
    dataset = pd.read_csv(label_path, header=None)
    dataset.columns = ['cve_id', 'cwe_id', 'descript', 'label']
    dataset['contents'] = dataset['descript'].apply(clean_text)
    from gensim.models import fasttext  # 导入gensim包
    model = fasttext.FastText(vector_size=vec_len, min_count=count, window=window_len, sg=skip_gram, hs=hierarchical_softmax, workers=worker_num)
    model.build_vocab(dataset['contents'])
    model.train(dataset['contents'], total_examples=model.corpus_count, epochs=50)
    save_path = '..//..//models//fasttext//' + year + "_" + str(vec_len) + "_" + str(count) + "_" + str(window_len) + '.pkl'
    model.save(save_path)
