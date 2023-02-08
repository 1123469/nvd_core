import pandas as pd
import numpy as np
from gensim.models import fasttext

from preprocess.clean_text import clean_text

year = '2020'
vec_len = 100
min_count = 1
window_len = 5
n=30

fast_model_path  = '..//..//models//fasttext//'+year+"_"+str(vec_len)+"_"+str(min_count)+"_"+str(window_len)+'.pkl'
label_path = '..\\..\\data\\clean\\nvdcve-1.1-'+year+'_labels.csv'

def get_label_one_hot(list):
    values = np.array(list)
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]

if __name__ == '__main__':
    dataset = pd.read_csv(label_path, header=None)
    fast_model = fasttext.FastText.load(fast_model_path)
    dataset.columns = ['cve_id', 'cwe_id', 'descript', 'label']
    dataset['contents'] = dataset['descript'].apply(clean_text)
    train_dataset = []
    for line in dataset['contents']:
        length = len(line)
        if length > n:
            line = line[:n]
            word2vec_matrix = (fast_model.wv[line])
            train_dataset.append(word2vec_matrix)
        else:
            word2vec_matrix = (fast_model.wv[line])
            pad_length = n - length
            pad_matrix = np.zeros([pad_length, vec_len]) + 1e-10
            word2vec_matrix = np.concatenate([word2vec_matrix, pad_matrix], axis=0)
            train_dataset.append(word2vec_matrix)
    train_dataset = np.expand_dims(train_dataset, 3)
    label_dataset = get_label_one_hot(dataset['label'])

    print(len(train_dataset))
    print(train_dataset[0])