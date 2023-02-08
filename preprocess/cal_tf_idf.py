import math
import pandas as pd
from keras_preprocessing.text import Tokenizer
from clean_text import clean_text

def get_tf_idf(year,key_word_num):
    label_path = '..\\data\\clean\\nvdcve-1.1-'+year+'_labels.csv'
    dataset = pd.read_csv(label_path, header=None)
    dataset.columns = ['cve_id', 'cwe_id', 'descript', 'label']
    dataset['contents'] = dataset['descript'].apply(clean_text)
    tokenizer = Tokenizer()  # 创建一个Tokenizer对象
    tokenizer.fit_on_texts(dataset['contents'])  # fit_on_texts函数可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小
    vocab = tokenizer.word_index
    print(vocab)
    tf_idf = {}
    for doc in dataset['contents']:
        for word in doc:
            if word not in tf_idf:
                tf_idf[word] = math.log(float(tokenizer.document_count) / float(tokenizer.word_docs[word]))

    tf_idf = sorted(tf_idf.items(), key=lambda word_tfidf: word_tfidf[1], reverse=True)
    print(tf_idf)
    save_path = '..//data//tf_idf//'+year+"_"+str(key_word_num)
    with open(save_path, 'w') as f:
        f.write(str(tf_idf[:key_word_num]))



