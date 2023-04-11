import tensorflow as tf
from gensim.models import word2vec, fasttext
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import  sys


def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(w, pos='n') for w in words]
    stopwords = {}.fromkeys([line.rstrip() for line in open('F:\\PycharmProjects\\NVDproject\\nvdcve\\stopwords.txt')])
    eng_stopwords = set(stopwords)
    words = [w for w in lem_words if w not in eng_stopwords]
    return words

if __name__ == '__main__':
    years = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014',
             '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    infix = ''
    infix = str(years[0]) + "to" + str(years[len(years) - 1])
    infix += '_no'

    cwe_min_count = 3000

    infix += '_' + str(cwe_min_count)

    vec_len = 100
    # vec_len = 200
    # vec_len = 300
    min_count = 1
    window_len = 5
    # dense_unit = 128
    dense_unit = 256

    n = 30
    cwe_count = 11
    fast_model_path = 'E:\\软件开发实践\\毕设项目\\NVDProject\\nvd_core\\models\\fasttext\\2002to2022_no_3000_100_1_5.pkl'
    fast_model = fasttext.FastText.load(fast_model_path)
    load_path = 'E:\\软件开发实践\\毕设项目\\NVDProject\\nvd_core\\models\\fasttextcnn'
    model = tf.keras.models.load_model(load_path)
    test_text = "In the settings app, there is a possible app crash due to improper input validation. This could lead to local denial of service of the Settings app with User execution privileges needed. User interaction is not needed for exploitation.Product: AndroidVersions: Android-10Android ID: A-136005061"
    test_text = "A buffer overflow in the FTP list (ls) command in IIS allows remote attackers to conduct a denial of service and, in some cases, execute arbitrary commands."
    # test_text = sys.argv[1]
    sentence = clean_text(test_text)
    embedding_matrix = np.zeros((n, vec_len))
    padding_const = 1e-10
    embedding_matrix *= padding_const
    padding_line = np.ones(vec_len)
    padding_line *= padding_const

    length = len(sentence)

    range_num = n if length > n else length
    for i in range(range_num):
        try:
            embedding_matrix[i] = fast_model.wv[sentence[i]]
        except KeyError:
            continue

    line = sentence
    train_dataset = [embedding_matrix]

    train_dataset = np.array(train_dataset)
    print(train_dataset.shape)
    predict_res = model.predict(train_dataset,False)
    predict_label = int(tf.argmax(predict_res,axis=1))
    # print("预测的结果是")
    # print(predict_label)
    # decode_list_save_path = 'F:\\PycharmProjects\\nvd_reload\data\decode_label\\nvdcve-1.1-2020-2021-2022_700_decode_list'
    decode_list_save_path = 'E:\\软件开发实践\\毕设项目\\NVDProject\\nvd_core\\data\\decode_label\\nvdcve-1.1-2002to2022_no_3000_decode_list'
    with open(decode_list_save_path,'r') as f:
        decode_list = eval(f.read())
    print(decode_list[predict_label])
    raise Exception(str(decode_list[predict_label]))

