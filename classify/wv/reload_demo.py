import tensorflow as tf
from gensim.models import word2vec
import numpy as np

from preprocess.clean_text import clean_text

if __name__ == '__main__':
    years = ['2020', '2021', '2022']
    infix = ''
    infix = str(years[0])
    for i in range(1, len(years)):
        infix += '-' + str(years[i])

    # cwe_min_count = 500
    cwe_min_count = 700
    infix += '_' + str(cwe_min_count)

    vec_len = 100
    # vec_len = 200
    # vec_len = 300
    min_count = 1
    window_len = 5
    dense_unit = 128
    n = 30
    wv_model_path = '..//..//models//wv//' + infix + "_" + str(vec_len) + "_" + str(min_count) + "_" + str(
        window_len) + '.pkl'
    wv_model = word2vec.Word2Vec.load(wv_model_path)
    years = ['2020', '2021', '2022']
    infix = ''
    infix = str(years[0])

    for i in range(1, len(years)):
        infix += '-' + str(years[i])

    load_path = '../../models/textcnn'
    model = tf.keras.models.load_model(load_path)
    # test_text = "A cross site scripting vulnerability exists when Microsoft Dynamics 365 (on-premises) does not properly sanitize a specially crafted web request to an affected Dynamics server, aka 'Microsoft Dynamics 365 (On-Premise) Cross Site Scripting Vulnerability'."
    # 2015 right
    # test_text = "Cross-site scripting (XSS) vulnerability in the Policy Admin Tool in Apache Ranger before 0.5.0 allows remote attackers to inject arbitrary web script or HTML via the HTTP User-Agent header."
    #CVE-2015-0137  CWE-20 wrong

    # test_text = "IBM PowerVC Standard 1.2.0.x before 1.2.0.4 and 1.2.1.x before 1.2.2 validates Hardware Management Console (HMC) certificates only during the pre-login stage, which allows man-in-the-middle attackers to spoof devices via a crafted"
    # 2020  CWE-20 预测正确
    test_text = "In the settings app, there is a possible app crash due to improper input validation. This could lead to local denial of service of the Settings app with User execution privileges needed. User interaction is not needed for exploitation.Product: AndroidVersions: Android-10Android ID: A-136005061"
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
            embedding_matrix[i] = wv_model.wv[sentence[i]]
        except KeyError:
            continue

    train_dataset = [embedding_matrix]
    train_dataset = np.expand_dims(train_dataset, 3)
    print(train_dataset.shape)
    predict_res = model.predict(train_dataset,False)
    predict_label = int(tf.argmax(predict_res,axis=1))
    print(predict_res)
    decode_list_save_path = '..\\..\\data\\decode_label\\nvdcve-1.1-' + infix + '_' + str(cwe_min_count) + '_decode_list'
    with open(decode_list_save_path,'r') as f:
        decode_list = eval(f.read())

    print("预测的结果是")
    print(decode_list[predict_label])

