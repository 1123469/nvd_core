import pandas as pd
import numpy as np
from clean_text import clean_text
import joblib
from sklearn.ensemble import ExtraTreesClassifier

# 根据TF-IDF，将文本转换为向量
def word2vec(tf_idf_dict, doc_sentence):
    keywords = list(dict(tf_idf_dict).keys())  # 获取关键词
    tf_weight = list(dict(tf_idf_dict).values())  # 获取关键词tf_idf值

    docvec_list = []
    for sentence in doc_sentence:
        docvec = [0] * len(tf_idf_dict)
        for word in sentence:
            if word in keywords:
                docvec[keywords.index(word)] = tf_weight[keywords.index(word)]
        docvec_list.append(docvec)
    return docvec_list

# 将训练集和测试集换为文本向量
def doc_vec(x_train,tf_idf_dict):
    # 训练集转换为向量
    train_lenth = len(x_train)
    train_data_list = []
    for i in range(train_lenth):
        train_data_list.append(x_train[i])
    train_docvec_list = word2vec(tf_idf_dict, train_data_list)
    return train_docvec_list

def get_ig(year,tf_idf_num,ig_word_num):
    tf_idf_path  = '..//data//tf_idf//'+year+"_"+str(tf_idf_num)
    ig_save_path = '..//data//ig//'+year+"_"+str(ig_word_num)
    with open(tf_idf_path, 'r') as f:
        tf_idf_list = eval(f.read())
    tf_idf_dict = dict(tf_idf_list)
    label_path = '..\\data\\clean\\nvdcve-1.1-' + year + '_labels.csv'
    dataset = pd.read_csv(label_path, header=None)
    dataset.columns = ['cve_id', 'cwe_id', 'descript', 'label']
    dataset['contents'] = dataset['descript'].apply(clean_text)
    x_train = dataset['contents']
    cw = lambda x: int(x)
    y_train = np.array(dataset['label'].apply(cw))
    print(y_train[:2])
    x_train_new = doc_vec(x_train, tf_idf_dict)  # 训练集向量化
    print(x_train_new[:2])

    # 导入SelectFromModel结合ExtraTreesClassifier计算特征重要性，并按重要性阈值选择特征。
    clf_model = ExtraTreesClassifier(n_estimators=250, random_state=0)
    # clf_model=RandomForestClassifier(n_estimators=250,random_state=0)
    print("开始计算信息增益")
    clf_model.fit(x_train_new, y_train)
    # joblib.dump(clf_model, 'NVD/models/features_model')
    # 获取每个词的特征权重,数值越高特征越重要l
    importances = clf_model.feature_importances_
    # 选择特征重要性为1.5倍均值的特征
    # model = SelectFromModel(clf_model, threshold='1.5*mean', prefit=True)
    print("开始存入字典")
    feature_words_dic = {}
    # 将词和词的权重存入字典并写入文件
    for i in range(len(tf_idf_list)):
        feature_words_dic[tf_idf_list[i][0]] = importances[i]
    # 对字典按权重由大到小进行排序
    words_info_dic_sort = sorted(feature_words_dic.items(), key=lambda x: x[1], reverse=True)
    # 将前ig_word_num个词的权重字典写入文件
    key_words_importance = dict(words_info_dic_sort[:ig_word_num])
    with open(ig_save_path, 'w') as f:
        f.write(str(key_words_importance))







