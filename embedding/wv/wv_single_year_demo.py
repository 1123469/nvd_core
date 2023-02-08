import numpy as np
from gensim.models import word2vec  # 导入gensim包
import pandas as pd
import tensorflow as tf
from preprocess.clean_text import clean_text

# year = '2020'
# year = '2021'
year = '2022'
vec_len = 100
min_count = 1
window_len = 5
dense_unit = 128
wv_model_path  = '..//..//models//wv//'+year+"_"+str(vec_len)+"_"+str(min_count)+"_"+str(window_len)+'.pkl'
label_path = '..\\..\\data\\clean\\nvdcve-1.1-'+year+'_labels.csv'
n=30
# cwe_count = 11  # 2020
# cwe_count = 16  # 2021
cwe_count = 16  # 2022

def get_label_one_hot(list):
    values = np.array(list)
    n_values = np.max(values) + 1
    return np.eye(n_values)[values]

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

class TextCNN(Model):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.c1 = Conv2D(filters=12, kernel_size=(3, vec_len), padding='same')  # 卷积层
        self.b1 = BatchNormalization()  # BN层
        self.a1 = Activation('relu')  # 激活层
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)  # dropout层

        self.flatten = Flatten()
        self.f1 = Dense(dense_unit, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(cwe_count, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y



if __name__ == '__main__':
    wv_model = word2vec.Word2Vec.load(wv_model_path)
    dataset = pd.read_csv(label_path, header=None)
    dataset.columns = ['cve_id', 'cwe_id', 'descript', 'label']
    dataset['contents'] = dataset['descript'].apply(clean_text)
    train_dataset = []
    for line in dataset['contents']:
        length = len(line)
        if length > n:
            line = line[:n]
            word2vec_matrix = (wv_model.wv[line])
            train_dataset.append(word2vec_matrix)
        else:
            word2vec_matrix = (wv_model.wv[line])
            pad_length = n - length
            pad_matrix = np.zeros([pad_length, vec_len]) + 1e-10
            word2vec_matrix = np.concatenate([word2vec_matrix, pad_matrix], axis=0)
            train_dataset.append(word2vec_matrix)
    train_dataset = np.expand_dims(train_dataset, 3)
    label_dataset = get_label_one_hot(dataset['label'])

    from sklearn.model_selection import train_test_split
    model = TextCNN()
    x_train, x_test, y_train, y_test = train_test_split(train_dataset, label_dataset, test_size=0.2, random_state=217)
    batch_size = 32
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)



    model.compile(optimizer=tf.optimizers.Adam(1e-3),
                  loss=tf.losses.categorical_crossentropy, metrics=['accuracy'])
    model.fit(train_data, epochs=10)
    score = model.evaluate(x_test, y_test)
    print('last score:', score)
    print(model.summary())

    # model.fit(x=x_train, y=y_train, epochs=10, verbose=2)
    # model.evaluate(x=x_test, y=y_test)


