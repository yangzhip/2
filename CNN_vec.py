from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.utils import to_categorical
import re
import warnings
import multiprocessing
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import gensim

warnings.simplefilter(action = "ignore", category = RuntimeWarning)
r = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？【】?、~@#￥%……&*（）()]+")
all= pd.read_csv('预处理.csv', encoding='gbk')
for i in range(len(all)):
    all.loc[i,'entity'] = str(all.loc[i,'entity']).replace(' ','')
all.to_csv('预处理.csv',encoding='gbk')
npos = all.loc[0:,['entity']]
shit =all['entity']
shit = np.array(shit)
shit = shit.tolist()
x=shit

for i in range(len(shit)):
     shit[i]=str(shit[i]).replace(' ', '')
test_str = "\n".join(shit)
with open('dict.txt', 'w+', encoding="utf-8") as f:
        f.write(test_str)
jieba.load_userdict('./dict.txt')
cw = lambda x: list(jieba.cut(x)) #定义分词函数
all1 = pd.read_csv('Train_DataSet.csv', encoding='utf-8')
all2 = pd.read_csv('Test_Data.csv', encoding='utf-8')
data = all1[all1['text'].notnull()] #仅读取非空评论
data =data['text']
data=np.array(data)
for i in range(len(data)):
    data[i] = r.sub('',str(data[i]))
data=pd.DataFrame(data)
data1 = all2[all2['text'].notnull()] #仅读取非空评论
data1 =data1['text']
data1=np.array(data1)
for i in range(len(data1)):
    data1[i] = r.sub('',str(data1[i]))
data1=pd.DataFrame(data1)
data['words']=data[0].apply(cw)
data['mark']=all1['negative']
data1['words']=data1[0].apply(cw)
shit=pd.DataFrame(shit)
sentence=list()
for i in data['words']:
    sentence.append(i)
for i in data1['words']:
    sentence.append(i)
sentence.append(x)
sentence = LineSentence('dict.txt')

model = Word2Vec(sentence,min_count=1)
path = get_tmpfile("D:/PythonCode/yang/test/w2v_model.bin")  # 创建临时文件
model.save(path)
# pn=pd.concat([data[:3000]],ignore_index=True) #合并语料
# comment=pd.concat([data[3000:]],ignore_index=True)
# get_sent = lambda x: list(model[x])
# pn['sent'] = pn['words'].apply(get_sent) #速度太慢
# comment['sent']=comment['words'].apply(get_sent)
# maxlen = 50
# print("Pad sequences (samples x time)")
# pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
# comment['sent']=list(sequence.pad_sequences(comment['sent'],maxlen=maxlen))
# x = np.array(list(pn['sent'])) #训练集
# y = np.array(list(pn['mark']))
# xt = np.array(list(comment['sent'])) #测试集
# yt = np.array(list(comment['mark']))
# xa = np.array(list(pn['sent'])) #全集
# ya = np.array(list(pn['mark']))
# del all1,all2
# print('Build model...')
# model = Sequential()
#
# model.add(Conv1D(256, 5,padding='same',input_shape=(x.shape[1],100)))
# model.add(MaxPooling1D(3, 3, padding='same'))
# model.add(Conv1D(128, 5, padding='same'))
# model.add(MaxPooling1D(3, 3, padding='same'))
# model.add(Conv1D(64, 3, padding='same'))
# model.add(MaxPooling1D(3, 3, padding='same'))
# model.add(Conv1D(32, 1, padding='same'))
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(x, y,epochs=50, batch_size=900)
# classes = model.predict_classes(xt)
# acc = model.evaluate(xt,yt)
