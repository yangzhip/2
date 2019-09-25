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
from gensim.models.word2vec import LineSentencenpos = all.loc[0:,['entity']]
from gensim.models import KeyedVectors
import gensim
from gensim.models import Word2Vec
all= pd.read_csv('预处理.csv', encoding='gbk')
npos = all.loc[0:,['entity']]
label=all['mark']
npos = pd.DataFrame(npos)
sent=pd.DataFrame([])
model = Word2Vec.load("D:/PythonCode/yang/test/w2v_model.bin")
get_sent = lambda x: list(model[x])
npos['sent'] = npos['entity'].apply(get_sent)
# print("Pad sequences (samples x time)")
# npos['sent'] = list(sequence.pad_sequences(npos['sent'], maxlen=50))
# pn=pd.concat([npos[:1500]],ignore_index=True) #合并语料
# comment=pd.concat([npos[1500:]],ignore_index=True)
# x = np.array(list(pn['sent'])) #训练集
# y = np.array(list(pn['mark']))
# xt = np.array(list(comment['sent'])) #测试集
# yt = np.array(list(comment['mark']))
# xa = np.array(list(pn['sent'])) #全集
# ya = np.array(list(pn['mark']))
# print('Build model...')
# model = Sequential()
# model.add(Conv1D(256, 5,padding='same',input_shape=(x.shape[1],100)))
# model.add(MaxPooling1D(3, 3, padding='same'))
# model.add(Conv1D(128, 5, padding='same'))
# model.add(MaxPooling1D(3, 3, padding='same'))
# model.add(Conv1D(64, 3, padding='same'))
# model.add(MaxPooling1D(3, 3, padding='same'))
# model.add(Conv1D(32, 1, padding='same'))
# model.add(Flatten())

# acc = model.evaluate(xt,yt)# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.fit(x, y,epochs=50, batch_size=900)
# classes = model.predict_classes(xt)
