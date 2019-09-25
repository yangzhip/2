from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import re

r = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？【】?、~@#￥%……&*（）()]+")#去除噪声值
neg=pd.read_excel('neg.xls',header=None,index=None)#读取负面语句
pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
all= pd.read_csv('Train_Data.csv', encoding='utf-8')
npos = all.loc[0:,['title']]
npos=np.array(npos)
for i in range(len(npos)):
    npos[i] = r.sub('',str(npos[i]))
npos=pd.DataFrame(npos)
npos['mark']= all['negative']
pos['mark']=0
neg['mark']=1 #给训练语料贴上标签
pn=pd.concat([pos,neg,npos],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目
cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)

#读入评论内容
comment = pd.read_csv('Test_Data.csv', encoding='utf-8')
comment = comment[comment['text'].notnull()] #仅读取非空评论
comment['words'] = comment['text'].apply(cw) #评论分词

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)

w = [] #将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))

print('Build model...')
model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=16, nb_epoch=10) #训练时间为若干个小时

classes = model.predict_classes(xt)
comment['sent'] = comment['words'].apply(get_sent)
comment['sent'] = list(sequence.pad_sequences(comment['sent'], maxlen=maxlen))
yaa=np.array(list(comment['sent']))
new_classes=model.predict(yaa)
acc=model.evaluate(yaa,new_classes)
for i in range(len(new_classes)):
    if new_classes[i] >=float(0.5):
        new_classes[i]=1
    else:
        new_classes[i]=0
print(acc)