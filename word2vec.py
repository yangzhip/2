import multiprocessing
import jieba
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
# cw = lambda x: list(jieba.cut(x)) #定义分词函数
# data = pd.read_csv('Train_Data.csv', encoding='utf-8')
# data = data[data['title'].notnull()] #仅读取非空评论
# title_text = pd.DataFrame()
# title_text[0] = data['text']+data['title']
# title_text['words'] = title_text[0].apply(cw)
# del title_text[0]
#
# # str =  ' '.join(str(i)for i in title_text)
# # str_cut = jieba.cut(str)
# # result = ' '.join(str_cut)
# # with open('word.txt',encoding='utf-8') as f:
# #     document = f.read()
# #     document_cut = jieba.cut(document)
# #     result = ' '.join(document_cut)
# # with open('result.txt', 'w', encoding="utf-8") as f2:
# #     f2.write(result)
# #
# # sentences = LineSentence("D:/PythonCode/yang/test/result.txt")
# path = get_tmpfile("D:/PythonCode/yang/test/w2v_model.bin")  # 创建临时文件
# sentence=list()
# for i in title_text['words']:
#     sentence.append(i)
# model = Word2Vec(sentence, hs=1,min_count=1,window=10,size=100)
# # 模型储存与加载1
# model.save(path)
model = Word2Vec.load("D:/PythonCode/yang/test/w2v_model.bin")
