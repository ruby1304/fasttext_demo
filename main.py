import warnings
from gensim.models import FastText
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import jieba
import csv
from opencc import OpenCC
import sys
from gensim.test.utils import get_tmpfile
import operator
import pprint

cc = OpenCC('s2t')  # convert from Simplified Chinese to Traditional Chinese
# cc.set_conversion('t2s')
cc.set_conversion('s2tw')

sentences = []
words = {}
fname = get_tmpfile("fasttext.model")
allword = {}


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# jieba.load_userdict("C:/Program Files/Python37/Lib/site-packages/jieba/dict.txt")
jieba.load_userdict("C:/Program Files/Python37/Lib/site-packages/jieba/dict_voc.txt")

stopwords = stopwordslist("stop.txt")  # 这里加载停用词的路径

sys.setrecursionlimit(3000)

with open('articles.csv', newline='' , encoding='utf-8') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        # print(row['text'])
        tw = cc.convert(str(row['text']))
        # print(tw)
        seg_list = jieba.cut(tw, cut_all=False)

        # print(" ".join(seg_list))

        afterCut = []
        for x in seg_list:
            outstr =""
            # if x not in stopwords:
            if x != '\t' and x != '\n':
                if x in allword.keys():
                    allword[x] = allword[x] + 1
                else:
                    allword[x] = 1
                afterCut.append(x)
        # print(afterCut)
        sentences.append(afterCut)
# print(sentences)

with open("ducanci.txt","a", encoding='utf-8-sig') as f:
    for x in allword.keys():
        f.write(str(x)+","+str(allword[x])+"\n")


# with open("C:/Users/Ruby/Desktop/huawei_data/huawei_0000_1205.csv", encoding='utf-8-sig') as f:
#     for i,line in enumerate(f):
#         for idx, val in enumerate(line.split(',')):
#             if idx == 5:
#                 seg_list = jieba.cut(val, cut_all=False)
#
#
# sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]

print('gogogo')
model = FastText(sentences,  size=100, window=5, min_count=5, iter=10, min_n=1 , max_n=6, word_ngrams=1)
# print(model.most_similar('你'))  # 词向量获得的方式
# print(model.wv['你']) # 词向量获得的方式

model.save(fname)

for x in model.wv.most_similar("癌症",topn=50):
    print(x)