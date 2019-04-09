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
import pickle
import time

start = time.time()

cc = OpenCC('s2t')  # convert from Simplified Chinese to Traditional Chinese
# cc.set_conversion('t2s')
cc.set_conversion('s2tw')

sentences = []
words = {}
fname = get_tmpfile("fasttext.model")
allword = {}
frequency = {}


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# jieba.load_userdict("C:/Program Files/Python37/Lib/site-packages/jieba/dict.txt")
jieba.load_userdict("C:/Program Files/Python37/Lib/site-packages/jieba/dict_voc.txt")

keyword = stopwordslist("C:/Users/Ruby/PycharmProjects/gensim_test/keyword.txt")  # 这里加载停用词的路径

sys.setrecursionlimit(3000)

with open('C:/Users/Ruby/PycharmProjects/gensim_test/articles.csv', newline='' , encoding='utf-8') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        # print(row['text'])
        tw = cc.convert(str(row['text']))
        # print(tw)
        seg_list = jieba.cut(tw, cut_all=False)

        # print(" ".join(seg_list))
        if str(row['id']) not in allword.keys():
            allword[str(row['id'])] = 0
            countWord = []
            outstr =[]
            for x in seg_list:
                if x in keyword:
                    countWord.append(x)
                    if x not in outstr:
                        outstr.append(x)
            if outstr:
                with open("getKeyword.txt","a", encoding='utf-8-sig') as f:
                    # f.write(str(row['id'])+","+str(tw.replace('\r','').replace('\n','').replace(',','').replace(' ',''))+","+str(" ".join(outstr))+","+str(len(outstr))+","+str(len(countWord))+"\n")
                    f.write(str(" ".join(outstr))+","+str(len(outstr))+","+str(len(countWord))+"\n")


print('gogogo')


end = time.time()
print(end - start)