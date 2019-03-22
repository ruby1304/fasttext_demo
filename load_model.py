import warnings
from gensim.models import FastText
from gensim.test.utils import datapath
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import jieba
import csv
from opencc import OpenCC
import sys
from gensim.test.utils import get_tmpfile
from sklearn.cluster import KMeans
import numpy as np
from pprint import pprint

fname = get_tmpfile("fasttext.model")

model = FastText.load(fname)
# for x in model.wv.most_similar("小姑娘",topn=20):
#     print(x)
infered_vectors_list = []

i = 0

X =  model.wv[model.wv.vocab]

infered_vectors_list  = X

# pprint(X)
# pprint(infered_vectors_list)


# X =  np.array(model.wv[model.wv.vocab]).reshape(len(model.wv.vocab), -1)

for i in range(5,30):
    kmean_model = KMeans(n_clusters=i)
    kmean_model.fit(X)
    print (i,kmean_model.inertia_)


#
#
# labels= kmean_model.predict(infered_vectors_list[0:100])
# cluster_centers = kmean_model.cluster_centers_


# with open("own_claasify.txt", 'w') as wf:
#     for i in range(100):
#         string = ""
#         text = model.wv[i][0]
#         for word in text:
#             string = string + word
#         string = string + '\t'
#         string = string + str(labels[i])
#         string = string + '\n'
#         wf.write(string)

print('done.')




# cap_path = datapath("C:/Users/Ruby/Desktop/cc.zh.300.vec/cc.zh.300.vec")
# fb_full = FastText.load_fasttext_format("C:/Users/Ruby/Desktop/cc.zh.300.bin/cc.zh.300.bin")
#
#
# for x in fb_full.wv.vocab:
#     print(x)