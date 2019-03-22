import warnings
from gensim.models import FastText
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import jieba
import csv
from opencc import OpenCC
import sys
from gensim.test.utils import get_tmpfile

#
# f = open(r'C:/Users/Ruby/Desktop/dict (1).txt', 'r', encoding='utf-8')
# for i in f:
#     print(i)


from hanzi_chaizi import HanziChaizi

hc = HanziChaizi()
result = hc.query('鋶')
print("".join(result))

result = hc.edquery('金亠厶川')
print("".join(result))
