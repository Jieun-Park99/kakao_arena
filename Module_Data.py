#!/usr/bin/env python
# coding: utf-8

# # 0. 모듈 불러오기

# In[1]:


import multiprocessing
threads = multiprocessing.cpu_count()-1


# In[2]:

import re
import json
import time
import io
import os
import json
import pandas as pd
import numpy as np
import tqdm

from itertools import chain


# # 1. Data Loading & preprocessing
# > ## 1-1. Data Loading

# In[3]:


train_data = pd.read_json('train.json', encoding='UTF-8', typ = 'frame')
valid_data = pd.read_json('val.json', encoding='UTF-8', typ = 'frame')
song_meta = pd.read_json('song_meta.json', encoding='UTF-8', typ = 'frame')
genre_gn_all = pd.read_json('genre_gn_all.json',encoding='UTF-8', typ = 'series')
genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})


# > ## 1-2. Data Preprocessing

# In[4]:


def fn_as_kor_genre(j):
    b=[]
    for i in song_meta.song_gn_gnr_basket[j]:
        if (str(i) in np.array(genre_gn_all.gnr_code)) == False:
            b.append('None')
        else:
            b.append(genre_gn_all.gnr_name[genre_gn_all.gnr_code==str(i)].values[0])

    if j%1000==0:
        time.sleep(5)
        
    return b



# In[8]:


def fn_fill_kor_genre(j):
    uni_genre=[list(chain((list(chain(*song_meta[song_meta['id']==i].song_gn_gnr_basket_kor))))) for i in train_data.songs[j]]
    b = list(set(list(chain(*uni_genre))))

    if j%1000==0:
        time.sleep(3)
        
    return b


def fn_clean_str(text):
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]' # 특수기호 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = "\u3000\u3000"     # 유니코드 삭제
    text = re.sub(pattern=pattern, repl='',string=text)
    pattern = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE) #마지막점검 제거
    text = re.sub(pattern=pattern, repl='',string=text)
    return text
