#!/usr/bin/env python
# coding: utf-8

# # 0. 모듈 불러오기

# In[1]:


import multiprocessing
threads = multiprocessing.cpu_count()-1

from multiprocessing import Pool

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

# 데이터 불러오기
train_data = pd.read_json('train.json', encoding='UTF-8', typ = 'frame')
valid_data = pd.read_json('val.json', encoding='UTF-8', typ = 'frame')
song_meta = pd.read_json('song_meta.json', encoding='UTF-8', typ = 'frame')
genre_gn_all = pd.read_json('genre_gn_all.json',encoding='UTF-8', typ = 'series')
genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})


# > ## 1-2. Data Preprocessing

# In[4]:

# song_meta에 한국어 장르명도 넣기
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

# train에 song 장르 채워넣기
def fn_fill_kor_genre(j):
    uni_genre=[list(chain((list(chain(*song_meta[song_meta['id']==i].song_gn_gnr_basket_kor))))) for i in train_data.songs[j]]
    b = list(set(list(chain(*uni_genre))))

    if j%1000==0:
        time.sleep(3)
        
    return b

# 불용어 제거
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

# khaiii 사용자 사전 추가
def fn_add_khaiidic(words):
    _home = os.getcwd()
    _home
    
    f = open("khaiii/rsc/src/preanal.manual",'a')

    for i in range(len(words)):
        data = "{}* \t {}/NNG\n".format(words[i],words[i])
        f.write(data)
    f.close()
    
    os.chdir(_home+'/khaiii/rsc')
    os.system('mkdir -p ../build/share/khaiii')
    
    _ppath = 'PYTHONPATH=%s/khaiii/src/main/python/ ./bin/compile_preanal.py --rsc-src=./src --rsc-dir=%s/khaiii/build/share/khaiii'
    os.system(_ppath % (_home,_home))
    
    os.chdir(_home)
    
    print("#### Finished adding to dictionary ####")
    
def fn_pre_khaiii(i,playlist):
    tmp_list = []
    
    if len(playlist) == playlist.count(' '):
        # 빈칸으로만 되어있는 행
        pass
    elif playlist.find(" ")==-1:
        # 띄어쓰기 없는 행
        tmp_list.append(playlist)
    else:
        for word in api.analyze(playlist):
            for morph in word.morphs:
                # 실질형태소 가져오기
                if morph.tag in ['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ','SN','SL']:
                    tmp_list.append(morph.lex)
    if i%100==0:
        print(i)
    if i%1000==0:
        time.sleep(10)
        
    return tmp_list

def fn_exc_khaiii(args):
    return fn_pre_khaiii(*args)

def parallel_product(list_a, list_b):
    # spark given number of processes
    p = Pool(threads) 
    # set each matching item into a tuple
    job_args = [(item_a, list_b[i]) for i, item_a in enumerate(list_a)] 
    # map to pool
    return p.map(fn_exc_khaiii, job_args)