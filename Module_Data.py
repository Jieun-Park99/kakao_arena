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
import itertools

from itertools import chain


# # 1. Data Loading & preprocessing
# > ## 1-1. Data Loading

# In[3]:

# 데이터 불러오기
train_data = pd.read_json('train.json', encoding='UTF-8', typ = 'frame')
song_meta = pd.read_json('song_meta.json', encoding='UTF-8', typ = 'frame')
genre_gn_all = pd.read_json('genre_gn_all.json',encoding='UTF-8', typ = 'series')
genre_gn_all = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})


# > ## 1-2. Data Preprocessing

# In[4]:

#####################
#### song_meta ######
#####################

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

def fn_change_date(dates):
    
    date = str(dates)
    
    date = date.replace("-","",10)
    date = date.replace(":","",10)
    date = date.replace(".","",10)
    date = date.replace(" ","",10)
    date = date[:8]

    date = date[:4] + re.sub('00','01',date[4:6]) + re.sub('00','01',date[6:8])
    
    return date

def load_song_meta(song_meta=song_meta):
    print("Loading song_meta.json ...")
    start_time = time.time()
    with Pool(processes=threads) as p:
        result= p.map(fn_as_kor_genre, range(0,len(song_meta)))
    print("--- %s seconds ---" % (time.time()-start_time))
    song_meta['song_gn_gnr_basket_kor'] = result
    
    song_meta.issue_date = list(map(fn_change_date,song_meta.issue_date.tolist()))
    return song_meta

# In[8]:

#####################
#### train_data #####
#####################

# train에 song 장르 채워넣기
def fn_fill_kor_genre(j):
    uni_genre=[list(chain((list(chain(*song_meta[song_meta['id']==i].song_gn_gnr_basket_kor))))) for i in train_data.songs[j]]
    b = list(set(list(chain(*uni_genre))))

    if j%1000==0:
        time.sleep(3)
        
    return b

# train data 채워넣기
def load_train_data(train_data=train_data):
    print('Loading train.json...')
    start_time = time.time()
    with Pool(processes=threads) as p:
        result = p.map(fn_fill_kor_genre, range(0,len(train_data)))
    print("--- %s seconds ---" % (time.time()-start_time))
    train_data['genre'] = result
    
    return train_data

#####################
##### cleaning ######
#####################

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

def fn_clean_data(finals):
    for i in range(0,len(finals)):
        for j in range(0, len(finals[i])):
            finals[i][j] = finals[i][j].lower()

    import re
    p = re.compile('[a-z]')

    # 영어 한글자 제거
    for i in range(0,len(finals)):
        for j in range(0, len(finals[i])):
            if len(finals[i][j]) == 1:
                if p.match(finals[i][j]) != None:
                    finals[i][j] = ''

    for i in range(0,len(finals)):
        while '' in finals[i]:
            finals[i].remove('')

    # 중첩문자 제거
    for i in range(0,len(finals)):
        finals[i] = list(set(finals[i]))

    for i in range(0,len(finals)):
        for j in range(0, len(finals[i])):
            if finals[i][j].isdigit() == True:
                if len(finals[i][j]) != 4:
                    finals[i][j] = ''

    for i in range(0,len(finals)):
        while '' in finals[i]:
            finals[i].remove('')
    
    return(finals)

#####################
#### khaiii ######
#####################

# khaiii 사용자 사전 추가
def fn_add_khaiidic(words):
    _home = os.getcwd()
    _home
    
    f = open("khaiii/rsc/src/preanal.manual",'a')

    for i in range(len(words)):
        data = "{} \t {}/NNG\n".format(words[i],words[i])
        f.write(data)
    f.close()
    
    os.chdir(_home+'/khaiii/rsc')
    os.system('mkdir -p ../build/share/khaiii')
    
    _ppath = 'PYTHONPATH=%s/khaiii/src/main/python/ ./bin/compile_preanal.py --rsc-src=./src --rsc-dir=%s/khaiii/build/share/khaiii'
    os.system(_ppath % (_home,_home))
    
    os.chdir(_home)
    
    print("#### Finished adding to dictionary ####")



