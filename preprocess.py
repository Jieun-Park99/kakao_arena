#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import Module_Data as MD
from Module_Data import song_meta
from Module_Data import train_data
from Module_Data import load_song_meta
from Module_Data import load_train_data
from Module_Data import threads
from Module_Data import fn_clean_str
from Module_Data import fn_clean_data
from Module_Data import fn_change_date

import copy
import json
import time
from multiprocessing import Pool
import time
import io
import os
import json
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import chain

#valid_data = pd.read_json('val.json', encoding='UTF-8', typ = 'frame')
valid_data = pd.read_json('test.json', encoding='UTF-8', typ = 'frame')


song_meta = load_song_meta()
train_data = load_train_data()


# # In[3]:


i = 0
playlist = []

while i < len(train_data):
    mylist = [[train_data.plylst_title[i]]]
    clean = fn_clean_str(' '.join(sum(mylist,[])))
    if len(clean)==0:
        clean = ' '
    playlist.append(clean)
    i+=1


del mylist, clean

# 공백제거
for i in range(len(playlist)):
    playlist[i]=playlist[i].strip()


# # In[7]:


# # words = ['발라드','캐럴','케롤','스타워즈','뉴에이지','게임','프로필',
# #          '마쉬멜로','유산슬','감성발라드','일렉트로닉','섹시','록메탈','힐링']

# # MD.fn_add_khaiidic(words)


# # In[4]:


from khaiii import KhaiiiApi
from pprint import pprint
api = KhaiiiApi(rsc_dir='khaiii/build/share/khaiii') # 내 설치 경로

from multiprocessing import Pool
import time

def fn_analyze_khaiii(i):
    tmp_list = []
    
    if len(playlist[i]) == playlist[i].count(' '):
        # 빈칸으로만 되어있는 행
        pass
    elif playlist[i].find(" ")==-1:
        # 띄어쓰기 없는 행
        tmp_list.append(playlist[i])
    else:
        for word in api.analyze(playlist[i]):
            for morph in word.morphs:
                # 실질형태소 가져오기
                if morph.tag in ['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ','SN','SL']:
                    tmp_list.append(morph.lex)

    if i%1000==0:
        time.sleep(10)
        
    return tmp_list


def load_morphs(playlist=playlist):
    print("Loading morphs ...")
    start_time = time.time()
    with Pool(processes=threads) as p:
        morphs = p.map(fn_analyze_khaiii, range(0,len(playlist)))
    print("--- %s seconds ---" % (time.time()-start_time))
    return morphs
    
# final clean in morphs all
def fn_all_clean( morphs,train_data=train_data):
    for_w = train_data[['tags','genre']]
    for_w['plylst_title'] = morphs
    
    finals=[[]]
    for i in range(0,len(for_w)):
        if i == 0:
            finals[0] = for_w.iloc[0,0] + for_w.iloc[0,1] + for_w.iloc[0,2]
        else:
            finals.append(for_w.iloc[i,0]+for_w.iloc[i,1]+for_w.iloc[i,2])
            
    finals = fn_clean_data(finals)
    
    tmp_list = []
    for j in range(0,len(finals)):
        idx = np.where(np.array([len(i) for i in finals[j]])==1)
        if np.array(idx).shape[1]==0:
            continue
        else:
            tmp_list.append(np.array(finals[j])[idx])

    tmpp_list = []
    for i in range(0,len(tmp_list)):
        tmpp_list.append(' '.join(tmp_list[i]))
        
    tfidf = TfidfVectorizer(token_pattern = r"(?u)\b\w+\b")
    A_tfidf_sp = tfidf.fit_transform(tmpp_list)
    tfidf_dict = tfidf.get_feature_names()

    data_array = A_tfidf_sp.toarray()
    data = pd.DataFrame(data_array, columns=tfidf_dict)

    tfidf = data[data>0.97].dropna(how='all',axis=1).copy()
    
    non_in_list= []
    for i in data.columns:
        if i in tfidf.columns:
            pass
        else:
            non_in_list.append(i)
            
    for i in range(0,len(finals)):
        for j in range(0,len(finals[i])):
            if len(finals[i][j]) == 1:
                if finals[i][j] not in tfidf:
                    finals[i][j] =''

    for i in range(0,len(finals)):
        while '' in finals[i]:
            finals[i].remove('')
            
    return finals

#####################
##### valid_data #####
#####################

def clean_valid_data(valid_data=valid_data):

    prac = copy.deepcopy(valid_data[['tags','plylst_title','songs','updt_date','like_cnt']])
    prac.head()

    # fn_clean_str
    for i in range(0,len(prac)):
        mylist = [[prac['plylst_title'][i]]]
        clean = fn_clean_str(' '.join(sum(mylist,[])))
        if len(clean)==0:
            clean=' '
        prac['plylst_title'][i] = clean

    # delete blank
    for i in range(len(prac['plylst_title'])):
        prac['plylst_title'][i]=prac['plylst_title'][i].strip()
        
    return prac

prac = clean_valid_data()

def fn_morph_valid(i):
    tmp_list = []
    noun_list = []
    if len(prac['plylst_title'][i]) == prac['plylst_title'][i].count(' '):
        # 빈칸으로만 되어있는 행
        pass
    elif prac['plylst_title'][i].find(" ")==-1:
        # 띄어쓰기 없는 행
        tmp_list.append(prac['plylst_title'][i])
    else:
        for word in api.analyze(prac['plylst_title'][i]):
            for morph in word.morphs:
                # 실질형태소 가져오기
                if morph.tag in ['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ','SN','SL']:
                    tmp_list.append(morph.lex)
                if morph.tag in ['NNG','NNP','NP','SL']:
                    noun_list.append(morph.lex)

    if i%1000==0:
        time.sleep(5)
        
    return [tmp_list, noun_list]

def load_val_prac():
    print("Loading valid morphs ...")
    start_time = time.time()
    with Pool(processes=threads) as p:
        result= p.map(fn_morph_valid, range(0,len(prac)))
    print("--- %s seconds ---" % (time.time()-start_time))
    return result


def clean_all_valid(result,prac=prac):
### 영어 소문자로
    for i in range(len(result)):
        for j in range(len(result[i])):
            for z in range(len(result[i][j])):
                result[i][j][z] = result[i][j][z].lower()

    ### 영어 한글자 지우기
    p = re.compile('[a-z]')

    for i in range(len(result)):
        for j in range(len(result[i])):
            for z in range(len(result[i][j])):
                if len(result[i][j][z]) == 1:
                    if p.match(result[i][j][z]) != None:
                        result[i][j][z] = ''

    for i in range(len(result)):
        for j in range(len(result[i])):
            while '' in result[i][j]:
                result[i][j].remove('')

    ### 중복되는 단어 제거
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = list(set(result[i][j]))

    ### 숫자 4자리 제외 제거
    for i in range(len(result)):
        for j in range(len(result[i])):
            for z in range(len(result[i][j])):
                if result[i][j][z].isdigit() == True:
                    if len(result[i][j][z]) != 4:
                        result[i][j][z] = ''
                        
    for i in range(len(result)):
        for j in range(len(result[i])):
            while '' in result[i][j]:
                result[i][j].remove('')
                

    full_list = []
    noun_list = []
    for lst in result:
        full_list.append(lst[0])
        noun_list.append(lst[1])

    prac['plylst_title']=copy.deepcopy(full_list)
    prac['nouns']=copy.deepcopy(noun_list)
    pred_tag=[]
    pred_song=[]
    for i in range(len(prac)):
        pred_tag.append([])
    for i in range(len(prac)):
        pred_song.append([])
    prac['pred_song']=pred_song
    prac['pred_tag']=pred_tag
    prac['index'] = range(len(prac))
    prac.updt_date = list(map(fn_change_date, prac['updt_date'].tolist()))
    prac = prac[['index','tags','plylst_title','nouns','songs','updt_date','pred_tag','pred_song','like_cnt']]
    ### 들어있는 tag 영어 소문자로
    for i in range(0,len(prac['tags'])):
        for j in range(0, len(prac['tags'][i])):
            prac['tags'][i][j] = prac['tags'][i][j].lower()
        
            
    return prac

# # In[ ]:




