#!/usr/bin/env python
# coding: utf-8

# # 0. Load previous Data and Module

# In[1]:


import W2V_model as w2v
import copy
print("W2V.py Loaded ...")

from Module_Data import genre_gn_all
import fn_file as fn

import itertools
import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import re
import json
import os
import distutils.dir_util
import io

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


# In[25]:


every_thing=[]
for i in range(0,len(w2v.prac)):
    every_thing.append(w2v.prac.iloc[i,:].tolist())
    
uniq_tag = list(set(itertools.chain(*w2v.preprocess.train_data.tags)))
for_genre = copy.deepcopy(w2v.preprocess.song_meta[['song_gn_dtl_gnr_basket','id']])


# In[28]:


def do_work(i):
    try:
        lst = every_thing[i]

        tag, song = [], []

        if len(lst[1]) == 0 and len(lst[2]) !=0 and len(lst[4])==0:
            tag, song = fn.fn_only_play(lst,w2v.model_tag,uniq_tag,w2v.morphs,w2v.preprocess.train_data)
        elif len(lst[1]) != 0 and len(lst[2]) !=0 and len(lst[4])==0:
            tag, song = fn.fn_tag_play(lst,w2v.model_tag,uniq_tag,w2v.morphs,w2v.preprocess.train_data)
        elif len(lst[1]) != 0 and len(lst[2]) ==0 and len(lst[4])==0:
            tag, song = fn.fn_only_tag(lst,w2v.model_tag,uniq_tag,w2v.morphs,w2v.preprocess.train_data)
        elif len(lst[1]) != 0 and len(lst[2]) ==0 and len(lst[4])!=0:
            tag, song = fn.fn_tag_song(lst,w2v.model_tag,w2v.model_song,uniq_tag,w2v.morphs,w2v.preprocess.train_data,w2v.preprocess.song_meta)
        elif len(lst[1]) == 0 and len(lst[2]) ==0 and len(lst[4])!=0:
            tag, song = fn.fn_only_song(lst,w2v.model_tag,w2v.model_song,for_genre,uniq_tag,genre_gn_all,w2v.preprocess.train_data,w2v.preprocess.song_meta)
            #print(i)

        if i%1000==0:
            time.sleep(5)

        return tag,song
    
    except Exception as ex:
        print(ex,'at ----> ',i,'\n')



if __name__ == '__main__':
    start_time = time.time()
    print("Predicting the test data ...")
    with Pool(processes= w2v.preprocess.threads) as p:
        result = p.map(do_work, range(0,len(every_thing)))
    print("--- %s seconds ---" % (time.time()-start_time))


# In[30]:


w2v.prac.iloc[:,6] = pd.DataFrame(result).iloc[:,0]
w2v.prac.iloc[:,7] = pd.DataFrame(result).iloc[:,1]


# In[31]:


# Modify none type(tag)
for i in range(len(w2v.prac)):
    if w2v.prac.iloc[i,6] is None:
        w2v.prac.iloc[i,6] = [[0]]

for i in range(len(w2v.prac)):
    if w2v.prac.iloc[i,6] == [0]:
        del w2v.prac.iloc[i,6][-1]


# In[32]:


# Modify none type(plylist title)
for i in range(len(w2v.prac)):
    if w2v.prac.iloc[i,7] is None:
        w2v.prac.iloc[i,7] = [[0]]

for i in range(len(w2v.prac)):
    if w2v.prac.iloc[i,7] == [0]:
        del w2v.prac.iloc[i,7][-1]


# In[43]:


print("Doing final predicting ...")
plylst_tag_map = w2v.preprocess.train_data[['id', 'tags']]

# unnest tags
plylst_tag_map_unnest = np.dstack(
    (
        np.repeat(plylst_tag_map.id.values, list(map(len, plylst_tag_map.tags))), 
        np.concatenate(plylst_tag_map.tags.values)
    )
)

# unnested 데이터프레임 생성 : plylst_tag_map
plylst_tag_map = pd.DataFrame(data = plylst_tag_map_unnest[0], columns = plylst_tag_map.columns)
plylst_tag_map['id'] = plylst_tag_map['id'].astype(str)

# unnest 객체 제거
del plylst_tag_map_unnest


plylst_tag_list_sort = plylst_tag_map.sort_values(by = ['id', 'tags']).groupby('id').tags.apply(list).reset_index(name = 'tag_list')

# 1. unnest 데이터프레임인 plylst_tag_map 테이블에서 태그 이름 정렬 후 list로 묶기
plylst_tag_list_sort = plylst_tag_map.sort_values(by = ['id', 'tags']).groupby('id').tags.apply(list).reset_index(name = 'tag_list')

# 2. 집계를 위해 1번 테이블에서 list 타입을 문자열 타입으로 변경
plylst_tag_list_sort['tag_list'] = plylst_tag_list_sort['tag_list'].astype(str)

# 3. 태그 리스트 별 매핑되는 플레이리스트 수 집계 테이블 생성 : tag_list_plylst_cnt
tag_list_plylst_cnt = plylst_tag_list_sort.groupby('tag_list').id.nunique().reset_index(name = 'plylst_cnt')

# 4. 매핑 수 기준 상위 10개 필터링
tag_list_plylst_cnt = tag_list_plylst_cnt.nlargest(10, 'plylst_cnt')

# 상위 top10 태그 리스트 만들기
top10_tags=[]
for i in range(len(np.array(tag_list_plylst_cnt['tag_list']))):
    top10_tags.append(re.compile('[가-힣]+').findall(np.array(tag_list_plylst_cnt['tag_list'])[i]))

top10_tags = list(itertools.chain(*top10_tags))


for i in range(len(w2v.prac)):
    if len(w2v.prac.iloc[i,:][6]) < 10:
        for j in range(len(top10_tags)):
            if len(w2v.prac.iloc[i,:][6]) == 10:
                break
            else:
                if top10_tags[j] not in w2v.prac.iloc[i,:][1] + w2v.prac.iloc[i,:][6]:
                    w2v.prac.iloc[i,:][6].append(top10_tags[j])


# In[72]:


tmp = [y for x in w2v.preprocess.train_data["songs"] for y in x]

from collections import Counter
pp = pd.DataFrame(data=Counter(tmp).values(),index=Counter(tmp).keys(),columns=["count"]).sort_values(by=['count'],ascending=False)

tmp_list = []
for i in pp.index[0:1000]:
    tmp_list.append(w2v.preprocess.song_meta["song_name"][w2v.preprocess.song_meta["id"]==i])

from dateutil.parser import parse


for i in range(len(w2v.prac)):
    for j in range(len(tmp_list)):
        if len(w2v.prac_copy.pred_song[i]) == 100:
            break
        elif parse(w2v.prac_copy.iloc[i,5]) > parse(w2v.preprocess.song_meta["issue_date"][tmp_list[j].index[0]]):
            w2v.prac_copy.pred_song[i].append(tmp_list[j].index[0])


# In[79]:


valid_final = pd.DataFrame([w2v.preprocess.valid_data.id,w2v.prac.pred_song,w2v.prac.pred_tag])
valid_final = valid_final.rename(index={'id':'id','pred_song':'songs','pred_tag':'tags'})

valid_list = []
for i in range(0,len(valid_final.columns)):
    valid_list.append(valid_final.iloc[:,i].to_dict())
        
write_json(valid_list, 'results.json')


# In[48]:


for i in range(0,len(w2v.prac)):
    if len(w2v.prac.iloc[i,6])!=0:
        if len(w2v.prac.iloc[i,6])!=10:
            print(i,'yes')

