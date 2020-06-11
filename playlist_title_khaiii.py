# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:46:43 2020

@author: 최지은
"""

#%% 데이터 loading(playlist_title만)
with open('playlist.txt') as f:
    playlist = f.read().splitlines()
    
with open('preanal.txt') as f:
    data = f.read().splitlines()
    
#%% 명사 사전 만들기
import re
import itertools
nouns = []
# 모든 NNP/NNG/NP추출
for i in range(0,len(data)):
   tmp = list([item for item in data[i].replace('\t'," ").split(" ") if re.search('(NNP$)|(NNG$)|(NP$)', item) and (len(item)>=6)])
   if len(tmp)!=0:
       nouns.append(tmp)
#nouns unlist 진행
nouns = list(itertools.chain(*nouns))
nouns_final = list(set([re.sub("/NNG|/NNP|/NP","",item) for item in nouns])) #최종 preanal에 있는 unique 명사

#%% khaii import(설치는 알아서..ㅎ)
from khaiii import KhaiiiApi
api = KhaiiiApi()

#%% 형태소(명사)추출
#토큰화하기
morphs = []
tmp_list = []
for i in range(0,len(playlist)):
    if playlist[i].find(" ")==-1: #띄어쓰기 없는 것만 수동띄어쓰기
        for j in range(0,len(nouns_final)):
            playlist[i] = re.sub("(\w*)("+nouns_final[j]+")(\w+)|(\w+)("+nouns_final[j]+")(\s+)|(\w+)("+nouns_final[j]+")(\w+)",r"\1 \2 \3",playlist[i])
        print(playlist[i])
    if playlist[i].isspace():
        continue
    for word in api.analyze(playlist[i]):
        for morph in word.morphs:
            if morph.tag in ['NNG', 'NNP','NP']:
                tmp_list.append(morph.lex)
    morphs.append(tmp_list)
    tmp_list = []
    print(i)
    
# 이부분을 병렬처리 하면 됩니다.