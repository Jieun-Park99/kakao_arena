#!/usr/bin/env python
# coding: utf-8

# In[4]:


from collections import Counter
import time
import numpy as np
import pandas as pd

from gensim.models import Word2Vec

# In[5]:


def fn_only_play(lst,model_tag,uniq_tag,length,morphs,train_data):
    #for lst in data:
    # similarity 단어들
    words = []
    simi = model_tag.predict_output_word(lst[3],topn=30)

    # 유사도가 안나오니까 있는 명사만 넣자
    if simi == None:
        lst[6] = lst[3]

    # 유사도가 있는 경우
    else:

        for wr in simi:
            words.append(wr[0])


        # 유니크 태그(train)와 겹치는 similarity 단어들 / 겹치지 않는 단어들
        intr=[]
        nointr=[]
        for i in range(len(words)):
            if words[i] in uniq_tag:
                intr.append(words[i])
            else:
                nointr.append(words[i])

        # 최대 5개로 맞춤
        for i in range(len(intr)):
            if len(lst[6]) == 5:
                break
            else:
                if intr[i] not in lst[1]:
                    lst[6].append(intr[i])


        # 나머지는 similarity로 5개 맞추기
        if len(lst[6]) < 5:            
            for i in range(5-len(intr)):
                if nointr[i] not in lst[1] and len(nointr[i]) >= 2:
                    lst[6].append(nointr[i])

        ###### 여기까지 5개씩 다 채움 #######
        # nouns 넣어주기(혹시 모르니까 길이가 긴 순서대로 넣어줌)

        lst[3].sort(key=len, reverse=True)

        if len(lst[6]) < 10:
            for i in range(len(lst[3])):
                if len(lst[6]) == 10:
                    break
                else:
                    if lst[3][i] in lst[6]:
                        pass
                    else:
                        if len(lst[3][i]) >= 2:
                            lst[6].append(lst[3][i])

        # words에서 안쓰인 것들 채우기 (한글자 제외)

        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6]) == 10:
                    break
                else:
                    if words[i] not in lst[6] and len(words[i])>=2:
                        lst[6].append(words[i])

        # 이래도 10개가 안찬다면 남은 것들 다 넣기
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6])==10:
                    break
                else:
                    if words[i] not in lst[6]:
                        lst[6].append(words[i])

    update = lst[5]
    # 유사도 1순위
    b = model_tag.predict_output_word(lst[2])

    # 유사도 1순위 없으면 song 빈 리스트로 남김
    if b==None:
        print('No vocab')
        lst[4] = []

    # 1순위 있을경우에는 pred_plylist 만들어줌
    else:
        start = time.time()
        a = b[0][0]
        print("#--------- doing "+str(lst[0])+"th"+" of "+str(length)+" ------------#")
        pred_plylist = sum([lst[2],[a]],[])

        #하나라도 만족하는 경우의 song들을 cand_songs에 append 하기
        cand_songs =[]
        for i in range(0,len(morphs)):
            if (any(x in pred_plylist for x in morphs[i])):
                #valid updt_date보다 이전의 playlist에서 찾아야함
                if(train_data.updt_date[i] <= lst[5]):
                    show_sum = sum([x in pred_plylist for x in morphs[i]])
                    tmpp = [[x,show_sum] for x in train_data.songs[i]]
                    cand_songs.append(tmpp)

        # 모든 노래 후보
        cand_df = pd.DataFrame([y for x in cand_songs for y in x],columns=['song_num',"show_sum"]).sort_values(by='show_sum',ascending=False)
        sum_max = cand_df.groupby("song_num").max().sort_values("show_sum",ascending=False)

        # 빈도수 상위 100개 뽑아서 song에 넣기
        print("#----------- doing unlist -------------#")
        c = Counter(cand_df.song_num)
        count_df = pd.DataFrame(data=c.values(),index=c.keys(),columns=["count"])
        # 평균이 8이라서 count가 8 이상인 것만 뽑기
        count_df = count_df[count_df["count"] > 8]
        lst[7] = list(count_df.sort_values(by="count",ascending=False).iloc[0:100,:].index.values)
        print("#--------- Finished "+str(lst[0])+"th"+" took "+str((round(time.time()-start)/60))+"mins ------------#")
        


# In[6]:


def fn_tag_play(lst,model_tag,uniq_tag,length,morphs,train_data):
    # similarity 단어들
    
    simi = model_tag.predict_output_word(list(set(lst[1]+lst[3])),topn=30)
    
    # 유사도가 안나오면 비워두기
    if simi == None:
        for i in range(len(lst[3])):
            if lst[3][i] not in lst[1]:
                lst[6].append(lst[3][i])
    # 유사도가 있는 경우
    else:
        words = []
        for wr in simi:
            words.append(wr[0])
    
         
        # 유니크 태그(train)와 겹치는 similarity 단어들 / 겹치지 않는 단어들
        intr=[]
        nointr=[]
        for i in range(len(words)):
            if words[i] in uniq_tag:
                intr.append(words[i])
            else:
                nointr.append(words[i])
    
        
        # intr에서 5개 채우기
        for i in range(len(intr)):
            if len(lst[6]) == 5:
                break
            else:
                if intr[i] not in lst[1]:
                    lst[6].append(intr[i])
        
        # intr이 5개가 안된다면 nointr로 5개 맞추기 (한 글자 제외)
        if len(intr) < 5:
            for i in range(5-len(intr)):
                if nointr[i] not in lst[1] and len(nointr[i]) >=2:
                    lst[6].append(nointr[i])
        
        ##### 여기까지 word2vec으로 거의 5개씩 채움 #####
        
        ## 이제 nouns로 채워주기 (길이 긴 순서대로)
        lst[3].sort(key=len, reverse=True)
        
        if len(lst[6]) < 10:
            for i in range(len(lst[3])):
                if len(lst[6])==10:
                    break
                else:
                    if lst[3][i] not in lst[1]+lst[6] and len(lst[3][i])>=2:
                        lst[6].append(lst[3][i])

        ## 만약 아직 다 차지 않았다면 words에서 쓰이지 않은 것들로 채우기 (한 글자 제외)
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6]) == 10:
                    break
                else:
                    if words[i] not in lst[1]+lst[6] and len(words[i])>=2:
                        lst[6].append(words[i])

        ## 이래도 차지 않는다면 남는 것 다 넣기
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6])==10:
                    break
                else:
                    if words[i] not in lst[1]+lst[6]:
                        lst[1].append(words[i])
        
    update = lst[5]
    #태그 + plylist
    all_nouns = list(set(sum([lst[2],lst[1]],[])))
    #유사도 1순위
    b = model_tag.predict_output_word(all_nouns)
    
    # 유사도 1순위 없으면 song 빈 리스트로 남김
    if b==None:
        print('No vocab')
        lst[4] = []
        
    # 1순위 있을경우에는 pred_plylist 만들어줌
    else:
        start = time.time()
        a = np.array(b)[np.where(np.array([len(b[x][0]) for x in range(0,len(b))])>1)][0][0]
        print("#--------- doing "+str(lst[0])+"th"+" of "+str(length)+" ------------#")
        pred_plylist = sum([all_nouns,[a]],[])
        
        #하나라도 만족하는 경우의 song들을 cand_songs에 append 하기
        cand_songs =[]
        for i in range(0,len(morphs)):
            if (any(x in pred_plylist for x in morphs[i])):
                #valid updt_date보다 이전의 playlist에서 찾아야함
                if(train_data.updt_date[i] <= update):
                    show_sum = sum([x in pred_plylist for x in morphs[i]])
                    tmpp = [[x,show_sum] for x in train_data.songs[i]]
                    cand_songs.append(tmpp)
        
        # 모든 노래 후보
        cand_df = pd.DataFrame([y for x in cand_songs for y in x],columns=['song_num',"show_sum"]).sort_values(by='show_sum',ascending=False)
        sum_max = cand_df.groupby("song_num").max().sort_values("show_sum",ascending=False)
        
        # 빈도수 상위 100개 뽑아서 song에 넣기
        print("#----------- doing unlist -------------#")
        c = Counter(cand_df.song_num)
        count_df = pd.DataFrame(data=c.values(),index=c.keys(),columns=["count"])
        # 평균이 8이라서 count가 8 이상인 것만 뽑기
        count_df = count_df[count_df["count"] > 8]
        lst[7] = list(count_df.sort_values(by="count",ascending=False).iloc[0:100,:].index.values)
        print("#--------- Finished "+str(lst[0])+"th"+" took "+str((round(time.time()-start)/60))+"mins ------------#")
 


# In[7]:


def fn_only_tag(lst,model_tag,uniq_tag,length,morphs,train_data):
    # similarity 단어들
    
    simi = model_tag.predict_output_word(lst[1],topn=30)
    
    # 유사도가 안나오면 비워두기
    if simi == None:
        pass
    # 유사도가 있는 경우
    else:
        words = []
        for wr in simi:
            words.append(wr[0])
    
         
        # 유니크 태그(train)와 겹치는 similarity 단어들 / 겹치지 않는 단어들
        intr=[]
        nointr=[]
        for i in range(len(words)):
            if words[i] in uniq_tag:
                intr.append(words[i])
            else:
                nointr.append(words[i])
    
        
        # intr에서 5개 채우기
        for i in range(len(intr)):
            if len(lst[6]) == 5:
                break
            else:
                if intr[i] not in lst[1]:
                    lst[6].append(intr[i])
        
        # intr이 5개가 안된다면 nointr로 5개 맞추기 (한 글자 제외)
        if len(intr) < 5:
            for i in range(5-len(intr)):
                if nointr[i] not in lst[1] and len(nointr[i]) >=2:
                    lst[6].append(nointr[i])
        
        
        ## 만약 아직 다 차지 않았다면 words에서 쓰이지 않은 것들로 채우기 (한 글자 제외)
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6]) == 10:
                    break
                else:
                    if words[i] not in lst[1]+lst[6] and len(words[i])>=2:
                        lst[6].append(words[i])

        ## 이래도 차지 않는다면 남는 것 다 넣기
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6])==10:
                    break
                else:
                    if words[i] not in lst[1]+lst[6]:
                        lst[1].append(words[i])
        
    update = lst[5]
    #태그 + plylist
    all_nouns = lst[1]
    #유사도 1순위
    b = model_tag.predict_output_word(all_nouns)
    
    # 유사도 1순위 없으면 song 빈 리스트로 남김
    if b==None:
        print('No vocab')
        lst[4] = []
        
    # 1순위 있을경우에는 pred_plylist 만들어줌
    else:
        start = time.time()
        a = np.array(b)[np.where(np.array([len(b[x][0]) for x in range(0,len(b))])>1)][0][0]
        print("#--------- doing "+str(lst[0])+"th"+" of "+str(length)+" ------------#")
        pred_plylist = sum([all_nouns,[a]],[])
        
        #하나라도 만족하는 경우의 song들을 cand_songs에 append 하기
        cand_songs =[]
        for i in range(0,len(morphs)):
            if (any(x in pred_plylist for x in morphs[i])):
                #valid updt_date보다 이전의 playlist에서 찾아야함
                if(train_data.updt_date[i] <= update):
                    show_sum = sum([x in pred_plylist for x in morphs[i]])
                    tmpp = [[x,show_sum] for x in train_data.songs[i]]
                    cand_songs.append(tmpp)
        
        # 모든 노래 후보
        cand_df = pd.DataFrame([y for x in cand_songs for y in x],columns=['song_num',"show_sum"]).sort_values(by='show_sum',ascending=False)
        sum_max = cand_df.groupby("song_num").max().sort_values("show_sum",ascending=False)
        
        # 빈도수 상위 100개 뽑아서 song에 넣기
        print("#----------- doing unlist -------------#")
        c = Counter(cand_df.song_num)
        count_df = pd.DataFrame(data=c.values(),index=c.keys(),columns=["count"])
        # 평균이 8이라서 count가 8 이상인 것만 뽑기
        count_df = count_df[count_df["count"] > 8]
        lst[7] = list(count_df.sort_values(by="count",ascending=False).iloc[0:100,:].index.values)
        print("#--------- Finished "+str(lst[0])+"th"+" took "+str((round(time.time()-start)/60))+"mins ------------#")


# In[8]:


def fn_tag_song(lst,model_tag,uniq_tag,length,morphs,train_data):

    # similarity 단어들
    
    simi = model_tag.predict_output_word(lst[1],topn=30)
    
    # 유사도가 안나오면 비워두기
    if simi == None:
        pass
    # 유사도가 있는 경우
    else:
        words = []
        for wr in simi:
            words.append(wr[0])
    
         
        # 유니크 태그(train)와 겹치는 similarity 단어들 / 겹치지 않는 단어들
        intr=[]
        nointr=[]
        for i in range(len(words)):
            if words[i] in uniq_tag:
                intr.append(words[i])
            else:
                nointr.append(words[i])
    
        
        # intr에서 5개 채우기
        for i in range(len(intr)):
            if len(lst[6]) == 5:
                break
            else:
                if intr[i] not in lst[1]:
                    lst[6].append(intr[i])
        
        # intr이 5개가 안된다면 nointr로 5개 맞추기 (한 글자 제외)
        if len(intr) < 5:
            for i in range(5-len(intr)):
                if nointr[i] not in lst[1] and len(nointr[i]) >=2:
                    lst[6].append(nointr[i])
        
        ## 만약 아직 다 차지 않았다면 words에서 쓰이지 않은 것들로 채우기 (한 글자 제외)
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6]) == 10:
                    break
                else:
                    if words[i] not in lst[1]+lst[6] and len(words[i])>=2:
                        lst[6].append(words[i])
        
        ## 이래도 차지 않는다면 남는 것 다 넣기
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6])==10:
                    break
                else:
                    if words[i] not in lst[1]+lst[6]:
                        lst[1].append(words[i])
        
    print("#------ Finished ",lst[0]," th of ", length ," --------#")


# In[9]:


def fn_only_song(lst,model_tag,for_genre,uniq_tag,genre_gn_all,length,train_data):

    genre =[]
    # 장르 코드로 받아오기
    for i in range(len(lst[4])):
        genre.extend(for_genre[for_genre['id'].isin([lst[4][i]])].iloc[0,0])
    genre = list(set(genre))
    
    # 장르 코드를 장르명으로 바꾸기
    for i in range(len(genre)):
        genre[i] = genre[i].replace(genre[i], genre_gn_all[genre_gn_all['gnr_code'].isin([genre[i]])].iloc[0,1])
    
    # similarity 구하기
    simi = model_tag.predict_output_word(genre,topn=30)
    
    # 유사도가 안나오면 비워두기
    if simi == None:
        pass
    
    # 유사도가 있는 경우
    else:
        words = []
        for wr in simi:
            words.append(wr[0])
        
        # 유니크 태그(train)와 겹치는 similarity 단어들 / 겹치지 않는 단어들
        intr=[]
        nointr=[]
        for i in range(len(words)):
            if words[i] in uniq_tag:
                intr.append(words[i])
            else:
                nointr.append(words[i])
    
        # intr에서 5개 채우기
        for i in range(len(intr)):
            if len(lst[6]) == 5:
                break
            else:
                lst[6].append(intr[i])
        
        # intr이 5개가 안된다면 nointr로 5개 맞추기 (한 글자 제외)
        if len(intr) < 5:
            for i in range(5-len(intr)):
                if len(nointr[i]) >=2:
                    lst[6].append(nointr[i])
        
        ## 만약 아직 다 차지 않았다면 words에서 쓰이지 않은 것들로 채우기 (한 글자 제외)
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6]) == 10:
                    break
                else:
                    if words[i] not in lst[6] and len(words[i])>=2:
                        lst[6].append(words[i])
        
        ## 이래도 차지 않는다면 남는 것 다 넣기
        if len(lst[6]) < 10:
            for i in range(len(words)):
                if len(lst[6])==10:
                    break
                else:
                    if words[i] not in lst[6]:
                        lst[6].append(words[i])
        
    print("#------- Finished ",lst[0]," th of ",length," data -------#")

