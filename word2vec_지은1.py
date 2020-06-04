# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:08:12 2020

@author: danan
"""
import pandas as pd

train = pd.read_json('C:/Users/danan/Desktop/Projects/kakao arena/data/train.json',encoding='UTF-8', typ = 'frame')
train_tag = train[['id','tags']]

train_tag
     
from gensim.models import Word2Vec
model_tag = Word2Vec(sentences=train_tag.tags,
                 size=1000, window=5,
                 min_count=5,
                 workers=4,
                 sg=1)

model_result = model_tag.wv.most_similar("크리스마스")
print(model_result)

# 단어 벡터를 구한다.
word_vectors = model_tag.wv
vocabs = word_vectors.vocab.keys()
vocabs
word_vectors_list = [word_vectors[v] for v in vocabs]

word_vectors_list.head(3)
# 단어간 유사도를 확인하다
print(model_tag.wv.similarity(w1='새벽',w2='사랑'))
model_tag.save('tag_w2v.model')


# playlist title Word2Vec
train.columns

train_play = train[['id','plylst_title']]
train_play.plylst_title

# 플레이리스트 제목 txt로 저장
with open("C:/Users/danan/Desktop/Projects/kakao arena/data/train_play.txt",'w',encoding='utf-8') as f:
    for sent in train_play.plylst_title:
        f.write('{}\n'.format(sent))



# 모델 hyperparameter 설정
templates= "--input={} \
--pad_id={} \
--bos_id={} \
--eos_id={} \
--unk_id={} \
--model_prefix={} \
--vocab_size={} \
--character_coverage={} \
--model_type={} \
"


train_input_file = "train_play.txt"
pad_id=0  #<pad> token을 0으로 설정
vocab_size = 3000 # vocab 사이즈
prefix = 'playlist_spm' # 저장될 tokenizer 모델에 붙는 이름
bos_id=1 #<start> token을 1으로 설정
eos_id=2 #<end> token을 2으로 설정
unk_id=3 #<unknown> token을 3으로 설정
character_coverage = 1.0 # to reduce character set 
model_type ='word' # Choose from unigram (default), bpe, char, or word


cmd = templates.format(train_input_file,
                pad_id,
                bos_id,
                eos_id,
                unk_id,
                prefix,
                vocab_size,
                character_coverage,
                model_type)

cmd    

# Tokenizer 학습
import sentencepiece as spm
spm.SentencePieceTrainer.Train(cmd)

# 문장에 Tokenizer 적용
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
#sp.SetEncodeExtraOptions('bos:eos')
sp.Load('playlist_spm.model') #prefix 이름으로 저장된 모델

# 토큰화
tokens = [ ]
for i in range(len(train_play.plylst_title)):
    a = sp.EncodeAsPieces(train_play.plylst_title[i])
    tokens.append(a)
    
tokens

# word2vec with playlist_title
from gensim.models import Word2Vec
model_play = Word2Vec(sentences=tokens,
                 size=1000, window=5,
                 min_count=5,
                 workers=4,
                 sg=1)

model_result = model_play.wv.most_similar("새벽")
print(model_result)

# 단어 벡터를 구한다.
word_vectors = model_play.wv
word_vectors
vocabs = word_vectors.vocab.keys()
vocabs
word_vectors_list = [word_vectors[v] for v in vocabs]

print(word_vectors.similarity(w1='새벽',w2='사랑'))
model_play.save('play_w2v.model')

