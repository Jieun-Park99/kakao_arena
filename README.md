# :headphones: Melon Playlist Continuation using Word2vec

###  2020 kakao arena
플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, __주어지지 않은 곡들과 태그를 예측__ 하는 것이 목표<br>
만약 플레이리스트에 들어있는 곡들의 절반을 보여주고, 나머지 숨겨진 절반을 예측할 수  있는 모델을 만든다면, 플레이리스트에 들어 있는 곡이 전부  주어졌을 때 이 모델이 해당 플레이리스트와 어울리는 곡들을 추천해 줄 것이라고 기대할 수 있다.

## :bulb: 전체적인 우리의 목표
어떠한 공간으로 모았을 때에 비슷한 단어는 근처에 모여있을 것이라 기대. `예) 새벽 -> 감성, 밤`<br>
우리는 여기서 word2vec처럼 같은 플레이리스트에 있는 형태소들이나 태그, 장르를 하나의 군집처럼 만든다면 해당 플레이 리스트에서 비슷한 다른 단어들을 추천받을 수 있을 것이다. 같은 비정형 데이터인 tag를 추천받을 때에 이를 활용하기로 했다.<br>
또한, song을 추천받을 때에도 같은 song_id를 w2v으로 학습시키면 위와 같은 결과를 얻을 것으로 기대했다.

### :diamonds: 0. 실행
### :diamonds: 0-1. 데이터 다운로드
https://arena.kakao.com/c/7/data 에 제공된 파일과 해당 레포지토리에 있는 파일들을 원하는 위치에 다운로드 받습니다. 이 위치를 kakao_arena라고 하겠습니다.<br>
그리고 `khaiii` 모듈의 설치가 필요합니다. <br>
윈도우는 지원하지 않으므로 `Microsoft app`에서 지원하는 `Ubuntu18.04`에서 작업하였습니다.<br>

#### :point_right: 실행환경
+ python3.6이상
+ khaiii를 위한 우분투 환경
+ RAM 64GB 이상
+ CPU Threads 25개 이상

```
├── kakao_arena 
   ├── khaiii (설치된 모듈)
   ├── Module_Data.py
   ├── preprocess.py
   ├── train.py
   ├── W2V_model.py
   ├── train.json 
   ├── val.json 
   ├── test.json 
   └── genre_gn_all.json
```

### :diamonds: 1. playlist title의 khaiii를 통한 형태소 분석
* khaiii 이전의 데이터 전처리를 해준다.<br>
한글 자음, 모음, 특수기호, 이모티콘, 유니코드 등을 제거해준다.

* khaiii 형태소분석을 통한 Tokenizing
형태소 분석기를 통해 __실질형태소__ 만을 추출하여 word2vec에 사용하기로 했다.<br>
단, `예) 겜할때듣는음악` 같이 띄어쓰기가 없는 경우 tag에 그대로 쓰이는 경우가 많다고 판단하여 이는 형태소분석 없이 그대로 사용한다.<br>

* 불용어 제거를 직접하는 것 보다 의미없이 많이 쓰이는 한글자들을 찾아서 제거한다. (TF-IDF)


### :diamonds: 2. 데이터 속 다양한 경우의 수
![image](https://user-images.githubusercontent.com/56948006/87875892-4dfd7d80-ca0f-11ea-99c8-85e148ce87f9.png)
### :diamonds: 2-1. Word2vec을 이용한 Tag 예측<br>
원래 tag로 쓰이던 것들이 똑같이 추천된다면 맞을 확률이 높다고 판단하여 train 데이터에 있는 모든 tag들을 uniq_tag로 둔다.<br>
word2vec결과에 한 글자보다는 두 글자 이상 대부분 명사인 유사도가 나오길 원해서 실질형태소가 아닌 명사를 input으로 넣는다.<br>
  + only_plylst: 명사와 영어 -> word2vec으로 나온 유사도 중에서 uniq_tag에 있는 것들 먼저 추천하고 명사와 유사도단어들을 통해 나머지 추천<br>
  + tag & plylst: 명사와 영어 -> uniq_tag에 겹치는 것들 먼저 추천 -> 명사와 영어, 유사도단어들을 통해 나머지 추천
  + only_tag: tag -> uniq_tag에 겹치는 것들 먼저 추천 -> 나머지는 유사도단어들로 추천
  + tag & song: only_tag와 동일
  + only_song: 노래들의 장르명을 word2vec에 넣고 유사도로 추천
  

### :diamonds: 2-2. Word2vec을 이용한 Song 예측<br>
비정형 데이터만 있다면 tag가 모여있어서 추천받는 것처럼 tag와 plylist도 연관이 있으니 같은 방법으로 추천받을 수 있다.
  + only_plylst: plylst 실질형태소와 word2vec의 결과 유사도 1등을 사용하여 title과 tag가 하나라도 겹치는 플레이리스트의 song들을 대상으로 빈도수 top 100 추천
  + plylst_tag: only_plylst와 같은 방법으로 추천 (w2v의 input만 실질형태소와 tag)
  + only_tag: only_plylst와 같은 방법으로 추천 (w2c의 input만 tag)
  + only_song: train데이터의 song들을 w2v으로 학습하고 추천
  + tag & song: train데이터의 song들을 w2v으로 학습하고 추천
  
  
### :diamonds: 2-3. 마지막 점검을 통한 song 100곡, tag 10개 추천받기
위의 과정을 모두 거치고 난 후에 최종 점검을 통해 nothing에 해당하는 부분을 채운다.
  + nothing: 새로운 데이터의 플레이리스트의 업데이트 날짜 이전의 인기곡(train에서 받아옴)들로 추천
  
### :raised_hand: 보완하고 싶은 점
`Autoencoder`을 통해 tag&song을 학습시키고 추천받고 싶었으나 성능이 w2v에 비해 안좋아서 쓸 수 없었다.<br>
기회가 된다면 Autoencoder를 다시 한 번 공부해서 제대로 사용해보고 싶다.
