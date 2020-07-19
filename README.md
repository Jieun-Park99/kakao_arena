# :musical_score: Melon Playlist Continuation using Word2vec & Autoencoder

###  2020 kakao arena
플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, __주어지지 않은 곡들과 태그를 예측__ 하는 것이 목표
만약 플레이리스트에 들어있는 곡들의 절반을 보여주고, 나머지 숨겨진 절반을 예측할 수  있는 모델을 만든다면, 플레이리스트에 들어 있는 곡이 전부  주어졌을 때 이 모델이 해당 플레이리스트와 어울리는 곡들을 추천해 줄 것이라고 기대할 수 있다.

## :bulb: 전체적인 우리의 목표
어떠한 공간으로 모았을 때에 비슷한 단어는 근처에 모여있을 것이라 기대. `예) 새벽 -> 감성, 밤`<br>
우리는 여기서 word2vec처럼 같은 플레이리스트에 있는 형태소들이나 태그, 장르를 하나의 군집처럼 만든다면 해당 플레이 리스트에서 비슷한 다른 단어들을 추천받을 수 있을 것이다. 같은 비정형 데이터인 tag를 추천받을 때에 이를 활용하기로 했다.<br>
또한, song을 추천받을 때에는 autoencoder를 통해 갖고있는 데이터를 갖고 차원축소와 복원을 통해 새로운 곡들을 추천받는다.
이때, one-hot-encoding방식으로 input을 넣으면 데이터가 지나치게 커져서 song id를 그대로 사용하고 추출된 특징인 latent layer을 사용하여 축소된 공간에서 군집으로 모여있다는 것을 알고 이를 통한 추천을 받기로 한다.<br>
조금 더 자세하게 아래에서 설명


### :diamonds: 1. playlist title의 khaiii를 통한 형태소 분석
* khaiii 이전의 데이터 전처리를 해준다.<br>
한글 자음, 모음, 특수기호, 이모티콘, 유니코드 등을 제거해준다.

* khaiii 형태소분석을 통한 Tokenizing
형태소 분석기를 통해 __실질형태소__ 만을 추출하여 word2vec에 사용하기로 했다.<br>
단, `예) 겜할때듣는음악` 같이 띄어쓰기가 없는 경우 tag에 그대로 쓰이는 경우가 많다고 판단하여 이는 형태소분석 없이 그대로 사용<br>

* 불용어 제거를 직접하는 것 보다 의미없이 많이 쓰이는 한글자들을 찾아서 제거 (TF-IDF)


### :diamonds: 2. 데이터 속 다양한 경우의 수
![image](https://user-images.githubusercontent.com/56948006/87875892-4dfd7d80-ca0f-11ea-99c8-85e148ce87f9.png)
### :diamonds: 2-1. Word2vec을 이용한 Tag 예측<br>
train 데이터에 있는 모든 tag들을 uniq_tag로 둔다. 원래 tag로 쓰이던 것들이 똑같이 추천된다면 맞을 확률이 높다고 판단
  + only_plylst: 명사와 영어 -> word2vec으로 나온 유사도 중에서 uniq_tag에 있는 것들 먼저 추천하고 명사와 유사도단어들을 통해 나머지 추천<br>
  + tag & plylst: 명사와 영어 -> uniq_tag에 겹치는 것들 먼저 추천 -> 명사와 영어, 유사도단어들을 통해 나머지 추천
  + only_tag: tag -> uniq_tag에 겹치는 것들 먼저 추천 -> 나머지는 유사도단어들로 추천
  + tag & song: only_tag와 동일
  + only_song: 노래들의 장르명을 word2vec에 넣고 유사도로 추천
  + nothing: 
  

### :diamonds: 2-2. Word2vec을 이용한 Song 예측<br>
비정형 데이터만 있다면 tag가 모여있어서 추천받는 것처럼 tag와 plylist도 연관이 있으니 같은 방법으로 추천받을 수 있다고 판단
  + only_plylst: plylst 실질형태소와 word2vec의 결과 유사도 1등을 사용하여 title과 tag가 하나라도 겹치는 플레이리스트의 song들을 대상으로 빈도수 top 100 추천
  + plylst_tag: only_plylst와 같은 방법으로 추천 (w2v의 input만 실질형태소와 tag)
  + only_tag: only_plylst와 같은 방법으로 추천 (w2c의 input만 tag)


### :diamonds: 2-3. Autoencoder를 이용한 Song 예측<br>
데이터가 커서 one-hot-encoding을 쓸 수 없어 라벨인코딩을 통해 input데이터를 만들고 encoding과 decoding과정을 거쳐 모델을 만든다.
song id가 input이었기에 output을 정보로 이용할 수는 없다. decoder output이 아닌 latent space를 사용한다면 backpropagation도 input값에 가장 가까운 값이 되기 위해 진행되므로 모델의 가운데 hidden layer인 latent space를 사용하면 차원이 축소된 변수들의 군집이 형성되어 있음을 알 수 있다. 잘 만들어진 autoencoder의 latent space를 사용하여 latent vector 추출 후 가까운 거리의 playlist 추천받기

  + only_song: autoencoder
  + tag & song: autoencoder
