#!/usr/bin/env python
# coding: utf-8

# # 0. load Data and Module

# In[1]:


import preprocess

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import itertools


# In[2]:


morphs = preprocess.fn_all_clean(preprocess.load_morphs())
prac = preprocess.clean_all_valid(preprocess.load_val_prac())


# In[3]:


for_song = preprocess.train_data['songs'].tolist()

for i in range(len(for_song)):
    for_song[i] = list(map(str,for_song[i]))


# # 1. Word2Vec

# In[4]:


class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss


# In[ ]:

print("Making model_tag...")
model_tag = Word2Vec(sentences=morphs,
                 size=256, window=3,
                 min_count=2,
                 workers=1,
                 sg=1,
                 alpha=0.03,
                 iter=5,
                 compute_loss=True,
                 callbacks=[callback()],
                 seed=1234)


# In[12]:

print("Making model_song ...")
model_song = Word2Vec(sentences=for_song,
                 size=256, window=3,
                 min_count=2,
                 workers=1,
                 sg=1,
                 alpha=0.02,
                 iter=30,
                 compute_loss=True,
                 callbacks=[callback()],
                 seed=1234
                     )


# In[ ]:




