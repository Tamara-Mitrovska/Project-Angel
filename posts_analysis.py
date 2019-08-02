
# coding: utf-8

# In[2]:


import csv
import json
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import nltk


# In[4]:

with open('final_15K_dataset.json','r') as file:
    data=json.load(file)
positive_posts=[data['text'][i] for i in range(len(data['text'])) if data['label'][i]==1]

# In[5]:


#top pronouns used
pronouns=[]
tokens=[]
len_posts=[]
nouns=[]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for entry in positive_posts:
    len_posts.append(len(entry.split(' ')))
    word_lemmatized = WordNetLemmatizer()
    entry=entry.lower()
    entry=word_tokenize(entry)
    for word, tag in pos_tag(entry):
        if tag=='PRP' or tag=='PRP$':
            pronouns.append(word)
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word,tag_map[tag[0]])
            tokens.append(word_final)
            if tag=='NN' or tag=='NNS':
                nouns.append(word_final)
            


# In[11]:


fd_nouns=nltk.FreqDist(nouns)
print(fd_nouns.most_common(20))


# In[7]:


fdist_pronouns=nltk.FreqDist(pronouns)
print(fdist_pronouns.most_common(20))


# In[8]:


fdist_unigrams=nltk.FreqDist(tokens)
print(fdist_unigrams.most_common(20))


# In[9]:


bigrams=nltk.bigrams(tokens)
fdist = nltk.FreqDist(bigrams)
print (fdist.most_common(20))


# In[10]:


print('Average post length: '+str(sum(len_posts)/len(len_posts)))
print('Min post length: '+str(min(len_posts)))
print('Max post length: '+str(max(len_posts)))
