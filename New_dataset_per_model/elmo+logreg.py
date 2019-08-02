
# coding: utf-8

# In[4]:

import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pandas as pd
import string
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
from sklearn import metrics
import json
import csv
import random
from sklearn.linear_model import LogisticRegression



# In[6]:


with open('testing_20%.json','r') as file:
    Test_X=json.load(file)
with open('testing_20%_labels.json','r') as file:
    Test_Y=json.load(file)
with open('training_80%.json','r') as file:
    Train_X=json.load(file)
with open('training_80%_labels.json','r') as file:
    Train_Y=json.load(file)
with open('trainx_emb_80%.json','r') as f:
    Train_X_emb=json.load(f)
with open('testx_emb_20%.json','r') as f:
    Test_X_emb=json.load(f)


# In[7]:


with open('scraper_posts.json','r') as f:
    data=json.load(f)


# In[8]:


new_posts_emb=[list(i) for i in data['emb']]
new_posts=data['text']


# In[9]:


interval=0.9
reg=LogisticRegression(class_weight='balanced')
reg.fit(Train_X_emb,Train_Y)
dec_function=reg.decision_function(new_posts_emb)
positive_emb=[new_posts_emb[i] for i in range(len(new_posts_emb)) if dec_function[i]>interval]
negative_emb=[new_posts_emb[i] for i in range(len(new_posts_emb)) if dec_function[i]<-interval]
print(len(new_posts_emb))
print(len(positive_emb), len(negative_emb))


# In[10]:


new_dataset=[[i,1] for i in positive_emb]+[[i,0] for i in negative_emb]+[[Train_X_emb[i],Train_Y[i]] for i in range(len(Train_Y))]
random.shuffle(new_dataset)


# In[11]:


reg=LogisticRegression(class_weight='balanced')
reg.fit([i[0] for i in new_dataset],[i[1] for i in new_dataset])
pred=reg.predict(Test_X_emb)
print("Log Reg Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

