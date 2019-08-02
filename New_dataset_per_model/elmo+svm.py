
# coding: utf-8

# In[1]:


import numpy as np
import os
from allennlp.commands.elmo import ElmoEmbedder
import pandas as pd
import string
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sys
from sklearn import metrics
import json
import csv
import random


# In[2]:


#os.getcwd()
#os.chdir(r'C:\Users\tamar\OneDrive\Desktop\Classifier')


# In[3]:


#random.seed(500)


# In[3]:


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


# In[11]:


with open('scraper_posts.json','r') as file:
    scraper_posts=json.load(file)
new_posts_emb=[list(i) for i in scraper_posts['emb']]
new_posts=scraper_posts['text']
print(len(new_posts_emb),len(new_posts))


# In[10]:


interval=0.9
SVM = svm.SVC(C=1, kernel='linear', degree=1, gamma=0.001,class_weight='balanced')
SVM.fit(Train_X_emb,Train_Y)
dec_function=SVM.decision_function(new_posts_emb)
pos_label_emb=[tuple(new_posts_emb[i]) for i in range(len(dec_function)) if dec_function[i]>interval ]
neg_label_emb=[tuple(new_posts_emb[i]) for i in range(len(dec_function)) if dec_function[i]<-interval ]
print(len(pos_label_emb),len(neg_label_emb))


# In[11]:


all_posts=[[tuple(Train_X_emb[i]),Train_Y[i]] for i in range(len(Train_X))]
all_posts+=[[pos_label_emb[i],1] for i in range(len(pos_label_emb))]
all_posts+=[[neg_label_emb[i],0] for i in range(len(neg_label_emb))]
random.shuffle(all_posts)


# In[12]:


trainx=[all_posts[i][0] for i in range(len(all_posts))]
trainy=[all_posts[i][1] for i in range(len(all_posts))]


# In[8]:


SVM = svm.SVC(C=1, kernel='linear', degree=1, gamma=0.001,class_weight='balanced')
SVM.fit(trainx,trainy)
predictions_SVM = SVM.predict(Test_X_emb)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(metrics.confusion_matrix(Test_Y, predictions_SVM))

