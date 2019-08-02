
# coding: utf-8

# In[2]:


import numpy as np
import os
from allennlp.commands.elmo import ElmoEmbedder
import pandas as pd
import string
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
from sklearn import metrics
import json


# In[3]:


os.getcwd()
os.chdir(r'C:\Users\tamar\OneDrive\Desktop\Classifier')


# In[4]:


def elmo_emb(elmo, max_words_sent,sent_str):
	sent = sent_str.split(' ')
	a = np.zeros(shape=(max_words_sent,3072))
	vectors = elmo.embed_sentence(sent)
	#print (vectors)
	#print ("done printing")
	v = min(len(sent), max_words_sent)
	for i in range(v):
		a[i] = np.concatenate((vectors[0][i], vectors[1][i], vectors[2][i]), axis=None)
	avg_words = np.mean(a[:v], axis=0) 	
	return avg_words


# In[5]:


elmo = ElmoEmbedder() 


# In[6]:


labeled_data=pd.read_csv('sp+Ip+sn+In.csv',encoding='latin-1')
with open('training_80%.json','r') as file:
    Train_X=json.load(file)
Test_X=[i for i in list(labeled_data['text']) if i not in Train_X]
posts=Test_X
clean_posts=[]
n_words={}
for post in posts:
    post=post.lower()
    for c in string.punctuation:
        post= post.replace(c,"")
    clean_posts.append(post)
    n_words[post]=len(post.split(' '))
    
max_words_sent=max(n_words.values())


# In[ ]:


print(len(clean_posts))
post_emb=[]
i=0
for sentence in clean_posts:
    sen_arr = elmo_emb(elmo, max_words_sent,sentence)
    post_emb.append(sen_arr)
    i+=1
    print(i)


# In[7]:


Test_Y=[list(labeled_data['label'])[i] for i in range(len(list(labeled_data['label']))) if list(labeled_data['text'])[i] in Test_X]
with open('testing_20%_labels.json','w') as file:
    json.dump(Test_Y,file)
#with open('testing_20%.json','w') as file:
 #   json.dump(Test_X,file)
#with open('testx_emb_20%.json','w') as file:
 #   json.dump([list(i) for i in post_emb],file)

