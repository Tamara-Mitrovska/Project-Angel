from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import json
import random
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
import pandas as pd
import string
import sys
from sklearn import metrics
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_X_y


# In[ ]:


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


# In[ ]:
#
#
with open('scraper_posts.json','r') as  f:
    scraped_posts=json.load(f)
scraped_posts_text=scraped_posts['text']
scraped_posts_emb=scraped_posts['emb']

# In[ ]:


tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
trainx_tr = count_vec.fit_transform(Train_X)
testx_tr=count_vec.transform(scraped_posts_text)
train_x = tfidf_transformer.fit_transform(trainx_tr)
test_x= tfidf_transformer.transform(testx_tr)
train_x=train_x.toarray()
test_x=test_x.toarray()

res_train_x=[Train_X_emb[i]+list(train_x[i]) for i in range(len(Train_X))]
res_test_x=[scraped_posts_emb[i]+list(test_x[i]) for i in range(len(scraped_posts_emb))]

RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced') 
RF.fit(res_train_x,Train_Y)
dec_function=RF.predict(res_test_x)
positive_emb=[scraped_posts_emb[i] for i in range(len(res_test_x)) if dec_function[i]==1]
negative_emb=[scraped_posts_emb[i] for i in range(len(res_test_x)) if dec_function[i]==0]
positive_text=[scraped_posts_text[i] for i in range(len(res_test_x)) if dec_function[i]==1]
negative_text=[scraped_posts_text[i] for i in range(len(res_test_x)) if dec_function[i]==0]


new_dataset=[[positive_text[i],positive_emb[i],1] for i in range(len(positive_emb))]+[[negative_text[i],negative_emb[i],0] for i in range(len(negative_emb))]+[[Train_X[i],Train_X_emb[i],Train_Y[i]] for i in range(len(Train_Y))]
random.shuffle(new_dataset)

tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
trainx_tr = count_vec.fit_transform([new_dataset[i][0] for i in range(len(new_dataset))])
testx_tr=count_vec.transform(Test_X)
train_x = tfidf_transformer.fit_transform(trainx_tr)
test_x= tfidf_transformer.transform(testx_tr)
train_x=train_x.toarray()
test_x=test_x.toarray()

res_train_x=[new_dataset[i][1]+list(train_x[i]) for i in range(len(new_dataset))]
res_test_x=[Test_X_emb[i]+list(test_x[i]) for i in range(len(Test_X))]

RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced') 
RF.fit(res_train_x,[new_dataset[i][2] for i in range(len(new_dataset))])
pred=RF.predict(res_test_x)
print("RF Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

