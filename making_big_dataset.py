from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import json
import random

with open('scraper_posts.json','r') as  f:
    scraped_posts=json.load(f)
scraped_posts_text=scraped_posts['text']
scraped_posts_emb=scraped_posts['emb']
print(len(scraped_posts))
with open('training_80%.json','r') as file:
    posts=json.load(file)
with open('trainx_emb_80%.json','r') as file:
    posts_emb=json.load(file)
with open('45k_2.json','r') as file:
    rf_data=json.load(file) #second iteration from running tfidf(char)+rf
print(len(rf_data))
with open('svm_2.json','r') as file: #second iteration from running tfidf(char)+svm
    svm_data=json.load(file)
print(len(svm_data))
    
#print(len(instagram_posts))
new_training=[i for i in rf_data if i in svm_data]
trainx={scraped_posts_text[i]:scraped_posts_emb[i] for i in range(len(scraped_posts_emb))}
for i in range(len(posts)):
    trainx[posts[i]]=posts_emb[i]

final_dataset={}
final_dataset['text']=[]
final_dataset['emb']=[]
final_dataset['label']=[]
for i in range(len(new_training)):
    if new_training[i][0] in trainx:
        final_dataset['text'].append(new_training[i][0])
        final_dataset['label'].append(new_training[i][1])
        final_dataset['emb'].append(trainx[new_training[i][0]])
print(len(new_training),len(final_dataset['text']))
        
with open('final_15K_dataset.json','w') as file:
    json.dump(final_dataset,file)
    


#instagram_posts=[i for i in scraped_posts if i not in [new_training[j][0] for j in range(len(fourk))]]


#with open('testing_20%.json','r') as file:
#    Test_X=json.load(file)
#with open('testing_20%_labels.json','r') as file:
#    Test_Y=json.load(file)
##with open('training_80%.json','r') as file:
##    Train_X=json.load(file)
##with open('training_80%_labels.json','r') as file:
##    Train_Y=json.load(file)
#
#Train_X=[new_training[i][0] for i in range(len(new_training))]
#Train_Y=[new_training[i][1] for i in range(len(new_training))]
#
##tfidf_transformer = TfidfTransformer(norm = 'l2')
##count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
##trainx_tr = count_vec.fit_transform(Train_X)
##testx_tr=count_vec.transform(instagram_posts)
##train_x = tfidf_transformer.fit_transform(trainx_tr)
##test_x= tfidf_transformer.transform(testx_tr)
###
##interval=0.9
##RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced')  
##RF.fit(train_x,Train_Y)
##predictions_scraped_posts=RF.predict_proba(test_x)
##print(predictions_scraped_posts)
##positive=[instagram_posts[i] for i in range(len(instagram_posts)) if predictions_scraped_posts[i][1]>interval]
##negative=[instagram_posts[i] for i in range(len(instagram_posts)) if predictions_scraped_posts[i][0]>interval]
##data_rf=[[i,1] for i in positive]+[[i,0] for i in negative]+[[Train_X[i],Train_Y[i]] for i in range(len(Train_X))]
##random.shuffle(data_rf)
##new_trainx=[post[0] for post in data_rf]
##new_trainy=[post[1] for post in data_rf]
##print(len(positive))
##print(len(new_trainx),len(new_trainy))
##with open('45k_3.json','w') as file:
##    json.dump(data_rf,file)
#    
#tfidf_transformer = TfidfTransformer(norm = 'l2')
#count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
#trainx_tr = count_vec.fit_transform(Train_X)
#testx_tr=count_vec.transform(Test_X)
#train_x = tfidf_transformer.fit_transform(trainx_tr)
#test_x= tfidf_transformer.transform(testx_tr)
#
#RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced')  
#RF.fit(train_x,Train_Y)
#predictions_rf=RF.predict(test_x)
#print("RF Accuracy Score i-> ",accuracy_score(predictions_rf, Test_Y)*100)
#
#SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
#SVM.fit(train_x,Train_Y)
#predictions_SVM = SVM.predict(test_x)
#print("SVM Accuracy Score i-> ",accuracy_score(predictions_SVM, Test_Y)*100)
