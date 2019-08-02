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
import json
import random
from sklearn.linear_model import LogisticRegression

with open('scraper_posts.json','r') as  f:
    scraped_posts=json.load(f)
scraped_posts=scraped_posts['text']
print(len(scraped_posts))
with open('svm_2.json','r') as file:
    fourk=json.load(file)
scraped_posts=[i for i in scraped_posts if i not in [fourk[j][0] for j in range(len(fourk))]]
print(len(scraped_posts))

new_training=fourk
    
with open('testing_20%.json','r') as file:
    Test_X=json.load(file)
with open('testing_20%_labels.json','r') as file:
    Test_Y=json.load(file)
#with open('training_80%.json','r') as file:
#    Train_X=json.load(file)
#with open('training_80%_labels.json','r') as file:
#    Train_Y=json.load(file)
#    
Train_X=[new_training[i][0] for i in range(len(new_training))]
Train_Y=[new_training[i][1] for i in range(len(new_training))]
print(len(new_training))   
    
tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
trainx_tr = count_vec.fit_transform(Train_X)
testx_tr=count_vec.transform(scraped_posts)
train_x = tfidf_transformer.fit_transform(trainx_tr)
test_x= tfidf_transformer.transform(testx_tr)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced',probability=True)
SVM.fit(train_x,Train_Y)
interval=0.9
predictions_scraped_posts=SVM.decision_function(test_x)
positive=[scraped_posts[i] for i in range(len(scraped_posts)) if predictions_scraped_posts[i]>interval]
negative=[scraped_posts[i] for i in range(len(scraped_posts)) if predictions_scraped_posts[i]<-interval]
data_svm=[[i,1] for i in positive]+[[i,0] for i in negative]+[[Train_X[i],Train_Y[i]] for i in range(len(Train_X))]
random.shuffle(data_svm)
new_trainx=[post[0] for post in data_svm]
new_trainy=[post[1] for post in data_svm]
print(len(positive),len(negative))
print(len(new_trainx),len(new_trainy))

with open('svm_3.json','w') as file:
    json.dump(data_svm,file)

tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
trainx_tr = count_vec.fit_transform(new_trainx)
testx_tr=count_vec.transform(Test_X)
train_x = tfidf_transformer.fit_transform(trainx_tr)
test_x= tfidf_transformer.transform(testx_tr)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
SVM.fit(train_x,new_trainy)
predictions_SVM = SVM.predict(test_x)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

#logreg=LogisticRegression(class_weight='balanced')
#logreg.fit(train_x,)
#pred_logreg=logreg.predict(test_x)
#print('LogReg Acc: ', accuracy_score(pred_logreg,Test_Y)*100)