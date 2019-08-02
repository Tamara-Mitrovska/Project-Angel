from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.linear_model import LogisticRegression

with open('training_80%.json','r') as file:
    Train_X_text=json.load(file)
with open('training_80%_labels.json','r') as file:
    Train_Y=json.load(file)
with open('trainx_emb_80%.json','r') as file:
    Train_X_emb=json.load(file)

with open('testing_20%.json','r') as file:
    Test_X_text=json.load(file)
with open('testing_20%_labels.json','r') as file:
    Test_Y=json.load(file)
with open('testx_emb_20%.json','r') as file:
    Test_X_emb=json.load(file)
    
    
#tfidf char ngrams
tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="char",max_features = 10000,stop_words='english',ngram_range = (1,5))
trainx_tr = count_vec.fit_transform(Train_X_text)
testx_tr=count_vec.transform(Test_X_text)
train_x = tfidf_transformer.fit_transform(trainx_tr)
test_x= tfidf_transformer.transform(testx_tr)
train_x_char=train_x.toarray()
test_x_char=test_x.toarray()

#tfidf word ngrams
tfidf_transformer = TfidfTransformer(norm = 'l2')
count_vec = CountVectorizer(analyzer="word",max_features = 10000,stop_words='english',ngram_range = (1,2))
trainx_tr = count_vec.fit_transform(Train_X_text)
testx_tr=count_vec.transform(Test_X_text)
train_x_word = tfidf_transformer.fit_transform(trainx_tr)
test_x_word= tfidf_transformer.transform(testx_tr)

#tfidf+elmo emb.
train_x_tfidf_elmo=[Train_X_emb[i]+list(train_x_char[i]) for i in range(len(Train_X_emb))]
test_x_tfidf_elmo=[Test_X_emb[i]+list(test_x_char[i]) for i in range(len(Test_X_emb))]



#tfidf(char)+svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
SVM.fit(train_x_char,Train_Y)
pred=SVM.predict(test_x_char)
print("Tfidf(char)+SVM Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#tfidf(word)+svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
SVM.fit(train_x_word,Train_Y)
pred=SVM.predict(test_x_word)
print("Tfidf(word)+SVM Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#elmo+svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
SVM.fit(Train_X_emb,Train_Y)
pred=SVM.predict(Test_X_emb)
print("Elmo+SVM Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#elmo+tfidf+svm
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight='balanced')
SVM.fit(train_x_tfidf_elmo,Train_Y)
pred=SVM.predict(test_x_tfidf_elmo)
print("Elmo+tfidf+SVM Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#tfidf(char)+logreg
reg=LogisticRegression(class_weight='balanced')
reg.fit(train_x_char,Train_Y)
pred=reg.predict(test_x_char)
print("Tfidf(char)+LogReg Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#tfidf(word)+logreg
reg=LogisticRegression(class_weight='balanced')
reg.fit(train_x_word,Train_Y)
pred=reg.predict(test_x_word)
print("Tfidf(word)+LogReg Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#elmo+logreg
reg=LogisticRegression(class_weight='balanced')
reg.fit(Train_X_emb,Train_Y)
pred=reg.predict(Test_X_emb)
print("Elmo+LogReg Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#elmo+tfidf+logreg
reg=LogisticRegression(class_weight='balanced')
reg.fit(train_x_tfidf_elmo,Train_Y)
pred=reg.predict(test_x_tfidf_elmo)
print("Elmo+tfidf+LogReg Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#tfidf(char)+rf
RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced') 
RF.fit(train_x_char,Train_Y)
pred=RF.predict(test_x_char)
print("Tfidf(char)+RF Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#tfidf(word)+rf
RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced') 
RF.fit(train_x_word,Train_Y)
pred=RF.predict(test_x_word)
print("Tfidf(word)+RF Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#elmo+rf
RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced') 
RF.fit(Train_X_emb,Train_Y)
pred=RF.predict(Test_X_emb)
print("Elmo+RF Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)

#elmo+tfidf+rf
RF = RandomForestClassifier(n_estimators=1000, random_state=0,class_weight='balanced') 
RF.fit(train_x_tfidf_elmo,Train_Y)
pred=RF.predict(test_x_tfidf_elmo)
print("Elmo+tfidf+RF Accuracy Score -> ",accuracy_score(pred,Test_Y)*100)



