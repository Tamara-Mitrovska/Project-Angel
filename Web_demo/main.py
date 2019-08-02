from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn import metrics
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn

app = Flask(__name__)
@app.route("/",methods=['GET','POST'])
def home():
    if request.method=='POST':
        text=request.form['text']
        words=highlight(text)
        preds=get_predictions(text)
        if preds[1]==1:
            return render_template('home.html',text=text,label='body shaming',score=str(preds[0][0]),words=words)
        else:
            return render_template('home.html',text=text,label='not body shaming',score=str(preds[0][0]),words=words)
    else:
        return render_template('home.html')



def get_predictions(test_post):
    
    def elmo_emb(elmo, max_words_sent,sent_str):
    	sent = sent_str.split(' ')
    	a = np.zeros(shape=(max_words_sent,3072))
    	vectors = elmo.embed_sentence(sent)
    	v = min(len(sent), max_words_sent)
    	for i in range(v):
    		a[i] = np.concatenate((vectors[0][i], vectors[1][i], vectors[2][i]), axis=None)
    	avg_words = np.mean(a[:v], axis=0) 	
    	return avg_words
    
    with open('training_80%_labels.json','r') as file:
        Train_Y=json.load(file)
    with open('trainx_emb_80%.json','r') as f:
        Train_X_emb=json.load(f)

    elmo = ElmoEmbedder()                     
    max_words_sent=len(test_post.split(' '))
    post_emb=elmo_emb(elmo, max_words_sent,test_post)

    logreg=LogisticRegression(class_weight='balanced')
    logreg.fit(Train_X_emb,Train_Y)
    predictions_lg=logreg.predict([post_emb])
    prob=logreg.decision_function([post_emb])
    return prob,predictions_lg[0]

def highlight(text):
    #used to find words in post related to training dataset
    with open('most_common_tokens.json','r') as file:
        vocab=json.load(file)
    word_lemmatized = WordNetLemmatizer()
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    text=word_tokenize(text)
    words_to_highlight=[]
    for word,tag in pos_tag(text):
        if word not in stopwords.words('english') and word.isalpha():
            word_final = word_lemmatized.lemmatize(word,tag_map[tag[0]])
            if word_final in vocab:
                words_to_highlight.append(word)
    final_text=''
    for word in words_to_highlight:
        final_text+=word+', '
    if final_text[-2:]==', ':
        final_text=final_text[:-2]
    if len(words_to_highlight)==0:
        final_text='None'
    return final_text
    

    

if __name__ == "__main__":
    app.run(debug=True)
    
