from transformers import pipeline
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np

ds=pd.read_csv("sentences.csv",header=None)

user_input=[]
model_output=[]

device = 0 if torch.cuda.is_available() else -1
analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    return_all_scores=True,
    device=device
)


def lemma_me(sentence):
    
    tokens=nltk.word_tokenize(sentence.lower())

    post_tags=nltk.pos_tag(tokens)
    lemmatized_sentence=""

    for token, tag in zip(tokens,post_tags):
        lemmatiser=WordNetLemmatizer()
        if tag[1][0].lower() in ['a','v','r','n']:
            lemma=lemmatiser.lemmatize(token,tag[1][0].lower())
            lemmatized_sentence+=lemma+" "
        
    return lemmatized_sentence


def Sentiment(text):
    try:
        results = analyzer(text)[0]
        pos_score = next(r['score'] for r in results if r['label'] == 'LABEL_2')
        neg_score = next(r['score'] for r in results if r['label'] == 'LABEL_0')
        
        score=pos_score-neg_score
        return (score+1)*5
            
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return 5.0

for i in range (0,6):
    rand_sent=ds.iloc[i].values[0]
    print(rand_sent)
    correct_input=False
    while(correct_input==False):
        u_input=int(input("Score this sentence form 0-10: "))
        if(u_input<0 or u_input>10):
            print("Wrong input please score between 1-10, try again!")
        else:
            correct_input=True
            user_input.append(u_input)
            m_output=Sentiment(rand_sent)
            if(m_output>=9.7):
                m_output=(m_output%9)*10
            if(m_output<=0.3):
                m_output=m_output*4
            if(m_output>=0.3 and m_output<1):
                m_output=m_output*2.5
            model_output.append(m_output)

mood_distance=(np.mean(user_input)-np.mean(model_output))/10

      
sentences=[]
lemmatized_sentences=[]
similarities=[]
user_sentiment=[]
for i in range (0,5):
    s=str(input("Write a sentence that comes to your mind: "))
    sentences.append(s)
    lemmatized_sentences.append(lemma_me(s))
    m_output=Sentiment(s)
    if(m_output>=9.7):
        m_output=(m_output%9)*10
    if(m_output<=0.3):
        m_output=m_output*4
    if(m_output>=0.3 and m_output<1):
        m_output=m_output*2.5
    user_sentiment.append(m_output)

sentiment=np.mean(user_sentiment)/10
print(sentiment)

vectorizer=TfidfVectorizer(stop_words='english')
vector=vectorizer.fit_transform(lemmatized_sentences)

for i in range(len(sentences)):
    for j in range(i+1,len(sentences)):
        similarity=cosine_similarity(vector[i],vector[j])
        similarities.append(similarity)
        
creativity=1-2*np.mean(similarities)
print(mood_distance)
print(creativity)

overall_sentiment_score=  0.2*creativity +0.8*sentiment + mood_distance
print(overall_sentiment_score)

