import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk


ds=pd.read_csv("sentences.csv",header=None)

user_input=[]
model_output=[]

def Sentiment(text):
    analyser=SentimentIntensityAnalyzer()
    res=analyser.polarity_scores(text)
    return (res['compound'] +1)/2

for i in range (1,6):
    rand_sent=ds.sample(n=1).values[0][0]
    print(rand_sent)
    correct_input=False
    while(correct_input==False):
        u_input=int(input("Score this sentence form 0-10: "))
        if(u_input<0 or u_input>10):
            print("Wrong input please score between 1-10, try again!")
        else:
            correct_input=True
            user_input.append(u_input)
            model_output.append(Sentiment(rand_sent))
            
print(user_input)
print(model_output)

user_in=[]
score=0

weight_sum=0
weight_score_sum=0

# def sentiment(text):
#     analyser=SentimentIntensityAnalyzer()
#     res=analyser.polarity_scores(text)
#     return (res['compound']+1)/2
    

# for i in range (1,11):
#     sentence=input(str("Write a sentence that comes to your mind: "))
#     weight=len(sentence.split())
#     weight_sum+=weight
#     score_i=sentiment(sentence)
#     weight_score_sum+=score_i*weight
#     score=weight_score_sum/weight_sum
#     print(score)

