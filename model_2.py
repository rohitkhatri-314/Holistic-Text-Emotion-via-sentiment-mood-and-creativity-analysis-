from nltk.sentiment import SentimentIntensityAnalyzer
user_in=[]
score=0

weight_sum=0
weight_score_sum=0

def sentiment(text):
    analyser=SentimentIntensityAnalyzer()
    res=analyser.polarity_scores(text)
    return (res['compound']+1)/2
    

for i in range (1,11):
    sentence=input(str("Write a sentence that comes to your mind: "))
    weight=len(sentence.split())
    weight_sum+=weight
    score_i=sentiment(sentence)
    weight_score_sum+=score_i*weight
    score=weight_score_sum/weight_sum
    print(score)
    
    
    