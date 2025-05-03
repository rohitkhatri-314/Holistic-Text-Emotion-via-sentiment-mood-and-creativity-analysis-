from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1  
analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english",device=device,return_all_scores=True)

# def Sentiment(text):
#     device = 0 if torch.cuda.is_available() else -1  
#     sentiment_analyzer = pipeline("sentiment-analysis", 
#                                 model="distilbert-base-uncased-finetuned-sst-2-english",device=device,return_all_scores=True)
#     result = sentiment_analyzer(text)[0]
    
#     pos_score=next(r['score'] for  r in result if r['label']=='POSITIVE')
#     neg_score=next(r['score'] for  r in result if r['label']=='NEGATIVE')
    
#     if(abs(neg_score-pos_score)<=0.3):
#         neutrality=min(pos_score,neg_score)
#         return 5+(neutrality*2 -1)
#     else:
#         return (pos_score-neg_score + 1)*5
    
def Sentiment(text):
    try:
        # Get probability scores for both positive and negative
        results = analyzer(text)[0]
        
        pos_score = next(r['score'] for r in results if r['label'] == 'POSITIVE')
        neg_score = next(r['score'] for r in results if r['label'] == 'NEGATIVE')
        
        # Calculate neutrality (1 - confidence)
        confidence = max(pos_score, neg_score)
        neutrality = 1 - confidence
        
        # If neutral (small difference between pos/neg)
        if abs(pos_score - neg_score) < 0.3:  # Threshold for neutrality
            # Scale neutrality to 4-6 range (middle of 0-10)
            return 5 + (neutrality * 2 - 1)  # Maps to 4-6 range
        else:
            # For clear sentiment, scale to full 0-10 range
            sentiment = pos_score - neg_score  # Range [-1, 1]
            return (sentiment + 1) * 5  # Scale to [0, 10]
            
    except Exception as e:
        print(f"Error analyzing text: {e}")
        return 5.0  # Neutral on error
    
print(Sentiment("I am learning french"))