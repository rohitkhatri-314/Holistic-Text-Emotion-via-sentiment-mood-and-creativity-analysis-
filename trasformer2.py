from transformers import pipeline
import torch

# Initialize analyzer
device = 0 if torch.cuda.is_available() else -1
analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    return_all_scores=True,
    device=device
)

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


