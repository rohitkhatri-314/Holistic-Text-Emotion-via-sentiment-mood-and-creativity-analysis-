from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentences=['Paris is in France','I hate Paris']
vectorizer= TfidfVectorizer()
vector=vectorizer.fit_transform(sentences)

score=cosine_similarity(vector[0],vector[1])
print(score)


