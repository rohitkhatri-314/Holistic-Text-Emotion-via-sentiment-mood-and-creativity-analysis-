from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk

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



# nltk.download('wordnet')

sentences =["I love hurting myself","I got hurt"]

final_sentences=[]

for sentence in sentences:
    final_sentences.append(lemma_me(sentence))

print(final_sentences)
vectorizer=TfidfVectorizer(stop_words='english')
vector=vectorizer.fit_transform(final_sentences)
print(vector)
print("Vocabulary:", vectorizer.get_feature_names_out())
similarity= cosine_similarity(vector[0],vector[1])
print(similarity)