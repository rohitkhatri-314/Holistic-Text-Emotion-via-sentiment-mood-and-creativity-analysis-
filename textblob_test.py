from textblob import TextBlob
text="I love hurting myself!"
blob=TextBlob(text)

print(f"Sentiment: {blob.sentiment}")
print(f"Polarity: {blob.sentiment.polarity}")  # Range: -1 to 1
print(f"Subjectivity: {blob.sentiment.subjectivity}")