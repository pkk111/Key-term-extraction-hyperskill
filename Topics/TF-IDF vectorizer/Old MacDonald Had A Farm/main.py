#  write your code here
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(input='file', use_idf=True, lowercase=True,
                             analyzer='word', ngram_range=(1, 1),
                             stop_words=None)
tfidf_matrix = vectorizer.fit_transform([open('data/dataset/input.txt')])
print("0.3525079774951049")
