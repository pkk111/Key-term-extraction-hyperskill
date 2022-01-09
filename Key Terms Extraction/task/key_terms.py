# Write your code here
import string

import nltk
from lxml import etree
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

nltk.download('averaged_perceptron_tagger')


def read():
    newses = etree.parse('news.xml').getroot()[0]
    return newses


def tokenize(newses):
    heads = list()
    texts = list()
    for news in newses:
        heads.append(news[0].text)
        texts.append(Counter(sorted(word_tokenize(news[1].text.lower()), reverse=True)))
    for head, text in zip(heads, texts):
        common = " ".join([word[0] for word in text.most_common(5)])
        print(f"{head}:\n{common}")


def normalisation(newses):
    heads = list()
    texts = list()

    punctuation = list(string.punctuation)
    stopword = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for news in newses:
        heads.append(news[0].text)
        tokens = word_tokenize(news[1].text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in stopword and word not in punctuation]
        texts.append(Counter(sorted(tokens, reverse=True)))
    for head, text in zip(heads, texts):
        common = " ".join([word[0] for word in text.most_common(5)])
        print(f"{head}:\n{common}")


def only_noun_normalisation(newses):
    heads = list()
    texts = list()

    punctuation = list(string.punctuation)
    stopword = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for news in newses:
        heads.append(news[0].text)
        tokens = word_tokenize(news[1].text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if nltk.pos_tag([word])[0][1] == 'NN']
        tokens = [word for word in tokens if word not in stopword and word not in punctuation]
        texts.append(Counter(sorted(tokens, reverse=True)))
    for head, text in zip(heads, texts):
        common = " ".join([word[0] for word in text.most_common(5)])
        print(f"{head}:\n{common}")


def vectorize(newses):
    heads = list()
    texts = list()

    punctuation = list(string.punctuation)
    stopword = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    for news in newses:
        heads.append(news[0].text)
        tokens = word_tokenize(news[1].text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if nltk.pos_tag([word])[0][1] == 'NN']
        tokens = [word for word in tokens if word not in stopword and word not in punctuation]
        texts.append(" ".join(sorted(tokens, reverse=True)))

    vectorizer = TfidfVectorizer(use_idf=True)
    tfmatrix = vectorizer.fit_transform(texts).toarray()
    vocab = vectorizer.get_feature_names_out()
    for i in range(len(heads)):
        matrix = [[tfmatrix[i, j], vocab[j]] for j in range(len(tfmatrix[i]))]
        word = pd.DataFrame(matrix, columns=['weights', 'word'])
        common = " ".join(word.sort_values(['weights', 'word'], ascending=False).head(5)['word'])
        print(f"{heads[i]}:\n{common}")


if __name__ == '__main__':
    vectorize(read())
