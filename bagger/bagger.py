from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import solve, norm
from numpy.random import rand
import numpy as np


print("Hello, world!")

vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
corpus = [
    "This is a sentence",
    "This is also a sentence",
    "Not sentence",
    "Sentience arises in humans"
]

analyze = vectorizer.build_analyzer()
print(analyze(corpus[0]))

X = vectorizer.fit_transform(corpus)

print(X.toarray())

print(vectorizer.get_feature_names())

test_sentences = [
    ["Some wonder if monkeys have gained sentience"],
    ["The prisoner was given a sentence"]
]

a = vectorizer.transform(test_sentences[0]).toarray()
b = vectorizer.transform(test_sentences[1]).toarray()

print (test_sentences[0])
print(np.dot(X.toarray(), a[0]))

print (test_sentences[1])
print(np.dot(X.toarray(), b[0]))
