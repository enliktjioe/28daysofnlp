#!/usr/bin/env python
# coding: utf-8

# # References
# 
# - [1] https://www.geeksforgeeks.org/nlp-gensim-tutorial-complete-guide-for-beginners/
# - [2] https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-download-auto-examples-core-run-core-concepts-py
# - [3] https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92#e71b

# # Intro to Gensim
# 
# - Gensim with its tagline "Topic Modelling for Humans"
# - Open-source python library written by Radim Rehurek, used in unsupervised topic modelling and NLP.
# - It can handle large text collection
# - Provides multi-processing implementations for various algorithm to increase processing speed

## installation
# !pip install gensim


# # Core Concept [2]

# ## Document
# 
# - In Gensim, a document is an object of the text sequence type (commonly known as `str` in Python 3)
# - It could be anything from a short 140 character teweet, a single paragraph, a news article, app description, app review, or a book 

document = "Human machine interface for lab abc computer applications"


# ## Corpus
# 
# - Corpus is a collection of *Documents* objects.
# - Serve 2 roles:
#  - Input for training a model
#  - Documents to organize. After training, a topic model can be used to extract topics from new documents

import pprint

text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
pprint.pprint(processed_corpus)


# associate each word in the corpus with a unique integer ID

from gensim import corpora

dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)


# - our corpus is small, there are only 12 unique tokens
# - for larger purposes, dictionaries that contains hundreds of thousands of tokens are quite common 

# ## Vector
# 
# - This is the way to represent documents that we can manipulate mathematically
# - Represent each document as a vector of features
# - For example, single feature can be thought of as a question-answer pair

pprint.pprint(dictionary.token2id)


new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)


bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)


# ## Model
# 
# - Transform the vectorized corpus using models
# - Model is abstract term referring to a transformation from one document representation to another 

from gensim import models

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform the "system minors" string
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])


from gensim import similarities

index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)


query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))


# How to read this output?
# 
# - Document 1 has 32% similarity, document 2 has 41% similarity

for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)


# ## Summary
# 
# - There are 4 core concepts of Gensim as mentioned above
# - The actions:
#  - Start with a corpus of documents
#  - Transform these documents to a vector space representation
#  - Create the model that transformed the original vector representation to *Tfldf*
#  - Finally, we used our model to calculate the similarity between some query document and all documents in the corpus

# # A Beginner’s Guide to Word Embedding with Gensim Word2Vec Model [3]
# 
# ## What is Word Embedding?
# 
# - One of the most important techniques in NLP, where wrods are mapped to vectors of real numbers.
# - Enable machine to capture the meaning of a word in a document, semantic, and syntactic similarity, relation with other words.
# - Also be used for recommender systems and text classification
# - This tutorial showing an example of generating word embedding for vehicle make model

# ## Pre-processing

import pandas as pd
df = pd.read_csv('data/data.csv')
df.head()


df['Maker_Model']= df['Make']+ " " + df['Model']

# Select features from original dataset to form a new dataframe 
df1 = df[['Engine Fuel Type','Transmission Type','Driven_Wheels','Market Category','Vehicle Size', 'Vehicle Style', 'Maker_Model']]
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)
df_clean = pd.DataFrame({'clean': df2})


# Create the list of list format of the custom corpus for gensim modeling 
sent = [row.split(',') for row in df_clean['clean']]

# show the example of list of list format of the custom corpus for gensim modeling 
sent[:2]


# ## Gensim word2vec Model Training

from gensim.models import Word2Vec
model = Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1)


model['Toyota Camry']


# ## Compute Similarities

model.similarity('Porsche 718 Cayman', 'Nissan Van')


model.similarity('Porsche 718 Cayman', 'Mercedes-Benz SLK-Class')


model.most_similar('Mercedes-Benz SLK-Class')[:5]


# ## T-SNE Visualizations
# 
# T-SNE is a visualization tool for high-dimensional data by dimension reduction while keeping relative pairwise distance between points

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def display_closestwords_tsnescatterplot(model, word, size):
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    
    close_words = model.similar_by_word(word)

    arr = np.append(arr, np.array([model[word]]), axis=0)

    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
        plt.show()


display_closestwords_tsnescatterplot(model, 'Porsche 718 Cayman', 50)
# display_closestwords_tsnescatterplot(model, 'Maserati Coupe', 50)




