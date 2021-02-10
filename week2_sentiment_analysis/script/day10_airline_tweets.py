#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# - Tutorial source: https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/
# - This tutorial will do sentiment analysis using six US airlines tweets data
# - Dataset source: https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv

# # Let's Begin

# ## Importing the Dataset

#resolve macOS 10.12 SSL issue
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)
airline_tweets.head()


# ## Data Analysis

plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 


# ### Percentage of Public Tweets per Each Airline

airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')


# ### Distribution of Sentiments Across All The Tweets

airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])


# ### Distribution of Sentiment for Each Individual Airline

airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')


import seaborn as sns

sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)


# ## Data Cleaning

features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values


processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


# ## Representing Text in Numeric Form

# ### TF-IDF
# 
# - Combination of two terms, `Terms Frequence` and `Inverse Document Frequency`

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


# ## Dividing Data into Training and Test Sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# ## Training the Model

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier.fit(X_train, y_train)


predictions = text_classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


# From the output, we can see that the algorithm used achieved and accuracy of 75.30%

# # Extra: Topic Modeling
# 
# Source: https://stackabuse.com/python-for-nlp-topic-modeling/
# 
# - Topic modeling is an unsupervised technique that intends to analyze large volumes of text data by clustering the documents into groups
# - In case of topic modeling, the text data don't have any labels attached to it.
# - Rather, topic modeling tries to group the documents into clusters based on similar characteristics.

# ## Latend Dirichlet Allocation (LDA)
# 
# - The LDA is based on two general assumptions
#  - Documents that have similar words usually have the same topic
#  - Documents that have groups of words frequently occuring together usually have the same topic

# ### Dataset
# 
# - Source: https://www.kaggle.com/sdxingaijing/topic-model-lda-algorithm/data

# ### LDA for Topic Modeling

import pandas as pd
import numpy as np

reviews_datasets = pd.read_csv(r'data/Reviews.csv')
reviews_datasets = reviews_datasets.head(20000)
reviews_datasets = reviews_datasets.dropna()


reviews_datasets.head()


reviews_datasets['Text'][350]


# #### Count Vectorizer

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))


doc_term_matrix


# #### Use LDA

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)


"""
Randomly fetches 10 words from our vocabulary
"""

import random

for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])


first_topic = LDA.components_[0]
top_topic_words = first_topic.argsort()[-10:]


for i in top_topic_words:
    print(count_vect.get_feature_names()[i])


# #### 10 Words with Highest Probabilities

for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# #### Transform
# 
# - Add a column to the original data frame that will store the topic for the text

topic_values = LDA.transform(doc_term_matrix)
topic_values.shape


reviews_datasets['Topic'] = topic_values.argmax(axis=1)


reviews_datasets.head()


# ## Non-Negative Matrix Factorization (NMF) for Topic Modeling

import pandas as pd
import numpy as np

reviews_datasets = pd.read_csv(r'data/Reviews.csv')
reviews_datasets = reviews_datasets.head(20000)
reviews_datasets = reviews_datasets.dropna()


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = tfidf_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))


from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
nmf.fit(doc_term_matrix )


# import random

for i in range(10):
    random_id = random.randint(0,len(tfidf_vect.get_feature_names()))
    print(tfidf_vect.get_feature_names()[random_id])


first_topic = nmf.components_[0]
top_topic_words = first_topic.argsort()[-10:]

for i in top_topic_words:
    print(tfidf_vect.get_feature_names()[i])


for i,topic in enumerate(nmf.components_):
    print(f'Top 10 words for topic #{i}:')
    print([tfidf_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


topic_values = nmf.transform(doc_term_matrix)
reviews_datasets['Topic'] = topic_values.argmax(axis=1)
reviews_datasets.head()


# ## Conclusion
# 
# - Topic Modeling is one of the most sought after research areas in NLP
# - Used to group large volumes of unlabeled text data



