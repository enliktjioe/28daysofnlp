#!/usr/bin/env python
# coding: utf-8

# # References
# 
# - [1] https://realpython.com/python-nltk-sentiment-analysis/
# - [2] https://github.com/hb20007/hands-on-nltk-tutorial

# # Intro to NLTK
# 
# - NLTK (Natural Language ToolKit) is a python library used mainly for processing and analyzing text, from basic functions to sentiment analysis powered by machine learning [1]

# # NLTK Basic

# ## Installation
# - Use this command: `pip install nltk`
# - To get the resources (sample text and data models), need to run this:
# 
# ```
# import nltk
# nltk.download
# ```

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


# import nltk
# nltk.download()


# download specific resources from nltk
import nltk

nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])


# ## Error example
# 
# - caused by missing NLTK resource

w = nltk.corpus.shakespeare.words()


# ## Compiling Data

words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]


stopwords = nltk.corpus.stopwords.words("english")


# ### Removing stop words from original word list

words = [w for w in words if w.lower() not in stopwords]


# ### Tokenize process

from pprint import pprint

text = """
For some quick analysis, creating a corpus could be overkill.
If all you need is a word list,
there are simpler ways to achieve that goal."""
pprint(nltk.word_tokenize(text), width=79, compact=True)


# ## Downloading Libs and Testing That They Are Working [2]

import nltk # https://www.nltk.org/install.html
import numpy # https://www.scipy.org/install.html
import matplotlib.pyplot # https://matplotlib.org/downloads.html
import tweepy # https://github.com/tweepy/tweepy
import TwitterSearch # https://github.com/ckoepp/TwitterSearch
import unidecode # https://pypi.python.org/pypi/Unidecode
import langdetect # https://pypi.python.org/pypi/langdetect
import langid # https://github.com/saffsd/langid.py
import gensim # https://radimrehurek.com/gensim/install.html


# import nltk

# nltk.download()


# # Text Analysis Using nltk.text [2]

from nltk.tokenize import word_tokenize
from nltk.text import Text


my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. Smoke trees. See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."
tokens = word_tokenize(my_string)
tokens = [word.lower() for word in tokens]
tokens[:5]


t = Text(tokens)
t


t.concordance('uckers') # concordance() is a method of the Text class of NLTK. It finds words and displays a context window. Word matching is not case-sensitive.
# concordance() is defined as follows: concordance(self, word, width=79, lines=25). Note default values for optional params.


t.collocations() # def collocations(self, num=20, window_size=2). num is the max no. of collocations to print.


t.count('quack')


t.index('two')


t.similar('brother') # similar(self, word, num=20). Distributional similarity: find other words which appear in the same contexts as the specified word; list most similar words first.


t.dispersion_plot(['man', 'thing', 'quack']) # Reveals patterns in word positions. Each stripe represents an instance of a word, and each row represents the entire text.


t.plot(20) # plots 20 most common tokens


t.vocab()


from nltk.corpus import reuters
text = Text(reuters.words()) # .words() is one method corpus readers provide for reading data from a corpus. We will learn more about these methods in Chapter 2.
text.common_contexts(['August', 'June']) # It seems that .common_contexts() takes 2 words which are used similarly and displays where they are used similarly. It also seems that '_' indicates where the words would be in the text.


# # Deriving N-Grams from Text

# ## Tokenization

s = "Le temps est un grand maître, dit-on, le malheur est qu'il tue ses élèves."
s = s.lower()


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[a-zA-Z'`éèî]+")
s_tokenized = tokenizer.tokenize(s)
s_tokenized


from nltk.util import ngrams
generated_4grams = []

for word in s_tokenized:
    generated_4grams.append(list(ngrams(word, 4, pad_left=True, pad_right=True, left_pad_symbol='_', right_pad_symbol='_'))) # n = 4.
generated_4grams


generated_4grams = [word for sublist in generated_4grams for word in sublist]
generated_4grams[:10]


# ## Obtaining n-grams (n = 4)

ng_list_4grams = generated_4grams
for idx, val in enumerate(generated_4grams):
    ng_list_4grams[idx] = ''.join(val)
ng_list_4grams


# ## Sorting n-grams by frequency (n=4)

freq_4grams = {}

for ngram in ng_list_4grams:
    if ngram not in freq_4grams:
        freq_4grams.update({ngram: 1})
    else:
        ngram_occurrences = freq_4grams[ngram]
        freq_4grams.update({ngram: ngram_occurrences + 1})
        
from operator import itemgetter # The operator module exports a set of efficient functions corresponding to the intrinsic operators of Python. For example, operator.add(x, y) is equivalent to the expression x + y.

freq_4grams_sorted = sorted(freq_4grams.items(), key=itemgetter(1), reverse=True)[0:300] # We only keep the 300 most popular n-grams. This was suggested in the original paper written about n-grams.
freq_4grams_sorted


# ## Obtaining n-grams for multiple values of n

from nltk import everygrams

s_clean = ' '.join(s_tokenized) # For the code below we need the raw sentence as opposed to the tokens.
s_clean


def ngram_extractor(sent):
    return [''.join(ng) for ng in everygrams(sent.replace(' ', '_ _'), 1, 4) 
            if ' ' not in ng and '\n' not in ng and ng != ('_',)]

ngram_extractor(s_clean)




