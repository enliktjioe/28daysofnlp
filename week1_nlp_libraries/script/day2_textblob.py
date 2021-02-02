#!/usr/bin/env python
# coding: utf-8

# # References
# 
# - [1] https://textblob.readthedocs.io/en/dev/quickstart.html
# - [2] https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
# - [3] https://textblob.readthedocs.io/en/dev/classifiers.html#tutorial-building-a-text-classification-system

# # What is TextBlob?
# 
# - Python library for NLP which is built on the shoulders of NLTK and Pattern [2]
# - Some advantages:
#  - easy to learn and offers a lot of features like sentiment analysis, pos-tagging, noun phrase extraction, etc. [2]
#  

# # Tutorial

# ## Quick Start from Official Docs [1]

# ### Installation

# !pip install -U textblob


from textblob import TextBlob


# ### create our first TextBlob

wiki = TextBlob("Python is a high-level, general-purpose programming language")


# ### Part of speech tagging

wiki.tags


# ### noun phrase extraction

wiki.noun_phrases


testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
testimonial.sentiment


testimonial.sentiment.polarity


# ### Tokenization

zen = TextBlob("Beautiful is better than ugly. "
                "Explicit is better than implicit. "
                "Simple is better than complex.")
zen.words


zen.sentences


for sentence in zen.sentences:
    print(sentence)
    print(sentence.sentiment)


# ### Wordnet Integration

from textblob import Word
from textblob.wordnet import VERB
word = Word("octopus")
word.synsets


Word("hack").get_synsets(pos=VERB)


Word("octopus").definitions


from textblob.wordnet import Synset
octopus = Synset('octopus.n.02')
shrimp = Synset('shrimp.n.03')
octopus.path_similarity(shrimp)


# ### WordLists and Pluralize the word

animals = TextBlob("cat dog octopus")
animals.words


animals.words.pluralize()


# ### Spelling Correction

b = TextBlob("I havvv gooddd speling")
print(b.correct())


# ## Tutorial: Building a Text Classification System [3]

train = [
     ('I love this sandwich.', 'pos'),
     ('this is an amazing place!', 'pos'),
     ('I feel very good about these beers.', 'pos'),
     ('this is my best work.', 'pos'),
     ("what an awesome view", 'pos'),
     ('I do not like this restaurant', 'neg'),
     ('I am tired of this stuff.', 'neg'),
     ("I can't deal with this", 'neg'),
     ('he is my sworn enemy!', 'neg'),
     ('my boss is horrible.', 'neg')
]

test = [
     ('the beer was good.', 'pos'),
     ('I do not enjoy my job', 'neg'),
     ("I ain't feeling dandy today.", 'neg'),
     ("I feel amazing!", 'pos'),
     ('Gary is a friend of mine.', 'pos'),
     ("I can't believe I'm doing this.", 'neg')
]


from textblob.classifiers import NaiveBayesClassifier
class1 = NaiveBayesClassifier(train)


with open('data/day2_sample_text.csv', 'r') as fp:
    class2 = NaiveBayesClassifier(fp, format="csv")


class2.classify("this is amazing movie")


class2.classify("I like this hotel")


# ### Classifying Text

prob_dist = class1.prob_classify("This one's a doozy.")
prob_dist.max()


round(prob_dist.prob("pos"), 2)


round(prob_dist.prob("neg"), 2)


# ### Classifying TextBlobs

from textblob import TextBlob
blob = TextBlob("The beer is good. But the hangover is horrible.", classifier=class1)
blob.classify()


for s in blob.sentences:
    print(s)
    print(s.classify())


# ### Evaluating Classifiers

class1.accuracy(test)


# ### Diplay a Listing of the Most Informative Features

class1.show_informative_features(5)


# ### Updating Classifiers with New Data¶

new_data = [('She is my best friend.', 'pos'),
             ("I'm happy to have a new friend.", 'pos'),
             ("Stay thirsty, my friend.", 'pos'),
             ("He ain't from around here.", 'neg')]
class1.update(new_data)


class1.accuracy(test)


# ### Feature Extractors

def end_word_extractor(document):
    tokens = document.split()
    first_word, last_word = tokens[0], tokens[-1]
    feats = {}
    feats["first({0})".format(first_word)] = True
    feats["last({0})".format(last_word)] = False
    return feats

features = end_word_extractor("I feel happy")
assert features == {'last(happy)': False, 'first(I)': True}


class3 = NaiveBayesClassifier(test, feature_extractor=end_word_extractor)
blob = TextBlob("I'm excited to try my new classifier.", classifier=class3)
blob.classify()




