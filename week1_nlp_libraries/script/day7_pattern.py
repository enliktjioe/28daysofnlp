#!/usr/bin/env python
# coding: utf-8

# # References
# 
# - [1] https://stackabuse.com/python-for-nlp-introduction-to-the-pattern-library/
# - [2] [NLP Tutorial 3 - Extract Text from PDF Files in Python for NLP | PDF Writer and Reader in Python](https://youtu.be/_VSX7yd-zPE)
# - [3] https://analyticsindiamag.com/hands-on-guide-to-pattern-a-python-tool-for-effective-text-processing-and-data-mining/
# - [4] [General Comparison between different Python NLP Libraries](https://medium.com/towards-artificial-intelligence/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0)
# - [5] https://textminingonline.com/getting-started-with-pattern

# # Intro
# 
# - The Pattern library is a multipurpose library capable of handling the following tasks: [1]
#  - NLP: performing tasks such as tokenization, stemming, POS tagging, sentiment analysis, etc
#  - Data Mining: has API to mine data from sites like Twitter, Facebook, Wikipedia, etc
#  - ML: contains ML models such as SVM, KNN, and perceptron, which can be used for classification, regression, and clustering tasks
# - Even it's not as popular as spaCy or NLTK, it has unique functionalities such as finding superlatives and comparatives, get fact and opinion detecetion which other NLP libraries doesn't have [1]

## installation
# !pip install pattern


# # Python for NLP: Introduction to the Pattern Library [1]

# ## Pattern Library Functions for NLP

# ### Tokenizing, POS Tagging, and Chunking

from pattern.en import parse
from pattern.en import pprint


pprint(parse('I drove my car to the hospital yesterday', relations=True, lemmata=True))


print(parse('I drove my car to the hospital yesterday', relations=True, lemmata=True).split())


# ### Pluralizing and Singularizing the Tokens

from pattern.en import pluralize, singularize

print(pluralize('leaf'))
print(singularize('theives'))


# ### Converting Adjective to Comparative and Superlative Degrees

from pattern.en import comparative, superlative

print(comparative('good'))
print(superlative('good'))


# ### Finding N-Grams

from pattern.en import ngrams

print(ngrams("He goes to hospital", n=2))


# ### Finding Sentiments

from pattern.en import sentiment

print(sentiment("This is an excellent movie to watch. I really love it"))


# Explanation:
# 
# - 0.75 show the sentiment score of the sentence that means highly positive
# - 0.8 is the subjectivity score that is a personal of the user  

# ### Checking if a Statement is a Fact

from pattern.en import parse, Sentence
from pattern.en import modality

text = "Paris is the capital of France"
sent = parse(text, lemmata=True)
sent = Sentence(sent)

print(modality(sent))


text = "I think we can complete this task"
sent = parse(text, lemmata=True)
sent = Sentence(sent)

print(modality(sent))


# ### Spelling Corrections

from pattern.en import suggest

print(suggest("Whitle"))


from pattern.en import suggest
print(suggest("Fracture"))


# ### Working with Numbers

from pattern.en import number, numerals

print(number("one hundred and twenty two"))
print(numerals(256.390, round=2))


from pattern.en import quantify

print(quantify(['apple', 'apple', 'apple', 'banana', 'banana', 'banana', 'mango', 'mango']))


from pattern.en import quantify

print(quantify({'strawberry': 200, 'peach': 15}))
print(quantify('orange', amount=1200))


# ## Pattern Library Functions for Data Mining

# For macOS SSL issue when downloading file(s) from external sources
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context


# ### Accessing Web Pages

from pattern.web import download

page_html = download('https://en.wikipedia.org/wiki/Artificial_intelligence', unicode=True)


from pattern.web import URL, extension

page_url = URL('https://upload.wikimedia.org/wikipedia/commons/f/f1/RougeOr_football.jpg')
file = open('football' + extension(page_url.page), 'wb')
file.write(page_url.download())
file.close()


# ### Finding URLs within Text

from pattern.web import find_urls

print(find_urls('To search anything, go to www.google.com', unique=True))


# ### Making Asynchronous Requests for Webpages

from pattern.web import asynchronous, time, Google

asyn_req = asynchronous(Google().search, 'artificial intelligence', timeout=4)
while not asyn_req.done:
    time.sleep(0.1)
    print('searching...')

print(asyn_req.value)

print(find_urls(asyn_req.value, unique=True))


# ### Getting Search Engine Results with APIs

# #### Google

from pattern.web import Google

google = Google(license=None)
for search_result in google.search('artificial intelligence'):
    print(search_result.url)
    print(search_result.text)


# #### Twitter

from pattern.web import Twitter

twitter = Twitter()
index = None
for j in range(3):
    for tweet in twitter.search('artificial intelligence', start=index, count=3):
        print(tweet.text)
        index = tweet.id


# ### Converting HTML Data to Plain Text

from pattern.web import URL, plaintext

html_content = URL('https://stackabuse.com/python-for-nlp-introduction-to-the-textblob-library/').download()
cleaned_page = plaintext(html_content.decode('utf-8'))
print(cleaned_page)


# ### Parsing PDF Documments

# #### Using Pattern PDF module (doesn't work)

# # This doesn't work
# from pattern.web import URL, PDF

# pdf_doc = URL('http://demo.clab.cs.cmu.edu/NLP/syllabus_f18.pdf').download()
# # pdf_doc2 = URL('https://courses.cs.ut.ee/LTAT.01.001/2020_spring/uploads/Main/Lecture1_Introduction.pdf').download()
# print(PDF(pdf_doc2.decode('utf-8')))


# #### Using PyPDF2 library [4]

## dependencies
# !pip install PyPDF2


import PyPDF2 as pdf

file = open('data/syllabus_f18.pdf', 'rb') # source: http://demo.clab.cs.cmu.edu/NLP/syllabus_f18.pdf
file


pdf_reader = pdf.PdfFileReader(file)
pdf_reader


help(pdf_reader)


pdf_reader.getIsEncrypted()


# This PDF is not encrypted

pdf_reader.getNumPages()


page1 = pdf_reader.getPage(0)
page1


page1.extractText()


page2 = pdf_reader.getPage(1)
page2.extractText()


# ### Clearing the Cache

from pattern.web import cache

cache.clear()

