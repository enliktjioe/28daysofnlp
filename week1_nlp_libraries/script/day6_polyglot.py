#!/usr/bin/env python
# coding: utf-8

# # References
# 
# - [1] https://polyglot.readthedocs.io/en/latest/
# - [2] https://www.geeksforgeeks.org/natural-language-processing-using-polyglot-introduction/
# - [3] https://blog.jcharistech.com/2018/12/10/introduction-to-natural-language-processing-with-polyglot/

# # Intro
# 
# - Polyglot is a natural language pipeline that supports massive multilingual applications [1]
# - If we know how to use TextBlob, Polyglot has a similar learning curve [3]
# - Developed by Rami Al-Rfou
# - Core Features:
#  - Tokenization
#  - Language detection
#  - Named Entity Recognition
#  - Part of Speech (POS) Tagging
#  - Sentiment Analysis
#  - Word Embeddings
#  - Morphological Analysis
#  - Transliteration

## installation
# !pip install polyglot

# installing dependency packages [2]
get_ipython().system('pip install pyicu morfessor pycld2')


# # Quick Tutorial [1]

import polyglot
from polyglot.text import Text, Word


# ## Language Detection

text = Text("Bonjour, Mesdames.")
print("Language Detected: Code={}, Name={}\n".format(text.language.code, text.language.name))


zen = Text("Beautiful is better than ugly. "
           "Explicit is better than implicit. "
           "Simple is better than complex.")
print(zen.words)


print(zen.sentences)


# ## POS Tagging

## dependencies
get_ipython().system('polyglot download embeddings2.pt')
get_ipython().system('polyglot download pos2.pt')
get_ipython().system('polyglot download embeddings2.de')
get_ipython().system('polyglot download ner2.de')
get_ipython().system('polyglot download sentiment2.en')
get_ipython().system('polyglot download sgns2.en')
get_ipython().system('polyglot download morph2.en')
get_ipython().system('polyglot download transliteration2.ru')


text = Text(u"O primeiro uso de desobediência civil em massa ocorreu em setembro de 1906.")

print("{:<16}{}".format("Word", "POS Tag")+"\n"+"-"*30)
for word, tag in text.pos_tags:
    print(u"{:<16}{:>2}".format(word, tag))


# ## Named Entity Recognition

text = Text(u"In Großbritannien war Gandhi mit dem westlichen Lebensstil vertraut geworden")
print(text.entities)


# ## Polarity

print("{:<16}{}".format("Word", "Polarity")+"\n"+"-"*30)
for w in zen.words[:6]:
    print("{:<16}{:>2}".format(w, w.polarity))


# ## Embeddings

word = Word("Obama", language="en")
print("Neighbors (Synonms) of {}".format(word)+"\n"+"-"*30)
for w in word.neighbors:
    print("{:<16}".format(w))
print("\n\nThe first 10 dimensions out the {} dimensions\n".format(word.vector.shape[0]))
print(word.vector[:10])


# ## Morphology

word = Text("Preprocessing is an essential step.").words[0]
print(word.morphemes)


# ## Transliteration

from polyglot.transliteration import Transliterator
transliterator = Transliterator(source_lang="en", target_lang="ru")
print(transliterator.transliterate(u"preprocessing"))


# # Introduction to Natural Language Processing with Polyglot [3]

# Dependencies
get_ipython().system('polyglot download embeddings2.en')
get_ipython().system('polyglot download ner2.en')
get_ipython().system('polyglot download sentiment2.en')
get_ipython().system('polyglot download pos2.en')
get_ipython().system('polyglot download morph2.en')
get_ipython().system('polyglot download transliteration2.ar')
get_ipython().system('polyglot download transliteration2.fr')


# ## Tokenization

# Load packages
import polyglot
from polyglot.text import Text,Word

# Word Tokens
docx = Text(u"He likes reading and painting")
docx.words


docx2 = Text(u"He exclaimed, 'what're you doing? Reading?'.")
docx2.words


# Sentence tokens
docx3 = Text(u"He likes reading and painting.He exclaimed, 'what're you doing? Reading?'.")
docx3.sentences


# ## POS Tagging

docx


docx.pos_tags


# ## Language Detection

docx


docx.language.name


docx.language.code


from polyglot.detect  import Detector

en_text = "He is a student "
fr_text = "Il est un étudiant"
ru_text = "Он студент"

detect_en = Detector(en_text)
detect_fr = Detector(fr_text)
detect_ru = Detector(ru_text)


print(detect_en.language)


print(detect_fr.language)


print(detect_ru.language)


docx4 = Text(u"He hates reading and playing")


docx


docx.polarity


docx4.polarity


# ## Named Entities

docx5 = Text(u"John Jones was a FBI detector")
docx5.entities


docx6 = Text(u"preprocessing")
docx6.morphemes


# ## Transliteration

# Load 
from polyglot.transliteration import Transliterator
translit = Transliterator(source_lang='en',target_lang='ar')
translit.transliterate(u"hello")




