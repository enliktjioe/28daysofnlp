#!/usr/bin/env python
# coding: utf-8

# # References
# 
# - [1] https://www.machinelearningplus.com/spacy-tutorial-nlp/
# - [2] https://realpython.com/natural-language-processing-spacy-python/
# - [3] https://spacy.io/
# - [4] https://towardsdatascience.com/a-short-introduction-to-nlp-in-python-with-spacy-d0aa819af3ad

# # Intro
# 
# - spaCy is an advanced modern library for NLP developed by Matthew Honnibal and Ines Montani [1]
# - Designed to be industry quality, but open source
# - It comes with pre-trained NLP model that can perform most common NLP tasks, such as
#  - tokenization
#  - parts of speech (POS) tagging
#  - named entity recognition (NER)
#  - lemmatization
#  - transforming to word vectors

## installation
# !pip install spacy


# # spaCy Tutorial – Complete Writeup [1]

import spacy

nlp = spacy.load("en_core_web_sm")
nlp


# This returns a Language object that comes ready with multiple built-in capabilities.

# ## The Doc object

# Parse text through the `nlp` model
my_text = """The economic situation of the country is on edge , as the stock 
market crashed causing loss of millions. Citizens who had their main investment 
in the share-market are facing a great loss. Many companies might lay off 
thousands of people to reduce labor cost"""

my_doc = nlp(my_text)
type(my_doc)


# ## Tokenization with spaCy
# 
# - Tokenization is the process of converting a text into smaller sub-texts, based on certain predefined rules
# - For example:
#  - sentences are tokenized to words (and punctuation optionally)
#  - paragraph into sentences, depending on the context

# Printing the tokens of a doc
for token in my_doc:
    print(token.text)


# ## Text-Preprocessing with spaCy
# 
# - Preprocessing is the process of removing noise from the document
# - Some elements that need to be removed:
#  - stop words
#  - punctuation
# 

# Printing tokens and boolean values stored in different attributes
for token in my_doc:
  print(token.text,'--',token.is_stop,'---',token.is_punct)


# Removing StopWords and punctuations
my_doc_cleaned = [token for token in my_doc if not token.is_stop and not token.is_punct]

for token in my_doc_cleaned:
  print(token.text)


# ### Huge Text

# Reading a huge text data on robotics into a spacy doc
robotics_data= """Robotics is an interdisciplinary research area at the interface of computer science and engineering. Robotics involvesdesign, construction, operation, and use of robots. The goal of robotics is to design intelligent machines that can help and assist humans in their day-to-day lives and keep everyone safe. Robotics draws on the achievement of information engineering, computer engineering, mechanical engineering, electronic engineering and others.Robotics develops machines that can substitute for humans and replicate human actions. Robots can be used in many situations and for lots of purposes, but today many are used in dangerous environments(including inspection of radioactive materials, bomb detection and deactivation), manufacturing processes, or where humans cannot survive (e.g. in space, underwater, in high heat, and clean up and containment of hazardousmaterials and radiation). Robots can take on any form but some are made to resemble humans in appearance. This is said to help in the acceptance of a robot in 
certain replicative behaviors usually performed by people. Such robots attempt to replicate walking, lifting, speech, cognition, or any other human activity. Many of todays robots are inspired by nature, contributing to the field of bio-inspired 
robotics.The concept of creating machines that can operate autonomously dates back to classical times, but research into the functionality and potential uses of robots did not grow substantially until the 20th century. Throughout history, it has been frequently assumed by various scholars, inventors, engineers, and technicians that robots will one day be able to mimic human behavior and manage tasks in a human-like fashion. Today, robotics is a rapidly growing field, as technological advances continue; researching, designing, and building new robots serve various practical purposes, whether domestically, commercially, or militarily. Many robots are built to do jobs that are hazardous to people, such as defusing bombs, finding survivors in unstable ruins, and exploring mines and shipwrecks. Robotics is also used in STEM (science, technology, engineering, and mathematics) as a teaching aid. The advent of nanorobots, microscopic robots that can be injected into the human body, could revolutionize medicine and human health.Robotics is a branch of engineering that involves the conception, design, manufacture, and operation of robots. This field overlaps with computer engineering, computer science (especially artificial intelligence), electronics, mechatronics, nanotechnology and bioengineering.The word robotics was derived from the word robot, which was introduced to the public by Czech writer Karel Capek in his play R.U.R. (Rossums Universal Robots), whichwas published in 1920. The word robot comes from the Slavic word robota, which means slave/servant. The play begins in a factory that makes artificial people called robots, creatures who can be mistaken for humans – very similar to the modern ideas of androids. Karel Capek himself did not coin the word. He wrote a short letter in reference to an etymology in the 
Oxford English Dictionary in which he named his brother Josef Capek as its actual 
originator.According to the Oxford English Dictionary, the word robotics was first 
used in print by Isaac Asimov, in his science fiction short story "Liar!", 
published in May 1941 in Astounding Science Fiction. Asimov was unaware that he 
was coining the term  since the science and technology of electrical devices is 
electronics, he assumed robotics already referred to the science and technology 
of robots. In some of Asimovs other works, he states that the first use of the 
word robotics was in his short story Runaround (Astounding Science Fiction, March 
1942) where he introduced his concept of The Three Laws of Robotics. However, 
the original publication of "Liar!" predates that of "Runaround" by ten months, 
so the former is generally cited as the words origin.There are many types of robots; 
they are used in many different environments and for many different uses. Although 
being very diverse in application and form, they all share three basic similarities 
when it comes to their construction:Robots all have some kind of mechanical construction, a frame, form or shape designed to achieve a particular task. For example, a robot designed to travel across heavy dirt or mud, might use caterpillar tracks. The mechanical aspect is mostly the creators solution to completing the assigned task and dealing with the physics of the environment around it. Form follows function.Robots have electrical components which power and control the machinery. For example, the robot with caterpillar tracks would need some kind of power to move the tracker treads. That power comes in the form of electricity, which will have to travel through a wire and originate from a battery, a basic electrical circuit. Even petrol powered machines that get their power mainly from petrol still require an electric current to start the combustion process which is why most petrol powered machines like cars, have batteries. The electrical aspect of robots is used for movement (through motors), sensing (where electrical signals are used to measure things like heat, sound, position, and energy status) and operation (robots need some level of electrical energy supplied to their motors and sensors in order to activate and perform basic operations) All robots contain some level of computer programming code. A program is how a robot decides when or how to do something. In the caterpillar track example, a robot that needs to move across a muddy road may have the correct mechanical construction and receive the correct amount of power from its battery, but would not go anywhere without a program telling it to move. Programs are the core essence of a robot, it could have excellent mechanical and electrical construction, but if its program is poorly constructed its performance will be very poor (or it may not perform at all). There are three different types of robotic programs: remote control, artificial intelligence and hybrid. A robot with remote control programing has a preexisting set of commands that it will only perform if and when it receives a signal from a control source, typically a human being with a remote control. It is perhaps more appropriate to view devices controlled primarily by human commands as falling in the discipline of automation rather than robotics. Robots that use artificial intelligence interact with their environment on their own without a control source, and can determine reactions to objects and problems they encounter using their preexisting programming. Hybrid is a form of programming that incorporates both AI and RC functions.As more and more robots are designed for specific tasks this method of classification becomes more relevant. For example, many robots are designed for assembly work, which may not be readily adaptable for other applications. They are termed as "assembly robots". For seam welding, some suppliers provide complete welding systems with the robot i.e. the welding equipment along with other material handling facilities like turntables, etc. as an integrated unit. Such an integrated robotic system is called a "welding robot" even though its discrete manipulator unit could be adapted to a variety of tasks. Some robots are specifically designed for heavy load manipulation, and are labeled as "heavy-duty robots".one or two wheels. These can have certain advantages such as greater efficiency and reduced parts, as well as allowing a robot to navigate in confined places that a four-wheeled robot would not be able to.Two-wheeled balancing robots Balancing robots generally use a gyroscope to detect how much a robot is falling and then drive the wheels proportionally in the same direction, to counterbalance the fall at hundreds of times per second, based on the dynamics of an inverted pendulum.[71] Many different balancing robots have been designed.[72] While the Segway is not commonly thought of as a robot, it can be thought of as a component of a robot, when used as such Segway refer to them as RMP (Robotic Mobility Platform). An example of this use has been as NASA Robonaut that has been mounted on a Segway.One-wheeled balancing robots Main article: Self-balancing unicycle A one-wheeled balancing robot is an extension of a two-wheeled balancing robot so that it can move in any 2D direction using a round ball as its only wheel. Several one-wheeled balancing robots have been designed recently, such as Carnegie Mellon Universitys "Ballbot" that is the approximate height and width of a person, and Tohoku Gakuin University BallIP Because of the long, thin shape and ability to maneuver in tight spaces, they have the potential to function better than other robots in environments with people
"""

# Pass the Text to Model
robotics_doc = nlp(robotics_data)

print('Before PreProcessing n_Tokens: ', len(robotics_doc))

# Removing stopwords and punctuation from the doc.
robotics_doc=[token for token in robotics_doc if not token.is_stop and not token.is_punct]

print('After PreProcessing n_Tokens: ', len(robotics_doc))


# ## Lemmatization
# 
# - These words: played, playing, plays, play - are not entirely unique as they have same root word "play".
# - Lemmatization = method of converting a token to it's root/base form

# Lemmatizing the tokens of a doc
text='she played chess against rita she likes playing chess.'
doc=nlp(text)
for token in doc:
  print(token.lemma_)


# ## Strings to Hashes
# 
# - spaCy hashes or converts each string to a unique ID that is stored in the `StringStore`.
# - `StringStore` is a dictionary mapping of hash values to strings, for example `10543432924755684266` -> box

# Strings to Hashes and Back
doc = nlp("I love traveling")

# Look up the hash for the word "traveling"
word_hash = nlp.vocab.strings["traveling"]
print(word_hash)

# Look up the word_hash to get the string
word_string = nlp.vocab.strings[word_hash]
print(word_string)


# Create two different doc with a common word
doc1 = nlp('Raymond shirts are famous')
doc2 = nlp('I washed my shirts ')

# Printing the hash value for each token in the doc

print('-------DOC 1-------')
for token in doc1:
    hash_value=nlp.vocab.strings[token.text]
    print(token.text ,' ',hash_value)

print('-------DOC 2-------')
for token in doc2:
    hash_value=nlp.vocab.strings[token.text]
    print(token.text ,' ',hash_value)


# ## Lexical attributes of spaCy
# 
# - Lexical attributes of Token object

# Printing the tokens which are like numbers
text=' 2020 is far worse than 2009'
doc=nlp(text)
for token in doc:
    if token.like_num:
        print(token)


production_text=' Production in chennai is 87 %. In Kolkata, produce it as low as 43 %. In Bangalore, production ia as good as 98 %.In mysore, production is average around 78 %'


# Finding the tokens which are numbers followed by % 

production_doc=nlp(production_text)

for token in production_doc:
    if token.like_num:
        index_of_next_token=token.i+ 1
        next_token=production_doc[index_of_next_token]
        if next_token.text == '%':
            print(token.text)


# ## Detecting Email Addresses
# 
# 

# text containing employee details
employee_text=""" name : Koushiki age: 45 email : koushiki@gmail.com
                 name : Gayathri age: 34 email: gayathri1999@gmail.com
                 name : Ardra age: 60 email : ardra@gmail.com
                 name : pratham parmar age: 15 email : parmar15@yahoo.com
                 name : Shashank age: 54 email: shank@rediffmail.com
                 name : Utkarsh age: 46 email :utkarsh@gmail.com"""

# creating a spacy doc          
employee_doc=nlp(employee_text)

# Printing the tokens which are email through `like_email` attribute
for token in employee_doc:
    if token.like_email:
        print(token.text)


# ## Part of Speech analysis with spaCy

# POS tagging using spaCy
my_text='John plays basketball,if time permits. He played in high school too.'
my_doc=nlp(my_text)
for token in my_doc:
  print(token.text,'---- ',token.pos_)


spacy.explain('SCONJ')


# ## How POS tagging helps you in dealing with text based problems.

# Raw text document
raw_text="""I liked the movies etc The movie had good direction  The movie was amazing i.e.
            The movie was average direction was not bad The cinematography was nice. i.e.
            The movie was a bit lengthy  otherwise fantastic  etc etc"""

# Creating a spacy object
raw_doc=nlp(raw_text)

# Checking if POS tag is X and printing them
print('The junk values are..')
for token in raw_doc:
    if token.pos_=='X':
        print(token.text)

print('After removing junk')
# Removing the tokens whose POS tag is junk.
clean_doc=[token for token in raw_doc if not token.pos_=='X']
print(clean_doc)


# creating a dictionary with parts of speeach &amp; corresponding token numbers.

all_tags = {token.pos: token.pos_ for token in raw_doc}
print(all_tags)


# Importing displacy
from spacy import displacy
my_text='She never like playing , reading was her hobby'
my_doc=nlp(my_text)

# displaying tokens with their POS tags
displacy.render(my_doc,style='dep',jupyter=True)


# ## Named Entity Recognition

# Preparing the spaCy document
text='Tony Stark owns the company StarkEnterprises . Emily Clark works at Microsoft and lives in Manchester. She loves to read the Bible and learn French'
doc=nlp(text)

# Printing the named entities
print(doc.ents)


# Printing labels of entities.
for entity in doc.ents:
    print(entity.text,'--- ',entity.label_)


# Using displacy for visualizing NER
from spacy import displacy
displacy.render(doc,style='ent',jupyter=True)


# ## NER Application 1: Extracting brand names with Named Entity Recognition

mobile_industry_article=""" 30 Major mobile phone brands Compete in India – A Case Study of Success and Failures
Is the Indian mobile market a terrible War Zone? We have more than 30 brands competing with each other. Let’s find out some insights about the world second-largest mobile bazaar.There is a massive invasion by Chinese mobile brands in India in the last four years. Some of the brands have been able to make a mark while others like Meizu, Coolpad, ZTE, and LeEco are a failure.On one side, there are brands like Sony or HTC that have quit from the Indian market on the other side we have new brands like Realme or iQOO entering the marketing in recent months.The mobile market is so competitive that some of the brands like Micromax, which had over 18% share back in 2014, now have less than 5%. Even the market leader Samsung with a 34% market share in 2014, now has a 21% share whereas Xiaomi has become a market leader. The battle is fierce and to sustain and scale-up is going to be very difficult for any new entrant.new comers in Indian Mobile MarketiQOO –They have recently (March 2020) launched the iQOO 3 in India with its first 5G phone – iQOO 3. The new brand is part of the Vivo or the BBK electronics group that also owns several other brands like Oppo, Oneplus and Realme.Realme – Realme launched the first-ever phone – Realme 1 in November 2018 and has quickly became a popular brand in India. The brand is one of the highest sellers in online space and even reached a 16% market share threatening Xiaomi’s dominance.iVoomi – In 2017, we have seen the entry of some new Chinese mobile brands likeiVoomi which focuses on the sub 10k price range, and is a popular online player. They have an association with Flipkart.Techno &amp; Infinix – Transsion Group’s Tecno and Infinix brands debuted in India in mid-2017 and are focusing on the low end and mid-range phones in the price range of Rs. 5000 to Rs. 12000.10.OR &amp; Lephone – 10.OR has a partnership with Amazon India and is an exclusive online brand with phones like 10.OR D, G and E. However, the brand is not very aggressive currently.Kult – Kult is another player who launched a very aggressively priced Kult Beyond mobile in 2017 and followed up by launching 2-3 more models.However, most of these new brands are finding it difficult to strengthen their footing in India. As big brands like Xiaomi leave no stone unturned to make things difficult.Also, it is worth noting that there is less Chinese players coming to India now. As either all the big brands have already set shop or burnt their hands and retreated to the homeland China.Chinese/ Global  Brands Which failed or are at the Verge of Failing in India?
There are a lot more failures in the market than the success stories. Let’s first look at the failures and then we will also discuss why some brands were able to succeed in India.HTC – The biggest surprise this year for me was the failure of HTC in India. The brand has been in the country for many years, in fact, they were the first brand to launch Android mobiles. Finally HTC decided to call it a day in July 2018.LeEco – LeEco looked promising and even threatening to Xiaomi when it came to India. The company launched a series of new phones and smart TVs at affordable rates. Unfortunately, poor financial planning back home caused the brand to fail in India too.LG – The company seems to have lost focus and are doing poorly in all segments. While the budget and mid-range offering are uncompetitive, the high-end models are not preferred by buyers.Sony – Absurd pricing and lack of ability to understand the Indian buyers have caused Sony to shrink mobile operations in India. In the last 2 years, there are far fewer launches and hardly any promotions or hype around the new products.Meizu – Meizu is also a struggling brand in India and is going nowhere with the current strategy. There are hardly any popular mobiles nor a retail presence.ZTE – The company was aggressive till last year with several new phones launching under the Nubia banner, but with recent issues in the US, they have even lost the plot in India.Coolpad – I still remember the first meeting with Coolpad CEO in Mumbai when the brand started operations. There were big dreams and ambitions, but the company has not been able to deliver and keep up with the rivals in the last 1 year.Gionee – Gionee was doing well in the retail, but the infighting in the company and loss of focus from the Chinese parent company has made it a failure. The company is planning a comeback. However, we will have to wait and see when that happens."""


# creating spacy doc
mobile_doc=nlp(mobile_industry_article)

# List to store name of mobile companies
list_of_org=[]

# Appending entities which havel the label 'ORG' to the list
for entity in mobile_doc.ents:
    if entity.label_=='ORG':
        list_of_org.append(entity.text)

print(list_of_org)


# ## NER Application 2: Automatically Masking Entities

# Creating a doc on news articles
news_text="""Indian man has allegedly duped nearly 50 businessmen in the UAE of USD 1.6 million and fled the country in the most unlikely way -- on a repatriation flight to Hyderabad, according to a media report on Saturday.Yogesh Ashok Yariava, the prime accused in the fraud, flew from Abu Dhabi to Hyderabad on a Vande Bharat repatriation flight on May 11 with around 170 evacuees, the Gulf News reported.Yariava, the 36-year-old owner of the fraudulent Royal Luck Foodstuff Trading, made bulk purchases worth 6 million dirhams (USD 1.6 million) against post-dated cheques from unsuspecting traders before fleeing to India, the daily said.
The bought goods included facemasks, hand sanitisers, medical gloves (worth nearly 5,00,000 dirhams), rice and nuts (3,93,000 dirhams), tuna, pistachios and saffron (3,00,725 dirhams), French fries and mozzarella cheese (2,29,000 dirhams), frozen Indian beef (2,07,000 dirhams) and halwa and tahina (52,812 dirhams).
The list of items and defrauded persons keeps getting longer as more and more victims come forward, the report said.
The aggrieved traders have filed a case with the Bur Dubai police station.
The traders said when the dud cheques started bouncing they rushed to the Royal Luck's office in Dubai but the shutters were down, even the fraudulent company's warehouses were empty."""

news_doc=nlp(news_text)


# Function to identify  if tokens are named entities and replace them with UNKNOWN
def remove_details(word):
    if word.ent_type_ =='PERSON' or word.ent_type_=='ORG' or word.ent_type_=='GPE':
        return ' UNKNOWN '
    return word.string


# Function where each token of spacy doc is passed through remove_deatils()
def update_article(doc):
    # iterrating through all entities
    for ent in doc.ents:
        ent.merge()
    # Passing each token through remove_details() function.
    tokens = map(remove_details,doc)
    return ''.join(tokens)

# Passing our news_doc to the function update_article()
update_article(news_doc)


# ## Rule based Matching

from spacy.matcher import Matcher

# Initializing the matcher with vocab
matcher = Matcher(nlp.vocab)
matcher


# ### Token Matcher Example 1

# Define the matching pattern
my_pattern=[{"LOWER": "version"}, {"IS_PUNCT": True}, {"LIKE_NUM": True}]

# Define the token matcher
matcher.add('VersionFinder', None, my_pattern)

# Run the Token Matcher
my_text = 'The version : 6 of the app was released about a year back and was not very sucessful. As a comeback, six months ago, version : 7 was released and it took the stage. After that , the app has has the limelight till now. On interviewing some sources, we get to know that they have outlined visiond till version : 12 ,the Ultimate.'
my_doc = nlp(my_text)

desired_matches = matcher(my_doc)
desired_matches


# Extract the matches
for match_id, start, end in desired_matches :
    string_id = nlp.vocab.strings[match_id] 
    span = my_doc[start:end] 
    print(span.text)


# ### Token Matcher Example 2

# Parse text
text = """I visited Manali last time. Around same budget trips ? "
    I was visiting Ladakh this summer "
    I have planned visiting NewYork and other abroad places for next year"
    Have you ever visited Kodaikanal? """

doc = nlp(text)


# Initialize the matcher
matcher = Matcher(nlp.vocab)

# Write a pattern that matches a form of "visit" + place
my_pattern = [{"LEMMA": "visit"}, {"POS": "PROPN"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("Visting_places", None,my_pattern)
matches = matcher(doc)

# Counting the no of matches
print(" matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)


# ### Token Matcher Example 3

# Parse text
engineering_text = """If you study aeronautical engineering, you could specialize in aerodynamics, aeroelasticity, 
composites analysis, avionics, propulsion and structures and materials. If you choose to study chemical engineering, you may like to
specialize in chemical reaction engineering, plant design, process engineering, process design or transport phenomena. Civil engineering is the professional practice of designing and developing infrastructure projects. This can be on a huge scale, such as the development of
nationwide transport systems or water supply networks, or on a smaller scale, such as the development of single roads or buildings.
specializations of civil engineering include structural engineering, architectural engineering, transportation engineering, geotechnical engineering,
environmental engineering and hydraulic engineering. Computer engineering concerns the design and prototyping of computing hardware and software. 
This subject merges electrical engineering with computer science, oldest and broadest types of engineering, mechanical engineering is concerned with the design,
manufacturing and maintenance of mechanical systems. You’ll study statics and dynamics, thermodynamics, fluid dynamics, stress analysis, mechanical design and
technical drawing"""

doc = nlp(engineering_text)

# Initializing the matcher
matcher = Matcher(nlp.vocab)

# Write a pattern that matches a form of "noun/adjective"+"engineering"
my_pattern = [{"POS": {"IN": ["NOUN", "ADJ"]}}, {"LOWER": "engineering"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("identify_courses", None,my_pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the matching text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)


# #### Phrase Matcher

from spacy.matcher import PhraseMatcher

# PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# Terms to match
terms_list = ['Bruce Wayne', 'Tony Stark', 'Batman', 'Harry Potter', 'Severus Snape']

# Make a list of docs
patterns = [nlp.make_doc(text) for text in terms_list]

matcher.add("phrase_matcher", None, *patterns)


# Matcher Object
fictional_char_doc = nlp("""Superman (first appearance: 1938)  Created by Jerry Siegal and Joe Shuster for Action Comics #1 (DC Comics).Mickey Mouse (1928)  Created by Walt Disney and Ub Iworks for Steamboat Willie.Bugs Bunny (1940)  Created by Warner Bros and originally voiced by Mel Blanc.Batman (1939) Created by Bill Finger and Bob Kane for Detective Comics #27 (DC Comics).
Dorothy Gale (1900)  Created by L. Frank Baum for novel The Wonderful Wizard of Oz. Later portrayed by Judy Garland in the 1939 film adaptation.Darth Vader (1977) Created by George Lucas for Star Wars IV: A New Hope.The Tramp (1914)  Created and portrayed by Charlie Chaplin for Kid Auto Races at Venice.Peter Pan (1902)  Created by J.M. Barrie for novel The Little White Bird.
Indiana Jones (1981)  Created by George Lucas for Raiders of the Lost Ark. Portrayed by Harrison Ford.Rocky Balboa (1976)  Created and portrayed by Sylvester Stallone for Rocky.Vito Corleone (1969) Created by Mario Puzo for novel The Godfather. Later portrayed by Marlon Brando and Robert DeNiro in Coppola’s film adaptation.Han Solo (1977) Created by George Lucas for Star Wars IV: A New Hope. 
Portrayed most famously by Harrison Ford.Homer Simpson (1987)  Created by Matt Groening for The Tracey Ullman Show, later The Simpsons as voiced by Dan Castellaneta.Archie Bunker (1971) Created by Norman Lear for All in the Family. Portrayed by Carroll O’Connor.Norman Bates (1959) Created by Robert Bloch for novel Psycho.  Later portrayed by Anthony Perkins in Hitchcock’s film adaptation.King Kong (1933) 
Created by Edgar Wallace and Merian C Cooper for the film King Kong.Lucy Ricardo (1951) Portrayed by Lucille Ball for I Love Lucy.Spiderman (1962)  Created by Stan Lee and Steve Ditko for Amazing Fantasy #15 (Marvel Comics).Barbie (1959)  Created by Ruth Handler for the toy company Mattel Spock (1964)  Created by Gene Roddenberry for Star Trek. Portrayed most famously by Leonard Nimoy.
Godzilla (1954) Created by Tomoyuki Tanaka, Ishiro Honda, and Eiji Tsubaraya for the film Godzilla.The Joker (1940)  Created by Jerry Robinson, Bill Finger, and Bob Kane for Batman #1 (DC Comics)Winnie-the-Pooh (1924)  Created by A.A. Milne for verse book When We Were Young.Popeye (1929)  Created by E.C. Segar for comic strip Thimble Theater (King Features).Tarzan (1912) Created by Edgar Rice Burroughs for the novel Tarzan of the Apes.Forrest Gump (1986)  Created by Winston Groom for novel Forrest Gump.  Later portrayed by Tom Hanks in Zemeckis’ film adaptation.Hannibal Lector (1981)  Created by Thomas Harris for the novel Red Dragon. Portrayed most famously by Anthony Hopkins in the 1991 Jonathan Demme film The Silence of the Lambs.
Big Bird (1969) Created by Jim Henson and portrayed by Carroll Spinney for Sesame Street.Holden Caulfield (1945) Created by J.D. Salinger for the Collier’s story “I’m Crazy.”  Reworked into the novel The Catcher in the Rye in 1951.Tony Montana (1983)  Created by Oliver Stone for film Scarface.  Portrayed by Al Pacino.Tony Soprano (1999)  Created by David Chase for The Sopranos. Portrayed by James Gandolfini.
The Terminator (1984)  Created by James Cameron and Gale Anne Hurd for The Terminator. Portrayed by Arnold Schwarzenegger.Jon Snow (1996)  Created by George RR Martin for the novel The Game of Thrones.  Portrayed by Kit Harrington.Charles Foster Kane (1941)  Created and portrayed by Orson Welles for Citizen Kane.Scarlett O’Hara (1936)  Created by Margaret Mitchell for the novel Gone With the Wind. Portrayed most famously by Vivien Leigh 
for the 1939 Victor Fleming film adaptation.Marty McFly (1985) Created by Robert Zemeckis and Bob Gale for Back to the Future. Portrayed by Michael J. Fox.Rick Blaine (1940)  Created by Murray Burnett and Joan Alison for the unproduced stage play Everybody Comes to Rick’s. Later portrayed by Humphrey Bogart in Michael Curtiz’s film adaptation Casablanca.Man With No Name (1964)  Created by Sergio Leone for A Fistful of Dollars, which was adapted from a ronin character in Kurosawa’s Yojimbo (1961).  Portrayed by Clint Eastwood.Charlie Brown (1948)  Created by Charles M. Shultz for the comic strip L’il Folks; popularized two years later in Peanuts.E.T. (1982)  Created by Melissa Mathison for the film E.T.: the Extra-Terrestrial.Arthur Fonzarelli (1974)  Created by Bob Brunner for the show Happy Days. Portrayed by Henry Winkler.)Phillip Marlowe (1939)  Created by Raymond Chandler for the novel The Big Sleep.Jay Gatsby (1925)  Created by F. Scott Fitzgerald for the novel The Great Gatsby.Lassie (1938) Created by Eric Knight for a Saturday Evening Post story, later turned into the novel Lassie Come-Home in 1940, film adaptation in 1943, and long-running television show in 1954.  Most famously portrayed by the dog Pal.
Fred Flintstone (1959)  Created by William Hanna and Joseph Barbera for The Flintstones. Voiced most notably by Alan Reed. Rooster Cogburn (1968)  Created by Charles Portis for the novel True Grit. Most famously portrayed by John Wayne in the 1969 film adaptation. Atticus Finch (1960)  Created by Harper Lee for the novel To Kill a Mockingbird.  (Appeared in the earlier work Go Set A Watchman, though this was not published until 2015)  Portrayed most famously by Gregory Peck in the Robert Mulligan film adaptation. Kermit the Frog (1955)  Created and performed by Jim Henson for the show Sam and Friends. Later popularized in Sesame Street (1969) and The Muppet Show (1976) George Bailey (1943)  Created by Phillip Van Doren Stern (then as George Pratt) for the short story The Greatest Gift. Later adapted into Capra’s It’s A Wonderful Life, starring James Stewart as the renamed George Bailey. Yoda (1980) Created by George Lucas for The Empire Strikes Back. Sam Malone (1982)  Created by Glen and Les Charles for the show Cheers.  Portrayed by Ted Danson. Zorro (1919)  Created by Johnston McCulley for the All-Story Weekly pulp magazine story The Curse of Capistrano.Later adapted to the Douglas Fairbanks’ film The Mark of Zorro (1920).Moe, Larry, and Curly (1928)  Created by Ted Healy for the vaudeville act Ted Healy and his Stooges. Mary Poppins (1934)  Created by P.L. Travers for the children’s book Mary Poppins. Ron Burgundy (2004)  Created by Will Ferrell and Adam McKay for the film Anchorman: The Legend of Ron Burgundy.  Portrayed by Will Ferrell. Mario (1981)  Created by Shigeru Miyamoto for the video game Donkey Kong. Harry Potter (1997)  Created by J.K. Rowling for the novel Harry Potter and the Philosopher’s Stone. The Dude (1998)  Created by Ethan and Joel Coen for the film The Big Lebowski. Portrayed by Jeff Bridges.
Gandalf (1937)  Created by J.R.R. Tolkien for the novel The Hobbit. The Grinch (1957)  Created by Dr. Seuss for the story How the Grinch Stole Christmas! Willy Wonka (1964)  Created by Roald Dahl for the children’s novel Charlie and the Chocolate Factory. The Hulk (1962)  Created by Stan Lee and Jack Kirby for The Incredible Hulk #1 (Marvel Comics) Scooby-Doo (1969)  Created by Joe Ruby and Ken Spears for the show Scooby-Doo, Where Are You! George Costanza (1989)  Created by Larry David and Jerry Seinfeld for the show Seinfeld.  Portrayed by Jason Alexander.Jules Winfield (1994)  Created by Quentin Tarantino for the film Pulp Fiction. Portrayed by Samuel L. Jackson. John McClane (1988)  Based on the character Detective Joe Leland, who was created by Roderick Thorp for the novel Nothing Lasts Forever. Later adapted into the John McTernan film Die Hard, starring Bruce Willis as McClane. Ellen Ripley (1979)  Created by Don O’cannon and Ronald Shusett for the film Alien.  Portrayed by Sigourney Weaver. Ralph Kramden (1951)  Created and portrayed by Jackie Gleason for “The Honeymooners,” which became its own show in 1955.Edward Scissorhands (1990)  Created by Tim Burton for the film Edward Scissorhands.  Portrayed by Johnny Depp.Eric Cartman (1992)  Created by Trey Parker and Matt Stone for the animated short Jesus vs Frosty.  Later developed into the show South Park, which premiered in 1997.  Voiced by Trey Parker.
Walter White (2008)  Created by Vince Gilligan for Breaking Bad.  Portrayed by Bryan Cranston. Cosmo Kramer (1989)  Created by Larry David and Jerry Seinfeld for Seinfeld.  Portrayed by Michael Richards.Pikachu (1996)  Created by Atsuko Nishida and Ken Sugimori for the Pokemon video game and anime franchise.Michael Scott (2005)  Based on a character from the British series The Office, created by Ricky Gervais and Steven Merchant.  Portrayed by Steve Carell.Freddy Krueger (1984)  Created by Wes Craven for the film A Nightmare on Elm Street. Most famously portrayed by Robert Englund.
Captain America (1941)  Created by Joe Simon and Jack Kirby for Captain America Comics #1 (Marvel Comics)Goku (1984)  Created by Akira Toriyama for the manga series Dragon Ball Z.Bambi (1923)  Created by Felix Salten for the children’s book Bambi, a Life in the Woods. Later adapted into the Disney film Bambi in 1942.Ronald McDonald (1963) Created by Williard Scott for a series of television spots.Waldo/Wally (1987) Created by Martin Hanford for the children’s book Where’s Wally? (Waldo in US edition) Frasier Crane (1984)  Created by Glen and Les Charles for Cheers.  Portrayed by Kelsey Grammar.Omar Little (2002)  Created by David Simon for The Wire.Portrayed by Michael K. Williams.
Wolverine (1974)  Created by Roy Thomas, Len Wein, and John Romita Sr for The Incredible Hulk #180 (Marvel Comics) Jason Voorhees (1980)  Created by Victor Miller for the film Friday the 13th. Betty Boop (1930)  Created by Max Fleischer and the Grim Network for the cartoon Dizzy Dishes. Bilbo Baggins (1937)  Created by J.R.R. Tolkien for the novel The Hobbit. Tom Joad (1939)  Created by John Steinbeck for the novel The Grapes of Wrath. Later adapted into the 1940 John Ford film and portrayed by Henry Fonda.Tony Stark (Iron Man) (1963)  Created by Stan Lee, Larry Lieber, Don Heck and Jack Kirby for Tales of Suspense #39 (Marvel Comics)Porky Pig (1935)  Created by Friz Freleng for the animated short film I Haven’t Got a Hat. Voiced most famously by Mel Blanc.Travis Bickle (1976)  Created by Paul Schrader for the film Taxi Driver. Portrayed by Robert De Niro.
Hawkeye Pierce (1968)  Created by Richard Hooker for the novel MASH: A Novel About Three Army Doctors.  Famously portrayed by both Alan Alda and Donald Sutherland. Don Draper (2007)  Created by Matthew Weiner for the show Mad Men.  Portrayed by Jon Hamm. Cliff Huxtable (1984)  Created and portrayed by Bill Cosby for The Cosby Show. Jack Torrance (1977)  Created by Stephen King for the novel The Shining. Later adapted into the 1980 Stanley Kubrick film and portrayed by Jack Nicholson. Holly Golightly (1958)  Created by Truman Capote for the novella Breakfast at Tiffany’s.  Later adapted into the 1961 Blake Edwards films starring Audrey Hepburn as Holly. Shrek (1990)  Created by William Steig for the children’s book Shrek! Later adapted into the 2001 film starring Mike Myers as the titular character. Optimus Prime (1984)  Created by Dennis O’Neil for the Transformers toy line.Sonic the Hedgehog (1991)  Created by Naoto Ohshima and Yuji Uekawa for the Sega Genesis game of the same name.Harry Callahan (1971)  Created by Harry Julian Fink and R.M. Fink for the movie Dirty Harry.  Portrayed by Clint Eastwood.Bubble: Hercule Poirot, Tyrion Lannister, Ron Swanson, Cercei Lannister, J.R. Ewing, Tyler Durden, Spongebob Squarepants, The Genie from Aladdin, Pac-Man, Axel Foley, Terry Malloy, Patrick Bateman
Pre-20th Century: Santa Claus, Dracula, Robin Hood, Cinderella, Huckleberry Finn, Odysseus, Sherlock Holmes, Romeo and Juliet, Frankenstein, Prince Hamlet, Uncle Sam, Paul Bunyan, Tom Sawyer, Pinocchio, Oliver Twist, Snow White, Don Quixote, Rip Van Winkle, Ebenezer Scrooge, Anna Karenina, Ichabod Crane, John Henry, The Tooth Fairy,
Br’er Rabbit, Long John Silver, The Mad Hatter, Quasimodo """)


character_matches = matcher(fictional_char_doc)


# Matching positions
character_matches


# Matched items
for match_id, start, end in character_matches:
    span = fictional_char_doc[start:end]
    print(span.text)


# Using the attr parameter as 'LOWER'
case_insensitive_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Creating doc &amp; pattern
my_doc=nlp('I wish to visit new york city')
terms=['New York']
pattern=[nlp(term) for term in terms]

# adding pattern to the matcher
case_insensitive_matcher.add("matcher",None,*pattern)

# applying matcher to the doc
my_matches=case_insensitive_matcher(my_doc)

for match_id,start,end in my_matches:
    span=my_doc[start:end]
    print(span.text)




