#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# - Tutorial source: https://docs.tweepy.org/en/latest/getting_started.html
# - Original plan: https://realpython.com/twitter-sentiment-python-docker-elasticsearch-kibana/#twitter-streaming-api
#  - Later found out that this was obsolote tutorial, from more than 5 years ago, with old docker version

# # Hello Tweepy

from config_twitter import *
import tweepy

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)


# ## Models

# Get the User object for twitter...
user = api.get_user('twitter')

print(user.screen_name)
print(user.followers_count)
for friend in user.friends():
    print(friend.screen_name)


# # Streaming with Tweepy

# ## Create StreamListener

import tweepy
#override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):

    def on_status(self, status):
        print(status.text)


# ## Creating a Stream

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)


# ## Starting a Stream

myStream.filter(track=['python'])
myStream.filter(follow=["enlik"])











































