#!/usr/bin/env python
# coding: utf-8

# # Example Usage of TopicGPT: Amazon Reviews

# In this notebook, we will be using the Amazon Reviews dataset to show how TopicGPT can be useful when analyzing a large corpus of text.

# In[1]:


from topicgpt.TopicGPT import TopicGPT


# In[5]:


# load api key
import os
api_key_openai = os.environ.get('OPENAI_API_KEY')

import openai

openai.organization = "org-MOfdTrYSke1pXhlAdLXxwDKx"


# ### Load data

# In[3]:


import pandas as pd


# In[4]:


# data from https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?resource=download

review_data = pd.read_csv("../Data/AmazonReviews/amazon_review_polarity_csv/train.csv", header=None) # only use the first 10k reviews of the train set

reviews = list(review_data[2])
reviews = reviews[:10000] # only consider the first 10k reviews 


# In[5]:


tm = TopicGPT(
    openai_api_key = api_key_openai,
    corpus_instruction= "The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including 10000 reviews up to March 2013."
)


# In[ ]:


tm.fit(reviews)


# In[13]:


tm.save_embeddings()  #save the computed embeddings for later use


# In[ ]:


tm.visualize_clusters()


# In[ ]:


tm.topic_lis


# In[1]:


# load the model if available
import pickle
with open("../Data/SavedTopicRepresentations/TopicGPT_amazonReviews.pkl", "rb") as f:
    tm = pickle.load(f)


# Let's see what topic 2 is about

# In[3]:


print(tm.topic_lis[2].topic_description)


# In[8]:


tm.pprompt("Is the movie Avatar mentioned in topic 2?")


# To check the output, we actually inspect the respective document at index 1498: 

# In[9]:


print(tm.topic_lis[2].documents[1498])


# Let us go own with the analysis. Since it is easy to loose the overview over all the topics, lets find out which one is about books

# In[10]:


tm.pprompt("Which topic is about books?")


# In[11]:


print(tm.topic_lis[5].topic_description)


# In[12]:


tm.pprompt("Is Harry Potter mentioned in topic 5?")


# Topic 0 ("Musical genres and characteristics") sounds a bit general and from the visual inspection it seems to contain a lot of documents. So let's break it down a little bit

# In[13]:


tm.pprompt("please split topic 0 into subtopics. Do this inplace")


# We see that the topic 0 was split into two topics. One on music genres and the other one on remastering of music. 

# On the other hand, topic 0 and topic 2 seem very similar. Let's find out more about the difference between the two: 

# In[ ]:


tm.pprompt("What are the differences and similarities of topic 0 and topic 2?")


# The topics seem fairly related, so we can merge them into one topic.

# In[6]:


tm.pprompt("please merge topic 0 and topic 2. Do this inplace")


# Topics 39 and 44 seem very similar, so let's find out their difference

# In[19]:


tm.pprompt("What are the differences and similarities of topic 39 and topic 44?")


# Since the topics seem sufficiently similar, we will combine them 

# In[21]:


tm.pprompt("please merge topic 39 and topic 44. Do this inplace")


# In[ ]:




