#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:40:55 2019
@author: amitmalik
"""
# In[8]:
import pandas as pd
# In[52]:

quora = pd.read_csv('quora_questions.csv',encoding='ISO-8859-1')
quora['Question'] = quora['Question']

# In[53]:
quora.head()
# # Preprocessing 
# #### Task: Use TF-IDF Vectorization to create a vectorized document term matrix. You may want to explore the max_df and min_df parameters.
# In[40]:
from sklearn.feature_extraction.text import TfidfVectorizer
# In[41]:
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',encoding = "ISO-8859-1")
# In[42]:
dtm = tfidf.fit_transform(quora['Question'])
# In[43]:
dtm
# # Non-negative Matrix Factorization 
# #### TASK: Using Scikit-Learn create an instance of NMF with 20 expected components. (Use random_state=42)..
# In[44]:
from sklearn.decomposition import NMF
# In[48]:
nmf_model = NMF(n_components=20,random_state=42)
# In[49]:
nmf_model.fit(dtm)
# #### TASK: Print our the top 15 most common words for each of the 20 topics.
# In[50]:
for index,Question in enumerate(nmf_model.components_):
    print('THE TOP 15 WORDS FOR Question '.format(index))
    print([tfidf.get_feature_names()[i] for i in Question.argsort()[-15:]])
    print('\n')
# #### TASK: Add a new column to the original quora dataframe that labels each question into one of the 20 topic categories.
# In[54]:
quora.head()
# In[55]:
topic_results = nmf_model.transform(dtm)
# In[56]:
topic_results.argmax(axis=1)

quora['Topic'] = topic_results.argmax(axis=1)

quora.head(10)
 
# In[50]:
quora.Topic = quora.Topic.astype(str)
label ={'0' : 'Sci-Fi Movies', '1' : 'music broke', '2' : 'battle_gaming', '3' : 'funny', '4' : 'gaming', '5' : 'movie_game', '6' : 'game', '7' : 'sci-fi', '8' : 'buddha_snake_sword', '9':'gaming_indoor', '10':'GOT_fiction_TV', '11':'racing_sports_games','12':'trailers','13':'seasons_gameplay_festival','14':'News_cooking_bloopers','15':'sports_moments_outdoor','16':'car_sports_racing', '17':'explained-sci-fi', '18':'gaming console', '19':'discover/documentary'} 
quora['category'] = quora['Topic'].map(label) 

# In[56]: 
#print(quora['category','Topic','genres']) 
i=input("ur interests")
quora[quora.category == i]  

#quora.title[quora.category == i]
#rslt_df = quora.loc[quora['category'].isin(i)] 
#print(rslt_df) 
#quora['topic','category']
#quora['meta_title']
# In[56]:
'''
df1 = df[df.index.str.contains('ane')]
 
print(df1)
'''
# # Great job!
