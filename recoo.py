
# coding: utf-8

# In[7]:

import pandas as pd


# In[8]:

from sklearn.feature_extraction.text import TfidfVectorizer


# In[12]:

quora = pd.read_csv('quora_questions.csv',encoding='ISO-8859-1')
quora['Question'] = quora['Question']


# In[15]:

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english',encoding = "ISO-8859-1")


# In[16]:

dtm = tfidf.fit_transform(quora['Question'])


# In[17]:

dtm


# In[18]:

from sklearn.decomposition import NMF


# In[19]:

nmf_model = NMF(n_components=20,random_state=42)


# In[20]:

nmf_model.fit(dtm)


# In[22]:

for index,Question in enumerate(nmf_model.components_):
    print('THE TOP 15 WORDS FOR Question '.format(index))
    print([tfidf.get_feature_names()[i] for i in Question.argsort()[-15:]])
    print('\n')


# In[23]:

topic_results = nmf_model.transform(dtm)


# In[24]:

topic_results.argmax(axis=1)


# In[26]:

quora['Topic'] = topic_results.argmax(axis=1)


# In[27]:

quora.Topic = quora.Topic.astype(str)
label ={'0' : 'Sci-Fi Movies', '1' : 'music broke', '2' : 'battle_gaming', '3' : 'funny', '4' : 'gaming', '5' : 'movie_game', '6' : 'game', '7' : 'sci-fi', '8' : 'buddha_snake_sword', '9':'gaming_indoor', '10':'GOT_fiction_TV', '11':'racing_sports_games','12':'trailers','13':'seasons_gameplay_festival','14':'News_cooking_bloopers','15':'sports_moments_outdoor','16':'car_sports_racing', '17':'explained-sci-fi', '18':'gaming console', '19':'discover/documentary'} 
quora['category'] = quora['Topic'].map(label) 


# In[28]:

i=input("ur interests")
quora[quora.category == i]


# In[ ]:




# In[ ]:



