#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Get the data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']

path = "C:\\bhargav\\edu\\Bhar_eng\\6th sem\\file.tsv"

df = pd.read_csv(path, sep='\t', names=column_names)

# Check the head of the data
df.head()


# In[4]:


movie_titles = pd.read_csv('C:\\bhargav\\edu\\Bhar_eng\\6th sem\\Movie_Id_Titles.csv')
movie_titles.head()


# In[6]:


data = pd.merge(df, movie_titles, on='item_id')
data.head()


# In[8]:


# Calculate mean rating of all movies
data.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[10]:


# Calculate count rating of all movies
data.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[12]:


# creating dataframe with 'rating' count values
ratings = pd.DataFrame(data.groupby('title')['rating'].mean()) 

ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())

ratings.head()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


plt.figure(figsize=(10, 4))
plt.hist(ratings['num of ratings'], bins=70, color='blue', edgecolor='black')
plt.xlabel("Number of Ratings")
plt.ylabel("Count")
plt.title("Distribution of Number of Ratings per Movie")
plt.show()



# In[20]:


# Sorting values according to 
# the 'num of rating column'
moviemat = data.pivot_table(index ='user_id',
              columns ='title', values ='rating')

moviemat.head()

ratings.sort_values('num of ratings', ascending = False).head(10)


# In[26]:


# analysing correlation with similar movies
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

starwars_user_ratings.head()


# In[30]:


# analysing correlation with similar movies
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars, columns =['Correlation'])
corr_starwars.dropna(inplace = True)

corr_starwars.head()


# In[32]:


# Similar movies like starwars
corr_starwars.sort_values('Correlation', ascending = False).head(10)
corr_starwars = corr_starwars.join(ratings['num of ratings'])

corr_starwars.head()

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation', ascending = False).head()


# In[34]:


# Similar movies as of liarliar
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns =['Correlation'])
corr_liarliar.dropna(inplace = True)

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation', ascending = False).head()

