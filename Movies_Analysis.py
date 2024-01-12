#!/usr/bin/env python
# coding: utf-8

# ## Movies Analysis
# 
# ## 1. importing a necessary libraries to work with the dataset.
# ## 2. importing the dataset .
# ## 3. checking details of Dataset using info() and describe() for basic understanding about the data.
# ## 4. checking is there null value in the data using "isnull().sum()" method.
# ## 5. checking the null value percentage using "df.null().mean() * 100)" function.
# ## 6. treating missing values using some basic statistics method.
# ## 7. checking the outliers using boxplot for datapoint understandiing.
# ## 8. for Target Variable we can consider 'Revenue', because based on dataset, its most important.

# In[1]:


#Importing Necessary libraries for Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


# In[2]:


# loading the dataset:

file_path = 'D:\\New folder\\Task_Kovai\\movies_original.csv'
df = pd.read_csv(file_path)


# In[3]:


df


# In[4]:


# checking the shape of the dataset

df.shape


# In[5]:


# Basic information about the dataset

print("Dataset Info:")
print(df.info())


# In[6]:


# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())


# In[7]:


# checking the missing values
df.isnull().sum()


# In[8]:


# Display the percentage of missing values for each column
missing_percentage = (df.isnull().mean() * 100).round(2)
missing_percentage


# In[9]:


# changing the Date type of Release date and datetime

df['release_date'] = pd.to_datetime(df['release_date'])


# In[10]:


df.info()


# In[11]:


# in revenue and budget most of the data were 0 as a value, so I plnaeed to count the data 

bud_count = (df['budget'] == 0).sum()
rev_count = (df['revenue'] == 0).sum()

print(bud_count)
print(rev_count)


# ## considering the Industry entertainment (movie) we can keep this as same value, for better understand their impact, 
# ## but with the amount of below 50000 we can create a simple add but not a movie, so I am just removing budget which is below 50,000.
# 

# In[12]:


df = df[df['budget'] >= 50000]


# In[13]:


df.info()


# In[14]:


df = df[df['revenue'] >= 5000]


# In[15]:


df.info()


# ## Treating the missing values 

# In[16]:


# Treating categorical missing values 

mode_genres = df['genres'].mode()[0]
df['genres'].fillna(mode_genres, inplace = True)


# In[17]:


mode_overview = df['overview'].mode()[0]
df['overview'].fillna(mode_overview, inplace = True)


# In[18]:


mode_production_companies = df['production_companies'].mode()[0]
df['production_companies'].fillna(mode_production_companies, inplace = True)


# In[19]:


mode_tagline = df['tagline'].mode()[0]
df['tagline'].fillna(mode_tagline, inplace = True)


# In[20]:


mode_credits = df['credits'].mode()[0]
df['credits'].fillna(mode_credits, inplace = True)


# In[21]:


mode_keywords = df['keywords'].mode()[0]
df['keywords'].fillna(mode_keywords, inplace = True)


# In[22]:


# treating mising values for runtime

mean_runtime = df['runtime'].mean()
df['runtime'].fillna(mean_runtime, inplace= True)


# In[23]:


# checking the missing values count after the categorical missing Treatment

df.isnull().sum()


# ## treating the missing values in the "date" is not the best practice, cause the dates are unique for the releasing, so we are dropping the dates missing values, since we have lot other values are treated we are droping this
# 

# In[24]:


# to drop missing values in the release_date

df = df.dropna(subset = ['release_date'])


# In[25]:


df.isnull().sum()


# In[26]:


# after completing all the preporcessing lets check with the head and tail method to verify the cleaned data

df.head(10)


# In[27]:


df.tail(10)


# In[28]:


# checking outliers using boxplot.
plt.figure(figsize=(10,5))
plt.show(df.boxplot())


# In[29]:


# Treating Outliers
threshold_value = 1000000000

# Identify and print rows with revenue exceeding the threshold
outliers = df[df['revenue'] > threshold_value]
print(f"Number of outliers above the threshold: {len(outliers)}")


# In[30]:


# Create bins and assign 'revenue' to bins
bins = [0, 1000000, 50000000, 100000000, float('inf')]
labels = ['Low', 'Medium', 'High', 'Very High']
df['revenue_bins'] = pd.cut(df['revenue'], bins=bins, labels=labels)


# In[31]:


print(df.describe())


# ## All the missing values were treated, we can Explore our data with the detailed Analysis 

# ## EDA
# 

# In[32]:


# Distribution of Numerical values
df.hist(figsize=(15,10))
plt.show()


# In[33]:


# Univerient Analysis of Vote_average

plt.figure(figsize=(8, 5))
sns.histplot(df['budget'], bins=20, kde=True)
plt.title('Distribution of budget')
plt.show()


# In[34]:


# Univerient Analysis of Revenue

plt.figure(figsize=(8, 5))
sns.histplot(df['revenue'], bins=20, kde=True)
plt.title('Distribution of Revenue')
plt.show()


# In[35]:


# Bivarient Analysis Budget and Revenue

plt.figure(figsize=(8, 5))
sns.scatterplot(x='budget', y='revenue', data=df)
plt.title('Budget vs Revenue')
plt.show()


# In[36]:


# Bivarient Analysis with Original_value vs Vote_Average

plt.figure(figsize=(15,5))
sns.boxplot(x='original_language', y='vote_average', data=df)
plt.show()


# # correlation matrix

# In[37]:


# Correlation analysis
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()


# In[65]:


sns.regplot(x='budget', y='revenue', data=df)
plt.title('Correlation between Budget and Revenue')
plt.xlabel('Budget')
plt.ylabel('Revenue')
plt.show()

# Correlation coefficient

correlation_coefficient = df['budget'].corr(df['revenue'])
print(f'Correlation Coefficient: {correlation_coefficient}')


# In[39]:


# Budget and Revenue Relationship

fig = px.scatter(df, x='budget', y='revenue', color='status',
                 title='Budget vs. Revenue with Color Differentiation by Status',
                 labels={'budget': 'Budget', 'revenue': 'Revenue'})

fig.show()


# In[40]:


# Release Date Analysis
df['release_year'] = pd.to_datetime(df['release_date']).dt.year
plt.figure(figsize=(25, 13))
sns.countplot(x='release_year', data=df, palette='viridis', order=df['release_year'].value_counts().index)
plt.title('Number of Movies Released Each Year')
plt.xticks(rotation=90)
plt.show()


# In[41]:


# Average revenue per year
fig_line_tooltip = px.line(df.groupby('release_year')['revenue'].mean().reset_index(),
                           x='release_year', y='revenue',
                           labels={'release_year': 'Release Year', 'revenue': 'Average Revenue'},
                           title='Average Revenue Over the Years',
                           hover_data={'revenue': ':,.2f'})
fig_line_tooltip.show()


# In[59]:


# Word cloud for tagline and overview

from wordcloud import WordCloud

text_tagline = ' '.join(df['tagline'].dropna())
text_overview = ' '.join(df['overview'].dropna())

wordcloud_tagline = WordCloud(width=1000, height=600, background_color='white').generate(text_tagline)
wordcloud_overview = WordCloud(width=1000, height=600, background_color='white').generate(text_overview)

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_tagline, interpolation='bilinear')
plt.axis('off')
plt.title('Tagline Word Cloud')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_overview, interpolation='bilinear')
plt.axis('off')
plt.title('Overview Word Cloud')

plt.show()


# In[43]:


# Year wise release and Count of the movies 

df['release_year'] = pd.to_datetime(df['release_date']).dt.year

# Create a countplot with Plotly Express
fig = px.histogram(df, x='release_year', color_discrete_sequence=['#0000ff'],
                   category_orders={'release_year': sorted(df['release_year'].unique())})
fig.update_layout(title='Number of Movies Released Each Year',
                  xaxis_title='Release Year',
                  yaxis_title='Count',
                  showlegend=False)  # No need to show legend for a countplot

# Add count numbers on top of each bar
fig.update_traces(texttemplate='%{y}', textposition='outside')

# Show the interactive plot
fig.show()


# In[44]:


# Rating by Revenue using Plotly 

fig_scatter = px.scatter(df, x='vote_average', y='vote_count', color='revenue',
                         title='Vote Average vs. Vote Count with Color Differentiation by Revenue',
                         labels={'vote_average': 'Vote Average', 'vote_count': 'Vote Count'})

fig_scatter.show()


# In[45]:


# Rating by Budget using Plotly 

fig_scatter = px.scatter(df, x='vote_average', y='vote_count', color='budget',
                         title='Vote Average vs. Vote Count with Color Differentiation by Budget',
                         labels={'vote_average': 'Vote Average', 'vote_count': 'Vote Count'})

fig_scatter.show()


# In[49]:


# Runtime Analysis

sns.histplot(df['runtime'], bins=30, kde=True)
plt.title('Distribution of Runtime')
plt.show()


# In[63]:


# Release Date Trends

df['release_month'] = df['release_date'].dt.month
monthly_revenue = df.groupby('release_month')['revenue'].mean()
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o')
plt.title('Monthly Revenue Trends')
plt.xlabel('Month')
plt.ylabel('Average Revenue')
plt.show()


# In[64]:


# Top production companies Analysis count of 15

top_production_companies = df['production_companies'].str.split(',', expand=True).stack().value_counts().head(15)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_production_companies.values, y=top_production_companies.index)
plt.title('Top Production Companies')
plt.xlabel('Count')
plt.show()


# In[53]:


# Voters Analysis 

sns.scatterplot(x='vote_average', y='vote_count', data=df)
plt.title('Vote Average vs. Vote Count')
plt.show()


# In[55]:


# Revenue Bins Analysis

plt.figure(figsize=(8, 4))
sns.histplot(df['revenue_bins'], bins=20)
plt.title('Distribution of Revenue Bins')
plt.show()


# In[ ]:




