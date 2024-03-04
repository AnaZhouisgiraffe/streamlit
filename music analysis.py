#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import plotly.graph_objects as go
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


pd.set_option('display.max_rows', 100)  # 显示所有行
pd.set_option('display.max_columns', 100)


# In[3]:


df=pd.read_csv('C:/Users/anaru/Emlyon DMDS/SpotifyWeb/data/universal_top_spotify_songs.csv')


# In[4]:


df.head()


# In[5]:


keys_and_mode= df.groupby(['key', 'mode'])[['mode']].count().unstack().reset_index()
keys_and_mode


# In[6]:


def key_full(key):
    key_mapping = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'Eb', 4: 'E',
        5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'Bb', 11: 'B'
    }
    return key_mapping.get(key, key)

df['key'] = df['key'].apply(key_full)  # Apply the function to the 'key' column in df

# Assuming keys_and_mode is another DataFrame
keys_and_mode['key'] = keys_and_mode['key'].apply(key_full)  # Apply the function to the 'key' column in keys_and_mode
keys_and_mode


# In[10]:



# Replace this with your actual DataFrame
keys_and_mode_data = {
    'key': ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'],
    'mode_0': [12039, 20736, 18095, 9775, 22799, 23032, 30132, 21014, 13180, 15254, 17241, 26669],
    'mode_1': [25434, 41778, 30765, 3303, 7669, 25432, 14175, 23236, 24855, 30547, 12768, 14953]
}

keys_and_mode = pd.DataFrame(keys_and_mode_data)

# Melt the DataFrame for better compatibility with Plotly Express
keys_and_mode_melted = pd.melt(keys_and_mode, id_vars=['key'], var_name='mode', value_name='count')

# Map colors to modes and labels
colors = {'mode_0': '#1DB954', 'mode_1': '#C1E1C5'}
labels = {'mode_0': 'Minor', 'mode_1': 'Major'}

keys_and_mode_melted['color'] = keys_and_mode_melted['mode'].map(colors)
keys_and_mode_melted['mode_label'] = keys_and_mode_melted['mode'].map(labels)

# Create a scatter plot with size according to the count
fig = px.scatter(
    keys_and_mode_melted, x='key', y='count',
    size='count', color='mode_label', color_discrete_map=colors,
    labels={'count': 'Count', 'key': 'Key', 'mode_label': 'Mode'},
    title='Number of Keys the Songs Are In (Either Major or Minor)',
    template='plotly_dark',
    height=600,  # Increased plot height
    size_max=60,  # Increased maximum marker size
)

# Update the layout
fig.update_layout(
    font=dict(size=14, family="Franklin Gothic"),
    xaxis=dict(tickangle=45),  # Tilt the x-axis labels for better readability
    showlegend=True,
)

# Show the figure
fig.show()


# In[15]:


# Replace this with your actual DataFrame
keys_and_mode_data = {
    'key': ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B'],
    'mode_0': [12039, 20736, 18095, 9775, 22799, 23032, 30132, 21014, 13180, 15254, 17241, 26669],
    'mode_1': [25434, 41778, 30765, 3303, 7669, 25432, 14175, 23236, 24855, 30547, 12768, 14953]
}

keys_and_mode = pd.DataFrame(keys_and_mode_data)

# Calculate the total number of songs for each key and mode separately
keys_sum_minor = keys_and_mode['mode_0']
keys_sum_major = keys_and_mode['mode_1']

# Apply log transformation to the counts
log_counts_minor = np.log1p(keys_sum_minor)
log_counts_major = np.log1p(keys_sum_major)

# Normalize the log-transformed counts to a suitable range (e.g., 10-100) separately for major and minor
min_val_major, max_val_major = log_counts_major.min(), log_counts_major.max()
normalized_sizes_major = 10 + 90 * ((log_counts_major - min_val_major) / (max_val_major - min_val_major))

min_val_minor, max_val_minor = log_counts_minor.min(), log_counts_minor.max()
normalized_sizes_minor = 10 + 90 * ((log_counts_minor - min_val_minor) / (max_val_minor - min_val_minor))

# Create scatter plots for minor and major keys
minor_scatter = go.Scatter(
    name='Minor',
    x=keys_and_mode['key'],
    y=keys_and_mode['mode_0'].values,
    mode='markers',
    marker=dict(color='#C1E1C5', size=normalized_sizes_minor, line=dict(width=1, color='black'))
)

major_scatter = go.Scatter(
    name='Major',
    x=keys_and_mode['key'],
    y=keys_and_mode['mode_1'].values,
    mode='markers',
    marker=dict(color='#1DB954', size=normalized_sizes_major, line=dict(width=1, color='black'))
)

# Create the figure with scatter plots
fig = go.Figure(data=[minor_scatter, major_scatter])

# Update the layout
fig.update_layout(
    height=450,
    template='plotly_dark',
    font=dict(size=14, family="Franklin Gothic"),
    title='Number of Keys the Songs Are In (Either Major or Minor)',
    xaxis_title='Keys',
    yaxis_title='Count (log scale)',
    xaxis=dict(tickangle=45)  # Tilt the x-axis labels for better readability
)

# Show the figure
fig.show()


# In[13]:


import plotly.express as px

# Create box plot with custom green color
fig = px.box(df['popularity'], color=df['is_explicit'], template='plotly_dark',
             color_discrete_sequence=['#008000','#00ff00'])

# Update layout with dark background color and title
fig.update_layout(
    paper_bgcolor='rgb(30,30,30)',
    plot_bgcolor='rgb(30,30,30)',
    font=dict(size=14, family="Franklin Gothic", color='green'),
    legend=dict(font=dict(size=14, family="Franklin Gothic", color='green')),
    title=dict(text='Popularity of Explicit Songs', font=dict(size=20, family="Franklin Gothic", color='green'))
)

# Show plot
fig.show()


# In[ ]:


#We can see that explicit songs are likely to be more popular. Nevertheless, being explicit is not the most important factor.
#The median of popularity in explicit songs are higher than the median of popularity in non explicit songs. The Q1 also follows the median. The variety of popularity in explicit songs lower than the variety of popularity in non explicit songs


# In[14]:


px.scatter(df, x='loudness', y='energy', template='plotly_dark',
          color_discrete_sequence=['#1DB954'],
           title= 'Strong correlation between energy and loudness'
      ).update_layout(
        font = dict(size=14,family="Franklin Gothic"))


# 

# In[ ]:




