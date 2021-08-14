#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries and data

# In[48]:


import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


# In[49]:


import plotly.express as px


# In[50]:


from typing import List


# In[51]:


import string


# In[52]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[53]:


from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize 


# In[54]:


df = pd.read_csv('spam.csv', encoding='latin-1')
df.head()


# ### Exploratory Data Analysis

# In[55]:


df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns = {'v1':'type','v2':'message'},inplace=True)
df.head()


# In[56]:


df.describe()


# In[57]:


df.groupby('type').describe()


# In[58]:


df.isna().sum()


# In[59]:


df.type.value_counts()


# In[60]:


df["type"].value_counts().plot(kind='pie',explode=[0,0.1],colors=["#FF7256","#FFB90F"],autopct= '%1.2f%%',figsize=(9,9))
plt.legend(['Ham','Spam'])
plt.xlabel('SPAM v/s HAM')
plt.show


# In[61]:


fig = px.histogram(df, x="type", color="type", color_discrete_sequence=["#7FFF00","#00FFFF"])
fig.show()


# In[62]:


df['lenght'] = df['message'].apply(len)
df.head()


# In[63]:


fig = px.histogram(df, x="lenght", color="type", color_discrete_sequence=["#00FF7F","#FFA500"] )
fig.show()


# ### Preprocessing and feature Extraction

# In[64]:


spam_msg=df[df['type']=='spam']['message']
ham_msg=df[df['type']=='ham']['message']
#to separate the spam & ham messages 
spam_msg.head()


# In[65]:


ham_msg.head()


# In[66]:


def clear_text(text_arr: List[str]) -> List[str]:
    lines = []
    for text in text_arr:
        table = str.maketrans('', '', string.punctuation)
        line = text.translate(table)
        line = line.lower()
        lines.append(line)
    return lines


# In[67]:


spam_words=[]
ham_words=[]
#making 2 empty lists for spam words & ham words 

def extractSpamwords(spam_msg):
    global spam_words                            
    words=[ word for word in word_tokenize(spam_msg)]
    spam_words= spam_words + words

def extractHamwords(ham_msg):
    global ham_words                 
    words=[ word for word in word_tokenize(ham_msg)]
    ham_words= ham_words + words    

p = spam_msg.apply(extractSpamwords)
q = ham_msg.apply(extractHamwords)


# In[68]:


spam_words = clear_text(spam_words)
ham_words = clear_text(ham_words)


# In[69]:


print(spam_words[:5], ham_words[:5])


# In[70]:


spam_wordcloud = WordCloud(background_color="#EECBAD", width=600, height=400).generate(" ".join(spam_words))
plt.figure(figsize=(10,8))
plt.imshow(spam_wordcloud)
plt.title('Spam messages', fontsize=20)
plt.axis('off')  
plt.show()


# In[71]:


ham_wordcloud = WordCloud(background_color="#EECBAD", width=600, height=400).generate(" ".join(ham_words))
plt.figure(figsize=(10,8))
plt.imshow(ham_wordcloud)
plt.title('Ham messages', fontsize=20)
plt.axis('off')   
plt.show()


# In[72]:


y = df.type
X = df.message
X.head()


# In[73]:


y = pd.get_dummies(y, drop_first=True)
y.head()


# Spam = 1 and Ham = 0

# In[74]:


X[0]


# In[75]:


X = clear_text(X)
X[:5]


# In[76]:


cv = CountVectorizer()
X = cv.fit_transform(X)


# In[77]:


X.toarray()[:2]


# ### Model Deployment

# In[78]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[79]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[80]:


y_pred = lr.predict(X_test)


# In[81]:


accuracy_score(y_test, y_pred)


# In[82]:


tr = DecisionTreeClassifier()
tr.fit(X_train, y_train)


# In[83]:


y_pred2 = tr.predict(X_test)
accuracy_score(y_test, y_pred2)


# In[84]:


rr = RandomForestClassifier()
rr.fit(X_train, y_train)
y_pred3 = rr.predict(X_test)
accuracy_score(y_test, y_pred3)


# In[85]:


cm = confusion_matrix(y_test, y_pred)


# In[86]:


plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='0', cbar=False, cmap='Pastel2', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Prediction')
plt.ylabel('Actual')


# In[87]:


lr.predict(X.toarray()[:3])


# ### Model Validation

# In[88]:


#Spam = 1 and Ham = 0
def predict_spam(txt: str) -> bool:
    txt = clear_text([txt])
    txt = cv.transform(txt)
    pred = lr.predict(txt)
    return pred[0] == 1


# In[89]:


predict_spam(df.message[2])


# In[90]:


text = "Congratulations! You've won a $1,000 Walmart gift card. Go to http://bit.ly/123345 to claim now."


# In[91]:


predict_spam(text)


# In[92]:


text = "Ann your free bitcoin account is waiting for you. Claim NOW and make up to AUD762 per day. Fully automated and designed for beginners."


# In[93]:


predict_spam(text)


# In[94]:


predict_spam('This is you captain speaking.')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




