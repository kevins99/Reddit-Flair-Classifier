# %%

import pandas as pd
import tqdm 
import numpy as np 

import nltk 
import re

import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from praw.models import MoreComments

# %%

df_india = pd.read_csv('clean_reddit_india.csv')
df_india.head()

# %%
df_india.columns
df_india.drop(columns=['Unnamed: 0'], inplace=True)
df_india = df_india.rename(columns={'Unnamed: 0.1':'Index'})


# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_india['Flair_cat'] = le.fit_transform(df_india['Flair'])

# %%
df_india['Flair_cat'].value_counts()


# %%
df_india.columns

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_india['new_feature'],
                                                    df_india['Flair_cat'],
                                                    test_size=0.2)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', 
        encoding='latin-1', ngram_range=(1, 2), stop_words='english')

# %%
features = tfidf.fit_transform(df_india.new_feature).toarray()
labels = df_india.Flair_cat
print(features.shape)

features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.fit_transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)



# %%
from sklearn.feature_selection import chi2
import numpy as np

column_values = df_india['Flair_cat'].values
flair_list = pd.unique(column_values)

N = 5
for flair_cat in sorted(flair_list):
    print(flair_cat)
    features_chi2 = chi2(features, labels==flair_cat)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [word for word in feature_names if len(word.split(' '))==1]
    bigrams = [word for word in feature_names if len(word.split(' '))==2]
    flair = df_india.loc[lambda df_india:df_india['Flair_cat']==flair_cat]
    print(" '{}':".format(flair['Flair'].iloc[0]))
    print("Most correlated unigrams: \n{}".format('\n'.join(unigrams[-N:])))
    print("Most correlated bigrams: \n{}".format('\n'.join(bigrams[-N:])))

    
# %%
df_india['content_len'] = df_india['new_feature'].str.len()
sns.distplot(df_india['content_len']).set_title('Length')

# %%
df_india['content_len'].describe()

# %%
quantile_95 = df_india['content_len'].quantile(0.95)
df_95 = df_india[df_india['content_len']<quantile_95]

# %%
sns.distplot(df_95['content_len']).set_title('Length')

# %%
sns.boxplot(data=df_india, x='content_len', y='Flair', width=.5)


# %%
sns.boxplot(data=df_95, x='content_len', y='Flair', width=.5)

# %%
with open('pickles/clean_95.pkl', 'wb') as f:
    pickle.dump(df_95, f)

with open('pickles/X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('pickles/X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('pickles/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('pickles/y_test.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('pickles/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('pickles/features_train.pkl', 'wb') as f:
    pickle.dump(features_train, f)

with open('pickles/features_test.pkl', 'wb') as f:
    pickle.dump(features_test, f)

with open('Pickles/labels_train.pickle', 'wb') as f:
    pickle.dump(labels_train, f)

with open('Pickles/labels_test.pkl', 'wb') as f:
    pickle.dump(labels_test, f)
# %%
