# %%

import pandas as pd
import tqdm 
import numpy as np 

import nltk 
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from praw.models import MoreComments

# %%

df_india = pd.read_csv('clean_reddit_india.csv')
df_india.head()

# %%
df_india.columns
df_india.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)


# %%
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_india['Flair_cat'] = le.fit_transform(df_india['Flair'])

# %%
df_india['Flair_cat'].value_counts()


# %%
df_india.columns

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', 
        encoding='latin-1', ngram_range=(1, 2), stop_words='english')

# %%
features = tfidf.fit_transform(df_india.new_feature).toarray()
labels = df_india.Flair_cat
features.shape

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
