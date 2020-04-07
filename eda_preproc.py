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
df_india.head(20)

# %%

df_india.columns

# %%

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=3, norm='l2', 
        encoding='latin-1', ngram_range=(1, 2), stop_words='english')


# %%
features = tfidf.fit_transform(df_india.new_feature).toarray()
labels = df_india.Flair
features.shape

# %%

from sklearn.feature_selection import chi2
import numpy as np
# test