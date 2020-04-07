# %%
import praw
import pandas as pd
import datetime as dt
import nltk
import re
from tqdm import tqdm_notebook

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from praw.models import MoreComments


# %%
flairs = ["Scheduled", "Politics", "Photography", "AskIndia", "Sports",
        "Non-Political", "Science/Technology", "Food", "Business/Finance",
        "Coronavirus", "CAA-NRC-NPR"]


reddit = praw.Reddit(client_id='7hx8xkseiQkWNA', client_secret='UPd0tXn-Lt26mk4-Z38oCNLuG_8', user_agent='Kevin Stephen')
india = reddit.subreddit('india')

results = []

for flair in flairs:
    posts = india.search(flair, limit=100)

    for post in posts:
        comment = ''
        post.comment_sort = 'best'
        post.comment_limit = 3
        for top_level_comment in post.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            comment += top_level_comment.body
        results.append([post.title, post.score, post.id, post.url, post.num_comments, post.selftext, post.created, comment, flair])

results = pd.DataFrame(results ,columns=['title', 'score', 'id', 'url', 'num_comments', 'body', 'created', 'Comment', 'Flair'])
print(results)
results.to_csv('reddit_india_test.csv')

# %%
df_india = pd.read_csv('reddit_india_test.csv')
df_india['Comment'].head(20)

# %%


# REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
# BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))
import string 
string.punctuation

# %%

def ret_string(text):
    return str(text)

def remove_punc(text):
    text_nopunc = "".join([char for char in text if char not in string.punctuation])
    return text_nopunc

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]
    return tokens

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    return text

def lemmatizer(text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in text]
    return text
    
# %%
df_india['title'] = df_india['title'].apply(lambda x: ret_string(x))
df_india['title'] = df_india['title'].apply(lambda x: remove_punc(x))
df_india['title'] = df_india['title'].apply(lambda x: tokenize(x))
df_india['title'] = df_india['title'].apply(lambda x: remove_stopwords(x))
df_india['title'] = df_india['title'].apply(lambda x: lemmatizer(x))

df_india['Comment'] = df_india['Comment'].apply(lambda x: ret_string(x))
df_india['Comment'] = df_india['Comment'].apply(lambda x: remove_punc(x))
df_india['Comment'] = df_india['Comment'].apply(lambda x: tokenize(x))
df_india['Comment'] = df_india['Comment'].apply(lambda x: remove_stopwords(x))
df_india['Comment'] = df_india['Comment'].apply(lambda x: lemmatizer(x))

df_india['body'] = df_india['body'].apply(lambda x: ret_string(x))
df_india['body'] = df_india['body'].apply(lambda x: remove_punc(x))
df_india['body'] = df_india['body'].apply(lambda x: tokenize(x))
df_india['body'] = df_india['body'].apply(lambda x: remove_stopwords(x))
df_india['body'] = df_india['body'].apply(lambda x: lemmatizer(x))

df_india['url'] = df_india['url'].apply(lambda x: ret_string(x))
df_india['url'] = df_india['url'].apply(lambda x: remove_punc(x))
df_india['url'] = df_india['url'].apply(lambda x: tokenize(x))
df_india['url'] = df_india['url'].apply(lambda x: remove_stopwords(x))
df_india['url'] = df_india['url'].apply(lambda x: lemmatizer(x))



# %%

df_india['new_feature'] = df_india['title']+df_india['Comment']+df_india['url']
df_india['new_feature'].head(20)


# %%
# def listToStr(s):
#     retStr = ' '.join([str(elem) for elem in s])
#     return retStr

# def clean_text(text):
#     text = "".join([word.lower() for word in text if word not in string.punctuation])
#     tokens = re.split('\W+', text)
#     text = tokens
#     return text

# %%
df_india.to_csv('clean_reddit_india.csv')

# %%



# %%



# %%
