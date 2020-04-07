import praw
import pandas as pd
import praw

reddit = praw.Reddit(client_id='7hx8xkseiQkWNA', client_secret='UPd0tXn-Lt26mk4-Z38oCNLuG_8', user_agent='Kevin Stephen')

posts = []
india = reddit.subreddit('india')

for post in india.hot(limit=10):
    posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

posts.to_csv('india.csv')