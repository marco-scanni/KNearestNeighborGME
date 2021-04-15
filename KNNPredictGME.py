# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:19:43 2021

@author: marco
Data taken from https://www.kaggle.com/leukipp/reddit-finance-data
"""
import sklearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

#read in posts from 7 different subreddits
wsb = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\wallstreetbets\wallstreetbets.csv', delimiter=',', encoding="utf-8-sig")
rh = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\robinhood\robinhood.csv', delimiter=',', encoding="utf-8-sig")
investing = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\investing\investing.csv', delimiter=',', encoding="utf-8-sig")
stocks = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\stocks\stocks.csv', delimiter=',', encoding="utf-8-sig")
forex = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\forex\forex.csv', delimiter=',', encoding="utf-8-sig")
stockmarket = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\stockmarket\stockmarket.csv', delimiter=',', encoding="utf-8-sig")
options = pd.read_csv(r'D:\School\Non-class Items\Interning 2021\FinanceRedditML\Data\options\options.csv', delimiter=',', encoding="utf-8-sig")

#create one big list of all subreddit dataframes
subreddits = []
subreddits.append(rh)
subreddits.append(wsb)
subreddits.append(investing)
subreddits.append(stocks)
subreddits.append(forex)
subreddits.append(stockmarket)
subreddits.append(options)

#create on large dataframe, and drop unuseful/blank features
masterSr = pd.concat(subreddits)
masterSr = masterSr.drop(['id','author','retrieved','edited','pinned','archived','locked','is_self','is_video','is_original_content','thumbnail','shortlink'], 1)

#don't take into account any posts that got 0 traction(no upvotes), and deleted posts
masterSr = masterSr[masterSr.score > 1]
masterSr = masterSr[masterSr.removed != 1]
masterSr = masterSr[masterSr.deleted != 1]

#title now keeps track of whether or not the post mentioned gme, 1= mentioned, 0 = not mentioned
masterSr.loc[(~masterSr['title'].str.contains('gme|gamestop', case=False,na=False)) & (~masterSr['selftext'].str.contains('gme|gamestop', case=False,na=False)), 'title'] = 0
masterSr.loc[(masterSr['title'].str.contains('gme|gamestop', case=False,na=False)) | (masterSr['selftext'].str.contains('gme|gamestop', case=False,na=False)), 'title'] = 1


#take flair, number of awards, mentioning of GME, and like ratio and convert into useable, numerical lists
flair = le.fit_transform(list(masterSr['link_flair_text']))
awards = le.fit_transform(list(masterSr['total_awards_received']))
title = le.fit_transform(list(masterSr['title']))
likeRatio = le.fit_transform(list(masterSr['upvote_ratio']))

predict = "awarded"
X = list(zip(flair, likeRatio, awards))
y = list(title)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.25)

model = KNeighborsClassifier(n_neighbors = 135)

model.fit(x_train,y_train)
acc = model.score(x_test, y_test)
predicted = model.predict(x_test)


print("\nPredicted whether or not finance-related reddit posts mention GME with " + str(round(acc*100,2)) + "% accuracy.")