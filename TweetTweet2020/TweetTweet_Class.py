#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 06:16:54 2020

@author: j
"""
import numpy as np
import pandas
import re
import wordsegment as ws
import wordninja as ninja
#----------------------------------------------------------------------------#
class TweetTweet():
    
    def __init__(self, file):
        self.file = file
        self.df = pandas.read_csv(file)
        self.text = self.df['text']
        self.tweet_text_with_hashtags = pandas.Series()
        self.grouped_hashtags_list = []
        self.index_hashtags = pandas.Series()
        self.all_hashtags_singular = pandas.Series()
        self.all_hashtags_grouped = pandas.Series()
        self.singular_hashtags_list = []
        self.grouped_hashtags = pandas.DataFrame()
        self.singular_hashtags = pandas.DataFrame()
        self.hashtags_sep_list = []
        self.hashtags_sep = pandas.DataFrame()
        self.hashtags_num_seg_list = []
        self.hashtags_num_seg = pandas.DataFrame()
        
        self.hashtags_sep_list2 = []
        self.word_ninja_sep = pandas.DataFrame()
#----------------------------------------------------------------------------#        
    ##/ tweets_with_hashtags /##
    # method to find all tweets containing a hashtag
    # returns Series containing tweets with hashtags
    def get_tweets_with_hashtags(self):
            self.tweet_text_with_hashtags = self.text[self.text[:].astype(str).str.contains('#')].str.lower()
            return self.tweet_text_with_hashtags
#----------------------------------------------------------------------------#        
    ##/ return_hashtags /##
    # method to find all words/strings beginning with '#'
    # parameter n = length of Series
    # returns List of all '#hashtags', if posted in same tweet they will be
    # considered together has "one" hashtag
    def return_hashtags_grouped(self, Series):
        t = lambda x: re.findall(r'(?<=#)\w+', Series.iloc[x])
        self.grouped_hashtags_list = []
        for x in range(len(Series)):
            self.grouped_hashtags_list.append((t(x)))
            self.all_hashtags_grouped = pandas.Series(self.grouped_hashtags_list[:])
        return self.all_hashtags_grouped
#----------------------------------------------------------------------------#
    ##/ return_hashtags_singular /##
    # method to take all hashtags that were found previously
    # by the grouped method, and now take multiple hashtags that are found on
    # one line and place them on their own line, in order to consider
    # them in the total frequency count.
    # return list of singular #hashtags
    def return_hashtags_singular(self, Series):
        joined = pandas.Series(self.all_hashtags_grouped.str.join(' '))
        index = joined.str.contains(' ')
        index_len = len(index)
        for i in range(index_len):
            if (index[i] == False):
                self.singular_hashtags_list.append(joined[i])
            else:
                string = joined[i]
                count = string.count(' ')
                for j in range(count+1):
                    self.singular_hashtags_list.append(string.split(' ')[j])
                    index_len = index_len+1
        self.all_hashtags_singular = pandas.Series(self.singular_hashtags_list)
        return self.all_hashtags_singular    
#----------------------------------------------------------------------------#
    ##/ index_hashtags /##
    # method to find index in tweet where hashtag is found
    # returns List containing index of hashtag associated with each tweet
    def get_index_hashtags(self, Series):
        index_hashtags = []
        for i in range(len(Series)):
            index_hashtags.append([m.start() for m in re.finditer('#', Series.iloc[i])])
        self.index_hashtags = pandas.Series(index_hashtags[:])        
        return self.index_hashtags
#----------------------------------------------------------------------------#    
    # #/ remove__grouped_duplicates /##
    # # method to remove duplicate hashtags from List
    # # returns list of singular '#hashtags'
    # def remove_grouped_duplicates(self, Series):
    #     temp = pandas.DataFrame(self.grouped_hashtags_list)
    #     first_col = temp.loc[:,0]
    #     for i in range(1,len(temp.columns)):
    #         first_col.append(temp.loc[:,i].dropna(), ignore_index=True)
    #     self.grouped_hashtags = pandas.DataFrame(first_col.drop_duplicates())
    #     return self.grouped_hashtags
# #----------------------------------------------------------------------------#    
#     ##/ remove_singular_duplicates /##
#     # method to remove duplicate hashtags from List
#     # returns list of singular '#hashtags'
#     def remove_singular_duplicates(self, Series):
#         temp = pandas.DataFrame(self.singular_hashtags_list[:])
#         first_col = temp.loc[:,0]
#         sec_col = temp.loc[:,1]
#         for i in range(1,len(temp.columns)):
#             final_col = first_col.append(temp.loc[:,i],ignore_index=True)
#         self.singular_hashtags = pandas.Series(final_col.drop_duplicates())
#         return self.singular_hashtags
#----------------------------------------------------------------------------#
    ##/ remove_singular_duplicates /##
    # method to remove duplicate hashtags from List
    # returns list of singular '#hashtags'
    # this method makes use of pandas.DataFrame.index
    def remove_singular_duplicates(self):
        freq = self.all_hashtags_singular.value_counts()
        self.singular_hashtags = pandas.Series(freq.index)
        return self.singular_hashtags
#----------------------------------------------------------------------------#
    ##/ remove_grouped_duplicates /##
    # method to remove duplicate hashtags from List
    # returns list of singular '#hashtags'
    # this method makes use of pandas.DataFrame.index
    def remove_grouped_duplicates(self):
        freq = self.all_hashtags_grouped.value_counts()
        self.grouped_hashtags = pandas.Series(freq.index)
        return self.grouped_hashtags
#----------------------------------------------------------------------------#
    ##/ word_seg /##
    ## method to separate words in hashtag
    ## probably gonna return a Series with strings inside
    def word_seg(self, DataFrame):
        ws.load()
        for i in range(len(DataFrame)):
            self.hashtags_sep_list.append(ws.segment(str(DataFrame[i])))
        self.hashtags_sep = pandas.Series(self.hashtags_sep_list)
        return self.hashtags_sep
#----------------------------------------------------------------------------#
    def num_seg(self, DataFrame):
        for i in range(len(DataFrame)):
            self.hashtags_num_seg_list.append(re.findall('\d+|\D+', str(DataFrame[i])))
        self.hashtags_num_seg = pandas.Series(self.hashtags_num_seg_list)
        return self.hashtags_num_seg
#----------------------------------------------------------------------------#
    def word_ninja(self, DataFrame):
        for i in range(len(DataFrame)):
            new_string = ninja.split(str(DataFrame[i]))
            new_string = str(new_string).translate({ord(i): None for i in ['"',"'",',','[',']']})
            self.hashtags_sep_list2.append(new_string)
        self.word_ninja_sep = pandas.Series(self.hashtags_sep_list2)
        return self.word_ninja_sep
#----------------------------------------------------------------------------#
#Series are MUCH faster than DataFrame!

#df.replace(regex=r'^ba.$', value='new')


t1 = TweetTweet("./TrumpTweets08262020toBeginning")

df = pandas.DataFrame()
df['text'] = t1.text
df['index_of_tweet'] = pandas.Series(t1.get_tweets_with_hashtags().index)
df['tweet_text_with_hashtags'] = t1.get_tweets_with_hashtags().reset_index().drop('index',axis=1)
df['index_of_hashtags'] = t1.get_index_hashtags(t1.tweet_text_with_hashtags)
df['all_hashtags_grouped'] = t1.return_hashtags_grouped(t1.tweet_text_with_hashtags)
df['all_hashtags_singular'] = t1.return_hashtags_singular(t1.tweet_text_with_hashtags)
df['hashtags_grouped'] = t1.remove_grouped_duplicates()
df['frequency_grouped'] = pandas.Series(t1.all_hashtags_grouped.value_counts().values)
df['hashtags_singular'] = t1.remove_singular_duplicates()
df['frequency_singular'] = pandas.Series(t1.all_hashtags_singular.value_counts().values)
df['word_seg'] = pandas.Series(t1.word_seg(df['all_hashtags_grouped']))
df['num_seg+word_seg'] = pandas.Series(t1.num_seg(df['word_seg'])).str.join(' ')

df['word_ninja'] = pandas.Series(t1.word_ninja(df['all_hashtags_grouped']))

#df.to_csv('df.csv')
#df.to_excel('df.xlsx')
#df.to_html('df.html')
#df.to_csv('df.txt')









 
