from tweet_parser.tweet import Tweet
from tweet_parser.tweet_parser_errors import NotATweetError
import fileinput
import json
import multiprocessing
import os
import pandas
import uuid  # good for iterative naming
from time import process_time_ns


class Tget(object):

    def __init__(self, file, filepath):
        self.file = file
        self.text = []
        self.hashtags = []
        self.tweet_dict = json.loads()
        self.df = pandas.DataFrame()
        self.s = pandas.Series()
        
    def get_tweets(self):   

        # start=process_time_ns()
        # print('Start=' , start , 'Reading file: ' + file)
        with open(self.file, 'r', encoding='utf-8', errors='replace') as reader:  # read in file
            for line in reader:  # for line in file
                try:
                    self.tweet_dict = json.loads(line)  # read line containing all attributes
                    self.text.append(self.tweet_dict['full_text'].replace('@realDonaldTrump ', ''))  # grab full tweet text and remove @handle as it began 99% of tweets
                    if (self.tweet_dict['entities']['hashtags']):  # grab hashtags from entities attribute
                        for i in self.tweet_dict['entities']['hashtags']:  # tweets have more than one hashtag, read all of them
                            self.hashtags.append(i['text'])  #hashtag is under the text attribute of hashtags
                except Exception as e:
                    print(e)  # most errors are coming from non-english tweets
                    continue
                new_name = ''.join([f for f in self.file if f.isdigit()])  #rename file to their date within file name
                idx = str(uuid.uuid1())[:3]  # add index for split files, originally read files with generator/iterator, finally just split files in terminal
                self.text = pandas.Series(self.text)  # add tweet text to dataframe
                self.text.to_csv('tweets_' + new_name + '_' + idx + '.txt', index=False)  # write to file
                idx = str(uuid.uuid1())[:3]
                self.hashtags = pandas.Series(self.hashtags)
                self.hashtags.to_csv('hashtags_' + new_name + '_' + idx + '.txt', index=False)
                print('File: ' + self.file + 'complete at: ', (process_time_ns() - start))

if __name__ == '__main__':
    
    T = Tget()
    path = r"/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/All_Tweets_To_Trump_Processed/"
    files = [path+f for f in os.listdir(path)]
    start = process_time_ns()  
    for f in files:
        #get_tweets(f)
        T.get_tweets(f)
    print((process_time_ns() - start))
