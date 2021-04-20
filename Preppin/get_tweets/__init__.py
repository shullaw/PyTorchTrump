from tweet_parser.tweet import Tweet
from tweet_parser.tweet_parser_errors import NotATweetError
import fileinput
import json
import multiprocessing
import os
import pandas
import uuid  # good for iterative naming
from time import process_time_ns

#most recent method used for parsing tweet text from files
def get_tweets(file):   
    full_text = []
    hashtags = []
    start=process_time_ns()
    print('Start=' , start , 'Reading file: ' + file)
    with open(file, 'r', encoding='utf-8', errors='replace') as reader:  # read in file
        for line in reader:  # for line in file
            try:
                tweet_dict = json.loads(line)  # read line containing all attributes
                full_text.append(tweet_dict['full_text'].replace('@realDonaldTrump ', ''))  # grab full tweet text and remove @handle as it began 99% of tweets
                if (tweet_dict['entities']['hashtags']):  # grab hashtags from entities attribute
                    for i in tweet_dict['entities']['hashtags']:  # tweets have more than one hashtag, read all of them
                        hashtags.append(i['text'])  #hashtag is under the text attribute of hashtags
            except Exception as e:
                print(e)  # most errors are coming from non-english tweets
                continue
    new_name = ''.join([f for f in file if f.isdigit()])  #rename file to their date within file name
    idx = str(uuid.uuid1())[:3]  # add index for split files, originally read files with generator/iterator, finally just split files in terminal
    full_text = pandas.Series(full_text)  # add tweet text to dataframe
    full_text.to_csv('tweets_' + new_name + '_' + idx + '.txt', index=False)  # write to file
    idx = str(uuid.uuid1())[:3]
    hashtags = pandas.Series(hashtags)
    hashtags.to_csv('hashtags_' + new_name + '_' + idx + '.txt', index=False)
    print('File: ' + file + 'complete at: ', (process_time_ns() - start))

if __name__ == '__main__':
    
    path = r"/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/All_Tweets_To_Trump_Processed/"
    files = [path+f for f in os.listdir(path)]
    start = process_time_ns()
    # with multiprocessing.Pool(12) as p:  # I would like to test this again on smaller files (split files)
    #       p.map(get_tweets, files)     
    for f in files:
        get_tweets(f)
    print((process_time_ns() - start))
