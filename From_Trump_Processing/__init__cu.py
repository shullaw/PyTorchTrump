import pandas
import numpy as np
import preprocessor as prep
import re
import string
from time import process_time_ns
#import torch
#import torchtext
import nltk
import uuid
import os
import cudf
import dask.dataframe as dd
from dask_cuda import LocalCUDACluster
from dask.utils import parse_bytes
import dask_cudf
import dask
dask.config.set(scheduler='multiprocessing',)
# cluster = LocalCUDACluster(
#     CUDA_VISIBLE_DEVICES="0",
#     rmm_pool_size=parse_bytes("1GB"), # This GPU has 6GB of memory
#     device_memory_limit=parse_bytes("1GB"),
# )
# client = Client(cluster)
# client

trump_file = r'/media/j/BigMomma/Senior_Project/Datasets/Trumps_Legcy/Trumps_Legcy.csv'
to_trump_file = r'combined_realdonaldtrump_20180217_ids_hydrated_cleaned.txt'
path = r'/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets/'


# def DataSet(corpus_file, datafields, label_column, doc_start):
#     with open(corpus_file, encoding='utf-8') as f:
#     examples = []
#     for line in f:
#         columns = line.strip().split(maxsplit=doc_start)
#         doc = columns[-1]
#         label = columns[label_column]
#         examples.append(torchtext.data.Example.fromlist([doc, label], datafields))
#    return torchtext.data.Dataset(examples, datafields)


#if from trump, returns 2 Series, if to trump, turns into files
def clean_trump_tweets(trump_file):
    ## CLEAN TRUMP TWEETS ##
    # try:
    #     df = pandas.read_csv(trump_file)  # read in trump tweet data
    #     df['full_text'] = df['full_text'].astype('string')  # trump tweet text
    #     df['date'] = pandas.to_datetime(df['date'], infer_datetime_format=True)  # dates for sorting
    #     df = df.sort_values(by='date', ascending=True).reset_index().drop(columns='index')  # sort by date
    #     df = pandas.Series(df.full_text, name='full_text')  # tweet text
    #     df, hashtags_mentions = complete_clean(df)
    #     return df, hashtags_mentions
    # except:
        start = process_time_ns()
        #files = [f for f in os.listdir(path) if not f.startswith("Cleanest")]
        #for trump_file in files:
            #for df in dd.read_csv(path+trump_file, header=None, error_bad_lines=False):  # 64m lines
        df = dd.read_csv(trump_file, names=['full_text'], delimiter='\n', error_bad_lines=False,dtype={"full_text": "object"}).repartition(npartitions=70)
        # df['full_text'] = df['full_text'].astype('string')  # trump tweet text
        # df = dd.DataFrame(df)  # tweet text
        df, hashtags_mentions = complete_clean(df)
        idx = str(uuid.uuid1())
        df.to_csv(path+trump_file.strip('.txt') + '_clean_noHT_' + idx + '.txt', index=False)
        hashtags_mentions.to_csv(trump_file.strip('.txt') + '_ht_mentions_' + idx + '.txt', index=False)
        print(idx + '-->' , (start - process_time_ns()))

            
def complete_clean(text):
    prep.set_options(prep.OPT.URL, prep.OPT.EMOJI, prep.OPT.ESCAPE_CHAR, prep.OPT.SMILEY)
    hashtags_mentions = text.apply(lambda t: re.findall(r'(#[a-zA-Z0-9_*]\w+)|(@[a-zA-Z0-9_*]\w+)', str(t)), axis=1, meta=('full_text','object')).compute().explode().dropna()  # find @mentions
    hashtags_mentions = hashtags_mentions.str.translate(str.maketrans({'#' : ' ', '@' : ' '}))  # reserved words (#@) removed
    text = dd.Series(text,name='full_text', meta=('full_text','object'),divisions=(1,2))
    text = text.str.replace(r'(#[a-zA-Z0-9_*]\w+)', '')  # hashtag removed
    text = text.str.replace(r'(@[a-zA-Z0-9_*]\w+)', '').str.lower()  # mention removed and lower cased
    text = text.apply(lambda t: prep.clean(t)).compute()  # remove options ^^
    text = text.apply(lambda t: prep.clean(t)).compute()  # remove options ^^
    nums = str.maketrans(string.digits, ' '*len(string.digits))
    text = text.str.translate(nums)  # remove numbers
    punct = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
    text = text.str.translate(punct)  # replace punctuation with None
    text = text.replace(r'(?!<a(.*)>(.*))(&amp;|&)(?=(.*)<\/a>)', 'and')  # ampersand
    #text = text.apply(nltk.wordpunct_tokenize)
    text = text.replace(r'^\s*$', str(np.nan))  # empty cells? drop!
    text = text.replace(r'\s+', ' ')  # extra white space
    text = text.str.translate(str.maketrans({'[':None,']':None,"'":None,",":None}))
    return text, hashtags_mentions


# def make_vocab(cleaned_text_trump):
#     splits = cleaned_text_trump.apply(torchtext.data.utils._basic_english_normalize)
#     vocab = []
#     for s in splits:
#         for v in s:
#             if (len(v) > 1):
#                 vocab.append(v)
#     vocab = pandas.Series(vocab)
#     vocab_freq = vocab.value_counts()
#     vocab = vocab_freq[vocab_freq > 1]
#     vocab = vocab.drop('rt')  # retweet, insignificant
#     return vocab

def clean_text_csv(trump_file):
    text, hashtags_mentions = clean_trump_tweets(trump_file)
    text.to_csv(trump_file.strip('.csv') + '_clean.txt', index=False)
    return text
    
def tokenize(text):
    words = []
    for sentence in text.str.split():
        for word in sentence:
            words.append(word)
    return words
def Ngram(vocab):
    #words = tokenize(text)
    words = vocab.index
    trigrams = [([words[i], words[i + 1]], words[i + 2]) 
                        for i in range(len(words) - 2)]
    return trigrams 
    
if __name__ == '__main__':
                start = process_time_ns()
                clean_trump_tweets(path+to_trump_file)
                print((start - process_time_ns()))


