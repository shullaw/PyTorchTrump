#https://github.com/s/preprocessor
import preprocessor

#Andrew Lukyanenko - Kaggle - Preprocessing and other things
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm_notebook
import pickle
import gc
from sklearn.model_selection import KFold

# import os
# import operator
# import random
# from multiprocessing import Pool
# from gensim.models import KeyedVectors
# import re
# from tqdm import tqdm
# from collections import defaultdict
# import json
# import dask.dataframe as ddf
# import platform
# from torch.utils import data
# from keras.preprocessing import text, sequence
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data
# import time
# from sklearn import metrics
# from keras.preprocessing.sequence import pad_sequences
# import cProfile
# tqdm.pandas()
from time import process_time_ns
# import PreprocessingP as PP
# import psutil
# from multiprocessing import Pool
# import multiprocessing as mp
# from gensim.models import KeyedVectors
import string
import pandas
import re
import numpy as np
import torch
import torchtext
from To_Trump_Processing import full_text_to_csv
import time
import Ngram
from From_Trump_Processing import clean_trump_tweets, make_vocab
import From_Trump_Processing


torch.cuda.set_device('cuda:0')

if __name__ == '__main__':
    #start = process_time_ns()
    # set_seed(42)
    # file path to read and write TO trump tweet text
    # at_read_path = r"/media/j/Big Momma/Senior_Project/Datasets/Tweets_to_Trump/All_Tweets_To_Trump"
    # at_write_path = r"/media/j/Big Momma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets"
    # start = process_time_ns()
    # cleaned_text_trump, hashtags_mentions_trump = clean_trump_tweets()
    # print("Process time to clean Trump tweets: ", (start - process_time_ns()))



    # start = process_time_ns()
    # #vocab_trump = make_vocab(cleaned_text_trump)
    # Ngram_model = Ngram.Ngram(cleaned_text_trump)
    # print((start - process_time_ns()))

    # create vocab for trump tweets
    #torched_text_AT = cleaned_text_AT.apply(torchtext.data.utils._basic_english_normalize)
    #torched_text_at = cleaned_text_at.apply(torchtext.vocab.Vocab())
    # file path to read and write TO trump tweet text
 
    # full_text_to_csv(read_path, write_path)

    # read in TO trump full text tweets to create vocab
    # files = [f for f in os.listdir(at_write_path) if f.startswith("split")]
    # preprocess TO trump full text tweets
    # for f in files:
    #     with pandas.read_csv(at_write_path + '/' + f, header=None, delimiter='\t', chunksize=1e9) as reader:
    #         chunk_no = 0
    #         trans_table = str.maketrans(dict.fromkeys(string.punctuation))
    #         for chunk in reader:
    #             text = pandas.Series(chunk[0], name = 'full_text').astype('string')
    #             cleaned_text_TO, hashtags_mentions_TO = complete_clean(text)
    #             hashtags_mentions_TO.to_csv(at_write_path + '/' + f.strip('.txt') + '{}{}{}'.format('hashtags_mentions', chunk_no, '.txt'), 'a', encoding='utf8')
    #             del hashtags_mentions_TO
    #             torched_text_TO = cleaned_text_TO.apply(torchtext.data.utils._basic_english_normalize)
    #             del cleaned_text_TO
    #             torched_text_TO.str.translate(str.maketrans("[],", trans_table)) # remove list punctuation
    #             torched_text_TO.to_csv(at_write_path + '/' + f.strip('.txt') + '{}{}{}'.format('_vocab_', chunk_no, '.txt'), 'a', encoding='utf8')
    #             del torched_text_TO
    #             chunk_no +=1
    
    
    #ngram_model = Ngram.Ngram(cleaned_text_AT)
    print((start - process_time_ns()))

    # tweet_no_hashtag = pandas.Series(text).str.replace(r'(#[a-zA-Z0-9_*]\w+)', '', regex=True)
    # clean_text = tweet_no_hashtag.apply(PP.preprocess)
    # clean_text = clean_text.apply(lambda t: prep.clean(t))
    # clean_text = clean_text.replace(r'^\s*$', np.nan, regex0=True).explode().dropna()

    # load = True
    # if load:
    #     df = pandas.read_csv(r'X:\Senior_Project\Datasets\Trumps Legcy.csv')
    #     #df = df_parallelize_run(df, preprocess(df),6)
    # clean_text = clean_text.apply(p.preprocess)
    # df.to_csv(r'C:\Users\j\PycharmProjects\TensorFlowC\processed_df.txt', index=False)
    # else:
    #     df = pandas.read_csv(r'X:\Senior_Project\Datasets\Trumps Legcy.txt')
    #     # after processing some of the texts are empty
    #     df['comment_text'] = df['comment_text'].fillna('')
