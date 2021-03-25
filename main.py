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

import os
import operator
import random
from multiprocessing import Pool
from gensim.models import KeyedVectors
import re
from tqdm import tqdm
from collections import defaultdict
import json
import dask.dataframe as ddf
import platform
from torch.utils import data
from keras.preprocessing import text, sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import time
from sklearn import metrics
from keras.preprocessing.sequence import pad_sequences

tqdm.pandas()
from time import process_time_ns
import PreprocessingP as PP
import preprocessor as prep
import psutil
from multiprocessing import Pool
import multiprocessing as mp
from gensim.models import KeyedVectors
import string
import pandas
import re
import numpy as np
import torch
import torchtext


def set_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def df_parallelize_run(df: pd.DataFrame(), func, npartitions=os.cpu_count()/2):
    if platform.system() == 'Windows':
        dask_dataframe = ddf.from_pandas(df, npartitions=os.cpu_count()/2)
        result = dask_dataframe.map_partitions(func, meta=df)
        df = result.compute()
    elif platform.system() == 'Linux':
        df_split = np.array_split(df, npartitions)
        pool = Pool(npartitions)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()

    else:
        print('No idea what to do with your OS :(')

    return df

def camel_case(clean_text):
    clean_text = clean_text.apply(lambda x: re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x))
    return clean_text

def complete_clean(df):

    prep.set_options(prep.OPT.URL, prep.OPT.EMOJI, prep.OPT.ESCAPE_CHAR,
                     prep.OPT.RESERVED, prep.OPT.SMILEY)
    tweet_no_hashtag = pandas.Series(text).str.replace(r'(#[a-zA-Z0-9_*]\w+)', '', regex=True) #hashtag removed
    clean_text = tweet_no_hashtag.apply(lambda t: prep.clean(t)) #remove options ^^
    clean_text = clean_text.str.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) #replace punctuation with space
    clean_text = clean_text.replace(r'^\s*$', np.nan, regex=True).explode().dropna() #empty cells? drop!
    clean_text = clean_text.replace(r'\s+', ' ', regex=True) #extra white space
    clean_text = clean_text.replace(r'(?!<a(.*)>(.*))(&amp;|&)(?=(.*)<\/a>)', 'and', regex=True) # ampersand
    clean_text = camel_case(clean_text)
    return clean_text


if __name__ == '__main__':
    tqdm.pandas()
    start = process_time_ns()
    set_seed(42)


    df = pandas.read_csv(r'X:\Senior_Project\Datasets\Trumps Legcy.csv')
    df['comment_text'] = df['comment_text'].astype('string')
    df['date'] = pandas.to_datetime(df['date'], infer_datetime_format=True)
    df = df.sort_values(by='date', ascending=True).reset_index().drop(columns='index')
    text = pandas.Series(df.comment_text, name='comment_text')  # tweet text
    # list to store duplicate hashtags with multiple per line
    # hashtags in order of dataframe with multiple per line
    hashtags_multiD = text.apply(lambda t: re.findall(r'(#[a-zA-Z0-9_*]\w+)', t))  # find hashtags
    # keeping track of index for reinsertion to tweet
    hashtags_multiD = pandas.Series(hashtags_multiD, index=text.index).explode().dropna()  # (multitweet per line).split
    # tweet text with hashtag removed
    cleaned_text = complete_clean(text)

    torched_text = cleaned_text.apply(torchtext.data.utils._basic_english_normalize)



    # tweet_no_hashtag = pandas.Series(text).str.replace(r'(#[a-zA-Z0-9_*]\w+)', '', regex=True)
    # clean_text = tweet_no_hashtag.apply(PP.preprocess)
    # clean_text = clean_text.apply(lambda t: prep.clean(t))
    # clean_text = clean_text.replace(r'^\s*$', np.nan, regex=True).explode().dropna()

    # load = True
    # if load:
    #     df = pandas.read_csv(r'X:\Senior_Project\Datasets\Trumps Legcy.csv')
    #     #df = df_parallelize_run(df, preprocess(df),6)
    #clean_text = clean_text.apply(p.preprocess)
    #df.to_csv(r'C:\Users\j\PycharmProjects\TensorFlowC\processed_df.txt', index=False)
    # else:
    #     df = pandas.read_csv(r'X:\Senior_Project\Datasets\Trumps Legcy.txt')
    #     # after processing some of the texts are empty
    #     df['comment_text'] = df['comment_text'].fillna('')
    print(start - process_time_ns())


