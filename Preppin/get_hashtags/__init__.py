import pandas
import numpy as np
import preprocessor as prep
import re
import string
from time import process_time_ns

import nltk
import uuid
import os
from multiprocessing import Pool

import swifter
import sys

# original method used to get hashtags from parsed text (not parsed with tweet-parser module)
# used swifter and multiprocessing but neither were working due to memory constrains and other errors (file's too big, etc.)
def get_hm(path, to_path, files, write=False):            
    for trump_file in files:
        df = pandas.read_csv(path + trump_file, names=['full_text'], delimiter='\n', comment='\n', iterator=True, error_bad_lines=False, chunksize = 1e6)  # 1e6 = 19s @ 2.2GB, 1e7 = 105s
        while (True):
            try:
                text = df.get_chunk()
                hashtags = text['full_text'].swifter.allow_dask_on_strings().apply(lambda t: re.findall(r'( #[a-zA-Z0-9_*]\w+)', str(t))).explode().dropna()  
                mentions = text['full_text'].swifter.allow_dask_on_strings().apply(lambda t: re.findall(r'( @[a-zA-Z0-9_*]\w+)', str(t))).explode().dropna()  
                hm = pandas.concat([hashtags,mentions])
                #hm.reset_index(drop=True)  # dask says it will run faster with these settings
                hm = hm.swifter.allow_dask_on_strings().apply(lambda t: t.translate(str.maketrans({'#' : ' ', '@' : ' '})))  # reserved words (#@) removed
                if (write):
                    hm_to_csv(hm, path, to_path, trump_file)
            except StopIteration:
                stop = True
                break
            except Exception as e:
                print(e)
        if (stop):
            hm_to_csv(hm, path, to_path, trump_file)

# original method used to write parsed tweets to csv
def hm_to_csv(hm, path, to_path, file):
    idx = str(uuid.uuid1())[:7]
    hm.to_csv(path + to_path + file.strip('.txt') + '_ht_mentions_' + idx + '.txt', index=False)
     
    
if __name__ == "__main__":
    
    path = r'/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets/'
    ht_folder = r'hashtags/'
    
    start = process_time_ns()

    hashtag_files = [f for f in os.listdir(path) if not(f.startswith('hashtag'))]
    get_hm(path, ht_folder, hashtag_files, True)
    
    
    print((start - process_time_ns()))
    
