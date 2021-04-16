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
from multiprocessing import Pool
# import pandarallel.pandarallel as pandarallel
# pandarallel.initialize(progress_bar=True, verbose=True)
import swifter
import sys




#df = pandas.read_csv(path+trump_file, names=['full_text'], delimiter='\n', comment='\n', iterator=True, error_bad_lines=False, chunksize = 1280000)

def complete_clean(text):
    try:
        prep.set_options(prep.OPT.URL, prep.OPT.EMOJI, prep.OPT.ESCAPE_CHAR, prep.OPT.SMILEY)
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'( #[a-zA-Z0-9_*]\w+)', '', t))  # hashtag removed
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'( @[a-zA-Z0-9_*]\w+)', '', t)).str.lower()  # mention removed and lower cased
        text = text.swifter.allow_dask_on_strings().apply(lambda t: prep.clean(t))  # remove options ^^
        nums = str.maketrans(string.digits, ' '*len(string.digits))
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.translate(nums))  # remove numbers
        punct = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.translate(punct))  # replace punctuation with None
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.replace(r'(?!<a(.*)>(.*))(&amp;|&)(?=(.*)<\/a>)', 'and'))  # ampersand
        #text = text.apply(nltk.wordpunct_tokenize)
        #text = text.swifter.apply(lambda t: t.replace(r'^\s*$', str(np.nan))).explode().dropna()  # empty cells? drop!
        text = text[text.values != '']  # drop empty cells
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.replace(r'\s+', ' '))  # extra white space
        trans = str.maketrans({'[':None,']':None,"'":None,",":None})
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.translate(trans))
        return text
    except :
        print("Error: ", sys.exc_info()[0])
        
        
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


def hm_to_csv(hm, path, to_path, file):
    idx = str(uuid.uuid1())[:7]
    hm.to_csv(path + to_path + file.strip('.txt') + '_ht_mentions_' + idx + '.txt', index=False)
    



        

#if from trump, returns 2 Series, if to trump, turns into files
def read_to_clean(trump_file):
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
        #files = [trump_file for trump_file in os.listdir(path)]
        #for trump_file in files:
        for df in pandas.read_csv(trump_file, names=['full_text'], delimiter='\n', comment='\n', iterator=True, error_bad_lines=False):  # 64m lines, 100m did not work
            #df = df.read()  # trump tweet text
            #df = pandas.Series(df.full_text, name='full_text')  # tweet text
            #df = df['full_text'].astype('str')
            #df = complete_clean(df)
            #df = parallelize_dataframe(df, complete_clean)
            #df = df.apply(complete_clean)
            #hashtags_mentions = parallelize_dataframe(df, hasht_ment)
           # hashtags_mentions = hasht_ment(df)
            idx = str(uuid.uuid1())
            df.to_csv(trump_file.strip('.txt') + '_clean_noHT_' + idx + '.txt', index=False)
            #hashtags_mentions.to_csv(trump_file.strip('.txt') + '_ht_mentions_' + idx + '.txt', index=False)
            print(idx + '-->' , (start - process_time_ns()))

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
    text, hashtags_mentions = read_to_clean(trump_file)
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
    

def parallelize_dataframe(df, func, n_cores=12):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pandas.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    
    # #trump_file = r'/media/j/BigMomma/Senior_Project/Datasets/Trumps_Legcy/Trumps_Legcy.csv'
    # to_trump_file = r'combined_realdonaldtrump_20180217_ids_hydrated_cleaned.txt'
    # path = r'/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets/'
    # hashtag_files = [f for f in os.listdir(path)]
    
    start = process_time_ns()
    # read_to_clean(path+to_trump_file)
    # #parallelize_dataframe(df, read_to_clean)
    #print((start - process_time_ns()))
    path = r'/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets/'
    ht_folder = r'hashtags/'
    hashtag_files = [f for f in os.listdir(path) if not(f.startswith('hashtag'))]
    get_hm(path, ht_folder, hashtag_files, True)
    print((start - process_time_ns()))

