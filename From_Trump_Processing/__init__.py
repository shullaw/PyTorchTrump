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
from num2words import num2words





# found this on Kaggle, need to cite
def fixing_with_regex(text) -> str:
    """
    Additional fixing of words.

    :param text: text to clean
    :return: cleaned text
    """

    mis_connect_list = ['\b(W|w)hat\b', '\b(W|w)hy\b', '(H|h)ow\b', '(W|w)hich\b', '(W|w)here\b', '(W|w)ill\b']
    mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

    text = re.sub(r"Mr. ", "Mister ", text)  # added
    text = re.sub(r"Ms. ", "Misses ", text)  # added
    text = re.sub(r"Mrs. ", "Misses ", text)  # added

    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", ' WhatsApp ')

    # Clean repeated letters.
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
   # text = re.sub(r"(-+|\.+)", " ", text)

    text = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', text)
    text = re.sub(r'(\d+)(e)(\d+)', r'\g<1> \g<3>', text)  # is a dup from above cell...
    #text = re.sub(r"(-+|\.+)\s?", "  ", text)
    text = re.sub("\s\s+", " ", text)
    text = re.sub(r'ᴵ+', '', text)

    
    text = re.sub(r"(H|h)asn(\'|\’)t ", "has not ", text)  # added
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)

    text = re.sub(
        r'(by|been|and|are|for|it|TV|already|justhow|some|had|is|will|would|should|shall|must|can|his|here|there|them|these|their|has|have|the|be|that|not|was|he|just|they|who)(how)',
        '\g<1> \g<2>', text)

    return text

#originally built tweet cleaning methods inside of this module/method
def complete_clean(text):
    try:
        prep.set_options(prep.OPT.EMOJI, prep.OPT.SMILEY)
        hashtag_count = [text.apply(lambda x: x.count("#")) < 5]  #going to drop these later
        retweets = [~text.str.contains('RT')]  # going to drop retweets later
        text = text.swifter.allow_dask_on_strings().apply(fixing_with_regex)
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'( #[a-zA-Z0-9_*]\w+)', '', t))  # hashtag removed
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'(@[a-zA-Z0-9_*]\w+)', '', t))  # mention removed and lower cased
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'https\W+\w+\W+\w+\W+\w+', '', t))  # mention removed and lower cased
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'(?<!\w)([A-Z])\.', r'\1', t))       
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'RT', '', t)).str.lower()  # remove 'RT'

        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'&amp;|&', 'and', t))  # ampersand
        punct = '"$%&\'()+,-/:;=[\\]^_`{|}~'
        punct = str.maketrans(punct, ' '*len(punct)) #map punctuation to space
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.translate(punct))  # replace punctuation with None
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"\.+|\?+|!+", '<EOS>', t))  # remove 'RT'  
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"\d{4}",'<year>', t))  # remove 'RT'        
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"\d",'<number>', t))  # remove 'RT'        
        #text = text.swifter.apply(lambda t: t.replace(r'^\s*$', str(np.nan))).explode().dropna()  # empty cells? drop!
        text = text[text.values != ''].str.strip()  # drop empty cells
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.replace(r'\s+', '')) # extra white space
        #trans = str.maketrans({'[':None,']':None,"'":None,",":None})
        #text = text.swifter.allow_dask_on_strings().apply(lambda t: t.translate(trans))
        text = text.swifter.allow_dask_on_strings().apply(lambda t: prep.clean(t))  # remove options ^^
        text = text.swifter.allow_dask_on_strings().apply(lambda t: ' '.join(t.split()))
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"(\b\s+<EOS>\b)|(\b<EOS>\s+\b)|(\b\s+<EOS>\s+\b)", r"<EOS>", t)) # mention removed and lower cased
        text = text[hashtag_count[0]]  # drop tweets with more than 5 hashtags... can't make a coherent sentence out of them.
        text = text[retweets[0]]
        text = text.str.split('<EOS>').explode()
        text = text[text.values != '']  # drop empty cells
        text = text[text.values != ' '] # drop more empty
        return text.str.strip()
    except Exception as e:
        print("sys.exc.info: ", sys.exc_info()[0])
        print("Exception: ", e)

#if from trump, returns 2 Series, if to trump, turns into files
def read_to_clean(trump_file):
    ## CLEAN TRUMP TWEETS ##
     #try:
         df = pandas.read_csv(trump_file)  # read in trump tweet data
         df['full_text'] = df['full_text'].astype('string')  # trump tweet text
         df['date'] = pandas.to_datetime(df['date'], infer_datetime_format=True)  # dates for sorting
         df = df.sort_values(by='date', ascending=True).reset_index().drop(columns='index')  # sort by date
         df = pandas.Series(df.full_text, name='full_text')  # tweet text
         df, hashtags_mentions = complete_clean(df)
         return df, hashtags_mentions
    #except:
        #start = process_time_ns()
        #files = [trump_file for trump_file in os.listdir(path)]
        #for trump_file in files:
       # for df in pandas.read_csv(trump_file, names=['full_text'], delimiter='\n', comment='\n', iterator=True, error_bad_lines=False):  # 64m lines, 100m did not work
            #df = df.read()  # trump tweet text
            #df = pandas.Series(df.full_text, name='full_text')  # tweet text
            #df = df['full_text'].astype('str')
            #df = complete_clean(df.read(100000))
            #df = parallelize_dataframe(df, complete_clean)
            #df = df.apply(complete_clean)
            #hashtags_mentions = parallelize_dataframe(df, hasht_ment)
           # hashtags_mentions = hasht_ment(df)
            #idx = str(uuid.uuid1())
            #df.to_csv(trump_file.strip('.txt') + '_clean_noHT_' + idx + '.txt', index=False)
            #hashtags_mentions.to_csv(trump_file.strip('.txt') + '_ht_mentions_' + idx + '.txt', index=False)
            #print(trump_file + '-->' , (process_time_ns() - start))

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
    
    
# if __name__ == "__main__":
    
    trump_file = r'/media/j/BigMomma/Senior_Project/Datasets/Trumps_Legcy/Trumps_Legcy.csv'
    #to_trump_file = r'combined_realdonaldtrump_20180217_ids_hydrated_cleaned.txt'
    #path = r'/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets/'
    #hashtag_files = [f for f in os.listdir(path)]
    
    start = process_time_ns()
    read_to_clean(to_trump_file)
    # # #parallelize_dataframe(df, read_to_clean)

    # # path = r'/media/j/BigMomma/Senior_Project/Datasets/Tweets_to_Trump/Cleaned_Tweets/'
    # # ht_folder = r'hashtags/'
    # # hashtag_files = [f for f in os.listdir(path) if not(f.startswith('hashtag'))]
    # tweet_text = read_to_clean(r'./get_tweets/tweets_202100_93c.txt')
    

    print((start - process_time_ns()))

