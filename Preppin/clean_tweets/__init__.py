import pandas
import numpy as np
import preprocessor as prep
import re
import string
from time import process_time_ns
import nltk
import uuid
import os
import swifter
import sys

# taken from kaggle, need to cite, I've noted the methods that were added, this was used mainly
# for fixing contractions (you're == you are)
def fixing_with_regex(text) -> str:
    """
    Additional fixing of words.

    :param text: text to clean
    :return: cleaned text
    """

    # mis_connect_list = ['\b(W|w)hat\b', '\b(W|w)hy\b', '(H|h)ow\b', '(W|w)hich\b', '(W|w)here\b', '(W|w)ill\b']
    # mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))

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
   # text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", ' WhatsApp ')

    # Clean repeated letters.
    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
   # text = re.sub(r"(-+|\.+)", " ", text)

    # text = re.sub(r'[\x00-\x1f\x7f-\x9f\xad]', '', text)
    # text = re.sub(r'(\d+)(e)(\d+)', r'\g<1> \g<3>', text)  # is a dup from above cell...
    # #text = re.sub(r"(-+|\.+)\s?", "  ", text)
    # text = re.sub("\s\s+", " ", text)
    # text = re.sub(r'ᴵ+', '', text)

    
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
    
# this method works best reading in files under 700MB only a system with 15.5GB RAM   
def read_to_clean(trump_file):
    
        df = pandas.read_csv(trump_file, names=['full_text'], lineterminator='\n', delimiter='\n', comment='\n',quoting=3,  iterator=True, error_bad_lines=False)  # read in file
        for df in df:
            try:
                temp = complete_clean(df['full_text'])  # some files do not have first row '0' for column name
                new_path = trump_file[0:52] + 'clean_tweets/' + trump_file[63:].strip('.txt') + str(uuid.uuid1())  # send to new folder in same directory      
                temp.to_csv(new_path.strip('.txt') + '_CLEAN.txt', index=False)
            except Exception as e:
                print("sys.exc.info: ", sys.exc_info()[0])
                print("Exception: ", e)
                temp = complete_clean(df)  # catch no column header 
                new_path = trump_file[0:52] + 'clean_tweets/' + trump_file[63:].strip('.txt') + str(uuid.uuid1())  # send to new folder in same directory
                temp.to_csv(new_path.strip('.txt') + '_CLEAN.txt', index=False)      
            except Exception as e:
                print("sys.exc.info: ", sys.exc_info()[0])
                print("Exception: ", e)
                temp = complete_clean(df.read()['full_text'])  # last catch, just testing this out
                new_path = trump_file[0:52] + 'clean_tweets/' + trump_file[63:].strip('.txt') + str(uuid.uuid1())  # send to new folder in same directory      
                temp.to_csv(new_path.strip('.txt') + '_CLEAN.txt', index=False)
            

# swifter used for multiprocessing... hear that fan?
def complete_clean(text):
    try:
        prep.set_options(prep.OPT.EMOJI, prep.OPT.SMILEY, prep.OPT.ESCAPE_CHAR)
        hashtag_count = [text.apply(lambda x: x.count("#")) < 5]  #going to drop these later
        retweets = [~text.str.contains('RT')]  # going to drop retweets later
        text = text.swifter.allow_dask_on_strings().apply(lambda t: fixing_with_regex(t))
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'( #[a-zA-Z0-9_*]\w+)', '', t))  # hashtag removed
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'(@[a-zA-Z0-9_*]\w+)', '', t))  # mention removed
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'https\W+\w+\W+\w+\W+\w+', '', t))  # mention removed
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'(?<!\w)([A-Z])\.', r'\1', t))  # attempt to remove periods from acronyms (C.I.A.)       
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'RT', '', t)).str.lower()  # remove 'RT'

        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'&amp;|&', 'and', t))  # ampersand
        punct = '"$%&\'()+,-/:;=[\\]^_`{|}~'
        punct = str.maketrans(punct, ' '*len(punct)) #map punctuation to space
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.translate(punct))  # replace punctuation with None
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"\.+|\?+|!+", '<EOS>', t))  # replace end of sentence punctuation with EOS  
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"\d{4}",'<year>', t))  # replace 4 digits in a row with <year>        
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"\d",'<number>', t))  # replace digits with <number>        
        text = text.swifter.allow_dask_on_strings().apply(lambda t: t.replace(r'\s+', '')) # extra white space
     
        text = text.swifter.allow_dask_on_strings().apply(lambda t: prep.clean(t))  # remove options ^^
        text = text.swifter.allow_dask_on_strings().apply(lambda t: ' '.join(t.split()))  # remove extra white space
        text = text.swifter.allow_dask_on_strings().apply(lambda t: re.sub(r"(\b\s+<EOS>\b)|(\b<EOS>\s+\b)|(\b\s+<EOS>\s+\b)", r"<EOS>", t)) # remove whitespace before and after EOS
        text = text[hashtag_count[0]]  # drop tweets with more than 5 hashtags... can't make a coherent sentence out of them.
        text = text[retweets[0]]  # drop retweets
        text = text.str.split('<EOS>').explode()  #split sentences into new rows
        text = text[text.values != '']  # drop empty cells
        text = text[text.values != ' '] # drop more empty cells
        text = text.str.strip()  # remove beginning and ending white space
        return text
    except Exception as e:
        print("sys.exc.info: ", sys.exc_info()[0])
        print("Exception: ", e)

        
if __name__ == "__main__":
      
      # read in files from directory one by one, they are processed and new files created in directory /clean_tweets
      path = r'/home/j/anaconda3/envs/PyTorch/PyTorchTrump/Preppin/get_tweets/'
      files = [f for f in os.listdir(r'/home/j/anaconda3/envs/PyTorch/PyTorchTrump/Preppin/get_tweets') if f.startswith('tweets')]
      start_total = process_time_ns()
      print('Start all files: ', start_total/int(1e9), 'seconds\n')
      for f in files[100:]:
          start = process_time_ns()/int(1e9)
          print('Start: ', start, 'file: ' + f)
          read_to_clean(path+f)
          print((process_time_ns() - start))
      print('Total time: ', (process_time_ns()/int(1e9) - start_total), 'seconds\n')
