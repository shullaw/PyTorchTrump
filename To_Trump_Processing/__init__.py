import pandas
import json
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import os
import PreprocessingP as P
import gc
import tracemalloc
import linecache
import sys
from itertools import islice
import mmap





#tried reading files with jsonloads, did not like my files
# def clean_files(full_path):
#     id = []
#     full_text = []
#     with open(full_path, 'r', encoding='utf8') as fh:
#         for line in fh:
#             tweet = json.loads(line)['id']
#             id.append(tweet)
#             tweet = json.loads(line)['full_text']
#             full_text.append(tweet)
#     df = pandas.DataFrame({'id': id, 'full_text': full_text})
#     df.to_csv(full_path.strip('.txt') + '_clean.txt')

# original parsing method, no longer used
def full_text_df(file_path):
    full_text = []
    with open(file_path, 'r', encoding='utf8') as fh:
        for line in tqdm(fh):
            idx1 = line.find(r'"full_text":"@realDonaldTrump') + len(r'"full_text":"@realDonaldTrump')-1
            idx2 = line.find(r',"truncated')
            full_text.append(line[idx1:idx2])
            

# Attempt to read file by chunks.  This was before splitting files from 200GB+ down to 70-100MB inside of Preppin
def read_in_chunks(file_object, chunk_size=int(1000)):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 10000."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data
        
# Originally used when parsing text manually rather than with tweet-parser module
def full_text_to_csv(read_path, write_path):
    files = [f for f in os.listdir(read_path) if f.endswith(".txt")]
    full_text = []
    for f in files:
        with open(read_path + '/' + f, 'r', encoding='utf8', errors='ignore') as fp:
            for line in tqdm(fp, 'Parsing: '):
                idx1 = line.find(r'"full_text":"@realDonaldTrump') + len(r'"full_text":"@realDonaldTrump')
                idx2 = line.find(r',"truncated')
                full_text.append(line[idx1:idx2])
            with open(write_path + '/' + f.strip('.txt') + '_cleaned.txt', 'a', encoding='utf8') as wp:
                for item in tqdm(full_text, 'Writing: '):
                    wp.write(item + '\n')
            full_text = []




