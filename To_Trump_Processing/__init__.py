#import pandas
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



def full_text_df(file_path):
    full_text = []
    with open(file_path, 'r', encoding='utf8') as fh:
        for line in tqdm(fh):
            idx1 = line.find(r'"full_text":"@realDonaldTrump') + len(r'"full_text":"@realDonaldTrump')-1
            idx2 = line.find(r',"truncated')
            full_text.append(line[idx1:idx2])


# def json_to_pandas(file):
#     id = []
#     full_text = []
#     with open(file, 'r', encoding='utf8') as fh:
#         for line in fh:
#             tweet = json.loads(line)['id']
#             id.append(tweet)
#             tweet = json.loads(line)['full_text']
#             full_text.append(tweet)
#     df = pandas.DataFrame({'id': id, 'full_text': full_text})


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


def read_in_chunks(file_object, chunk_size=int(1000)):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 10000."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data
        

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


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



