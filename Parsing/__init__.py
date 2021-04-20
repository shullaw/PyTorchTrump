import pandas
import re
import string
from time import process_time_ns
import os
import swifter
import sys

# Although "Multi-task Pairwise Neural Ranking for Hashtag Segmentation"
# used simple segmentation, I don't think I'm going to continue
def simple_parse(directory):
        try:
            files = [f for f in os.listdir(directory) if f.startswith('#')]
        except Exception as e:
            print('Exception: ' + e)
        for f in files:
            hashtags = pandas.DataFrame(f)
            hashtags = hashtags[0].swifter.allow_dask_on_strings().apply(lambda t: re.sub(r'(?<=\d)(?=\D)|(?=\d)(?<=\D)', ' ', t))
camel = hashtags[0].apply(lambda t: re.sub(r'(^[a-z]|[A-Z])([a-z]*)', r'\1\2 ', t))

