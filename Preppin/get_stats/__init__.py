import pandas
import os
from time import process_time_ns


# get hashtags, drop duplicates
def read_ht_data(directory):
    try:
        files = [f for f in os.listdir(directory) if f.startswith('hashtags')]
    except Exception as e:
        print('Exception: ' + e)
    hashtags = []
    for f in files:
        with open(directory+f, 'r') as reader:
            lines = [line for line in reader.readlines()]
            lines = [line.strip('\n') for line in lines]
            for l in lines: hashtags.append(l)
    o_length = len(hashtags)
    hashtags = [h for h in hashtags if h.isascii()]  # looking to get rid of non-english/non-twitter standard hashtags(mainly containing underscores)
    good_count = len(hashtags)
    hashtags = pandas.DataFrame(hashtags)
    hashtags = hashtags[hashtags.values != '0']  # header was 0 in files
    stats = hashtags.describe(include='all')
  #  duplicates = Counter(hashtags[~hashtags.drop_duplicates()])
    hashtags = hashtags.drop_duplicates()
    print('Hashtags\n--------\nStarted with: ', o_length)
    print('Duplicates: ', stats[0]['count'] - stats[0]['unique'])
    print('Unique: ', stats[0]['unique'])
    print('Bad tags: ', o_length - good_count)
    print('Most Frequent: ', stats[0]['freq'], stats[0]['top'])
    print('Final count: ', hashtags.describe(include='all')[0]['count'])

    return hashtags#, duplicates

# # This is to get information on the docs, as well as drop duplicates
# # Sentences with more than 3 words, ascii chars only
# def create_docs(directory):  # docs = sentences
#     try:
#         files = [f for f in os.listdir(directory) if f.startswith('clean')]
#     except Exception as e:
#         print('Exception: ' + e)
#     docs = []
#     for f in files:
#         with open(directory+f, 'r') as reader:
#             for line in reader:
#                 split = line.split()
#                 if line.isascii():
#                     if len(split) > 2:
#                        # if len(split) < 20:
#                             docs.append(line.strip('\n'))
#                             if (len(docs) > 10000000):
#                                 print('Writing 10M lines')
#                                 with open('docs3.txt', 'a') as writer:
#                                     for doc in docs:
#                                         writer.write(doc + '\n')
#                                 docs = []
#         with open('docs3.txt', 'a') as writer:  # append remaining docs
#             for doc in docs:
#                 writer.write(doc + '\n')
                            
                                
#         # docs = pandas.DataFrame(docs)
#         # with open('docs3.txt', 'a') as writer:
#         #     for doc in docs:
#         #         writer.write(doc)
#         # o_size = docs.size
#         # stats = docs.describe()
#         # docs = docs.drop_duplicates()
#         # print('Docs\n--------\nStarted with: ', o_size)
#         # print('Duplicates: ', stats[0]['count'] - stats[0]['unique'])
#         # print('Unique: ', stats[0]['unique'])
#         # print('Most Frequent: ', stats[0]['freq'], stats[0]['top'])
#         # print('Final count: ', docs.describe(include='all')[0]['count'])

#     # return docs


# def find_normal_tweet_info(directory):
#     try:
#         files = [f for f in os.listdir(directory) if f.startswith('clean')]
#     except Exception as e:
#         print('Exception: ' + e)
#     words = 0
#     sentences = 0
#     maxlen = 0
#     for f in files:
#         with open(directory+f, 'r') as reader:
#             for line in reader:
#                 if len(line.split()) > 1:
#                     sentences +=1
#                     if maxlen < len(line.split()):
#                         maxlen = len(line.split())
#                     words += len(line.split())
#     print('Words: ', words, '\nSentences: ', sentences, 
#           '\nMean: ', words/sentences, '\nMax: ', maxlen)
    
# # splits docs into words (tokens) and creates counter object    

    

if __name__ == '__main__':
    
        start_total = process_time_ns()/int(1e9)
        print('Start: ', start_total, 'seconds\n')
    
        # hashtag_dir = r'/home/j/anaconda3/envs/PyTorch/PyTorchTrump/Preppin/get_tweets/'
        # hashtags = read_ht_data(hashtag_dir)
        # #hashtags.to_csv('#HASHTAGS.txt')

        # tweet_path = r'/home/j/anaconda3/envs/PyTorch/PyTorchTrump/Preppin/clean_tweets/cleaned_tweet_files/'
        # find_normal_tweet_info(tweet_path)
        # docs = create_docs(tweet_path)
        # docs.to_csv('doc3.txt', index=False)
        
        # doc_path = r'/home/j/anaconda3/envs/PyTorch/PyTorchTrump/Preppin/get_stats/'
        # tokens = tokenize(doc_path)
        # tokens = less_than_n(tokens,100)
        # vocab = get_vocab(tokens)

        print('Total time: ', (process_time_ns()/int(1e9) - start_total), 'seconds\n')
