import random

DATA_PATH = 'data/wikisent2.txt'

def get_wiki_sent(shuffle=True, lower=True, rand_seed="4774"):

    with open(DATA_PATH, 'r', encoding="utf8") as f:
        wiki_sent = f.readlines()

    if shuffle:
        random.seed(rand_seed)
        random.shuffle(wiki_sent)

    if lower:
        wiki_sent = [s.lower() for s in wiki_sent]

    return wiki_sent

def get_context_sent(target_word, sent_list, max_search_len=None):

    if max_search_len is not None:
        sent_list = sent_list[0:max_search_len]

    for s in sent_list:
        if s.find(target_word) > 0:
            return s
            
    print("No match!")

    return None