import tensorflow as tf
import numpy as np
import sys
import os
import logging
import collections
import json

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format = "%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(level=logging.INFO)

def _read_words(filename,delta = 100000):
    delta = delta
    with open(filename,'r',encoding="utf-8") as f:
        file_arr = f.readlines()
        num_line = len(file_arr)
        read_batch_len = num_line//delta
        s_offset = 0
        e_offset = delta
        for epoch in range(read_batch_len):
            lines = ' '.join(file_arr[s_offset:e_offset]).replace("\n", " <eos> ").split()
            s_offset += delta
            e_offset += delta
            yield lines
            logger.info(str((epoch + 1)*delta) + " lines data read")

def _build_vocab(filename, cut_vocab_size):

    tmp_vocab = np.array([])
    for data in _read_words(filename):
        tmp_vocab = np.hstack((tmp_vocab,data))
    counter = collections.Counter(tmp_vocab)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[0:cut_vocab_size]

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    logger.info("start convert file to ids")
    data = []
    for lines in _read_words(filename):
        data.extend(lines)
    #return [word_to_id[word] for word in data ]
    ids= []
    for word in data:
        if word in word_to_id:
            ids.append(word_to_id[word])
        else:
            ids.append(word_to_id["<unk>"])
    return ids

if __name__ == '__main__':
    # a = _build_vocab('./corpus/corpus.train.data', -1)
    # print(len(a))

    with open('./corpus/train_ids.object','r') as f:
        train_ids = json.load(f)
        print(len(train_ids))
