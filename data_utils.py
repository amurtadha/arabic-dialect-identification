import random

from torch.utils.data import Dataset
import  json
from tqdm import tqdm
import numpy as np
import re
import pickle
import os
from gensim.models.word2vec import Word2Vec
import gensim

def clean_str(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    text = text.strip()

    return text

def _load_word_vec(path,embed_dim, word2idx=None):
    word_vec = {}
    if '.bin' in path:
        embeddings = dict()
        # embeddings = Word2Vec.load_word2vec_format(path, binary=True, encoding='utf8',unicode_errors='ignore')
        # embeddings=gensim.models.Word2Vec.load(path)
        embeddings= gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, encoding='utf8',unicode_errors='ignore')
        for word in  word2idx.keys():
            try:
                word_vec[word] = embeddings[word]
            except:
                pass

    else:
        model = gensim.models.Word2Vec.load(path)
        for word in  word2idx.keys():
            try:
                word_vec[word] = model.wv[word]
            except:
                pass
    return word_vec


def build_embedding_matrix(opt,word2idx, embed_dim, dat_fname,embedding):
    if os.path.exists('embedding/{}_{}'.format(embedding,dat_fname)):
        print('loading embedding_matrix embedding/{}_{}'.format(embedding,dat_fname))
        embedding_matrix = pickle.load(open('embedding/{}_{}'.format(embedding,dat_fname), 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros

        word_vec = _load_word_vec(opt.glove_path,embed_dim, word2idx=word2idx)
        print('building embedding_matrix:', 'embedding/_'+dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open('embedding/{}_{}'.format(embedding,dat_fname), 'wb'))
    return embedding_matrix

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists('embedding/{}_{}'.format(embedding,dat_fname)):
        print('loading tokenizer: embedding/{}_{}'.format(embedding,dat_fname))
        tokenizer = pickle.load(open('embedding/{}_{}'.format(embedding,dat_fname), 'rb'))
    else:
        text = ''
        for fname in fnames:
            data = json.load(open(fname))
            for d in tqdm(data):
                text_raw, label = d['text'], d['label']

                text_raw=clean_str(text_raw)
                text += text_raw + " "
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open('embedding/{}_{}'.format(embedding,dat_fname), 'wb'))
    return tokenizer
def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        # if self.lower:
        #     text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)


class Process_Corpus_LSTM(Dataset):
    def __init__(self, opt, fname, tokenizer):
        all_data = []
        labels = json.load(open('../datasets/{0}/labels.json'.format(opt.dataset)))
        labels = {label: _ for _, label in enumerate(labels)}
        data = json.load(open(fname))
        for d in tqdm(data):
                text, label = d['text'], d['label']
                if label not in labels:
                    continue
                label =labels[label]
                text=clean_str(text)
                text_raw_indices = tokenizer.text_to_sequence(text)
                data = {
                    'text_raw_indices': text_raw_indices,
                    'label': label,
                }
                all_data.append(data)

        self.data = all_data


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Process_Corpus_BERT(Dataset):
    def __init__(self, fname, tokenizer, max_seq_len, dataset):
        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len

        labels = json.load(open('../datasets/{0}/labels.json'.format(dataset)))
        labels = {label: _ for _,label in enumerate(labels)}

        data = json.load(open(fname))

        all_data=[]
        for d in tqdm(data):
            text, label = d['text'], d['label']

            if label not in labels: continue
                # print(d)

            inputs = tokenizer.encode_plus(text.strip().lower(), None, add_special_tokens=True,
                                           max_length=max_seq_len, truncation='only_first', padding='max_length',
                                           return_token_type_ids=True)

            input_ids = inputs['input_ids']
            input_mask = inputs['attention_mask']
            segment_ids = inputs["token_type_ids"]

            assert len(input_ids) <= max_seq_len
            input_ids = np.asarray(input_ids, dtype='int64')
            input_mask = np.asarray(input_mask, dtype='int64')
            segment_ids = np.asarray(segment_ids, dtype='int64')

            data = {
                'text': text,
                'input_ids': input_ids,
                'segments_ids': segment_ids,
                'input_mask': input_mask,
                'label': labels[label]
            }
            all_data.append(data)
            # if len(all_data)>10000:break
        self.data = all_data


    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)
