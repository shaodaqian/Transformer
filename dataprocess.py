import torchtext as tt
from sacremoses import MosesTokenizer
from subword_nmt import learn_bpe, apply_bpe
import sentencepiece
import re
import os
import dill as pickle
from collections import Counter


DATA_FOLDER = './data/processed'

def tokenize(lang, corpus):
    if lang == 'en' or lang == 'de':
        mosestok = MosesTokenizer(lang=lang)
        tokenized = [mosestok.tokenize(line) for line in corpus]
        return tokenized
    else:
        print(f'Unrecognised language: {lang}')
        raise ValueError()


# Cleaning the corpus
def clean_corpus(en, de):
    en, de = remove_empty_lines(en, de)
    en, de = remove_redundant_spaces(en, de)
    en, de = drop_lines(en, de)
    return en, de

# removes empty lines
def remove_empty_lines(en, de):
    cleaned_en = []
    cleaned_de = []
    for l1, l2 in zip(en, de):
        if not (l1 == '' or l2 == '' or l1 == '\n' or l2 == '\n'):
            cleaned_en.append(l1)
            cleaned_de.append(l2)
    return cleaned_en, cleaned_de


# removes redundant space characters
def remove_redundant_spaces(en, de):
    regex = r' +' # 1 or more spaces
    en = [re.sub(regex, ' ', l1) for l1 in en]
    de = [re.sub(regex, ' ', l2) for l2 in de]
    return en, de


# drops lines (and their corresponding lines), that are empty, too short, too long or violate the 9-1 sentence ratio limit of GIZA++
def drop_lines(en, de, min_length=1, max_length=80):
    en_dropped = []
    de_dropped = []
    for l1, l2 in zip(en, de):
        len1 = len(l1.split(' '))
        len2 = len(l2.split(' '))
        if len1 < min_length or len2 < min_length:
            continue
        if len1 > max_length or len2 > max_length:
            continue
        # 9-1 sentence ratio
        if len1 > 9*len2 or len2 > 9*len1:
            continue
        en_dropped.append(l1)
        de_dropped.append(l2)
    return en_dropped, de_dropped


# bpe
def subword(en, de, merge_ops=32000):
    concatenated = en + de
    print('Learning BPE encoding. This may take a while.')
    outfile_name = os.path.join(DATA_FOLDER, f'bpe.{merge_ops}')
    learn_bpe.learn_bpe(concatenated, outfile_name, num_symbols=merge_ops)
    # Currently unfinished
    # bpe = apply_bpe.BPE(args.codes, , args.separator, None)
    return None



def load_data(name, src_field, trg_field):
    fields = (src_field, trg_field)
    path = os.path.join(DATA_FOLDER, name)
    train_data = tt.datasets.TranslationDataset(
        path=path,
        exts=('.en', '.de'),
        fields=fields
    )
    return train_data



def load_vocab(ext):
    filename = os.path.join(DATA_FOLDER, f'vocab.50k.{ext}')
    vocab = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            token, number = line.split()
            number = int(number)
            vocab[token] = number
    return Counter(vocab)


def load_data_dict():
    data = {}
    data['max_len'] = 80
    en_vocab = load_vocab('en')
    de_vocab = load_vocab('de')
    src_field = tt.data.Field(
        tokenize=str.split,
        lower=True,
    )
    trg_field = tt.data.Field(
        tokenize=str.split,
        lower=True,
    )
    src_field.build_vocab(en_vocab)
    trg_field.build_vocab(de_vocab)
    data['fields'] = {
        'src': src_field,
        'trg': trg_field,
    }
    data['train'] = load_data('TEST', src_field, trg_field)
    data['valid'] = load_data('TEST', src_field, trg_field)
    # data['valid'] = load_data('newstest2014.tok.clean.bpe.32000', src_field, trg_field)
    return data

# en-fr: 32000 word-piece vocab: Google’s neural machine translation system: Bridging the gap between human and machine translation (2016)
# https://arxiv.org/pdf/1609.08144.pdf

# Use sentencepiece


