
from torchtext.data import Field, BucketIterator, Example, Dataset
from sacremoses import MosesTokenizer
from subword_nmt import learn_bpe, apply_bpe
import sentencepiece
import re
import os
import dill as pickle
from collections import Counter
from torch.utils.data import DataLoader
import io

from special_tokens import PAD_WORD, UNK_WORD, EOS_WORD, BOS_WORD
from data_download import get_data_urls_and_filenames


PROCESSED_FOLDER = './data/processed'
UNPROCESSED_FOLDER = './data/unprocessed'


def tokenize(lang, corpus):
    if lang == 'en' or lang == 'de':
        mosestok = MosesTokenizer(lang=lang)
        return (mosestok.tokenize(line) for line in corpus)
    else:
        print(f'Unrecognised language: {lang}')
        raise ValueError()


def clean_corpus(en, de):
    zipped = zip(en, de)
    non_empty = remove_empty_lines(zipped)
    non_redundant = remove_redundant_spaces(non_empty)
    normalized_linelengths = drop_lines(non_redundant)
    return normalized_linelengths


def remove_empty_lines(zipped):
    for l1, l2 in zipped:
        if not (l1 == '' or l2 == '' or l1 == '\n' or l2 == '\n'):
            yield l1, l2


def remove_redundant_spaces(zipped):
    regex = r' +' # 1 or more spaces
    for l1, l2 in zipped:
        l1 = [re.sub(regex, ' ', word) for word in l1]
        l2 = [re.sub(regex, ' ', word) for word in l2]
        yield l1, l2


# drops lines (and their corresponding lines), that are empty, too short, too long or violate the 9-1 sentence ratio limit of GIZA++
def drop_lines(zipped, min_length=1, max_length=80):
    for l1, l2 in zipped:
        len1 = len(l1)
        len2 = len(l2)
        if len1 < min_length or len2 < min_length:
            continue
        if len1 > max_length or len2 > max_length:
            continue
        # 9-1 sentence ratio
        if len1 > 9*len2 or len2 > 9*len1:
            continue
        yield l1, l2


# Byte-pair encoding (aka subword)
def subword(zipped, langs, args, merge_ops=32000):
    print('Learning BPE encoding. This may take a while.')
    file_lang0 = os.path.join(UNPROCESSED_FOLDER, f'processed.{langs[0]}')
    file_lang1 = os.path.join(UNPROCESSED_FOLDER, f'processed.{langs[1]}')
    if not os.path.exists(file_lang0) or not os.path.exists(file_lang1):
        with open(file_lang0, 'w', encoding='utf-8') as f0, open(file_lang1, 'w', encoding='utf-8') as f1:
            for l0, l1 in zipped:
                f0.write(' '.join(l0))
                f0.write('\n')
                f1.write(' '.join(l1))
                f1.write('\n')
    concatenated_filename = os.path.join(UNPROCESSED_FOLDER, 'concatenated')
    if not os.path.exists(concatenated_filename):
        with open(concatenated_filename, 'w', encoding='utf-8') as f:
            for lang in langs:
                file_lang = os.path.join(UNPROCESSED_FOLDER, f'processed.{lang}')
                with open(file_lang, 'r', encoding='utf-8') as l:
                    f.writelines(l.readlines())
    outfile_name = os.path.join(PROCESSED_FOLDER, f'bpe.{merge_ops}')
    print('Learning bpe')
    if not os.path.exists(outfile_name):
        with open(concatenated_filename, 'r', encoding='utf-8') as infile, open(outfile_name, 'w', encoding='utf-8') as outfile:
            learn_bpe.learn_bpe(infile, outfile, num_symbols=merge_ops) # Get codecs, write codecs to outfile
    print('Creating BPE object')
    with open(outfile_name, 'r', encoding='utf-8') as codec:
        bpe = apply_bpe.BPE(codec)
    print('Writing bpe')
    for lang in ['en', 'de']:
        lang_file = os.path.join(UNPROCESSED_FOLDER, f'processed.{lang}')
        filename = os.path.join(PROCESSED_FOLDER, f'bpe.{lang}')
        with open(lang_file, 'r', encoding='utf-8') as f1, open(filename, 'w', encoding='utf-8') as f2:
            for line in f1:
                f2.write(bpe.process_line(line))


def endepreprocessing(args):
    _, _, CORPORA = get_data_urls_and_filenames('en-de')
    data = {}
    for lang in ['en', 'de']:
        print(f'Processing {lang}')
        full_set_file = f'{lang}-train-full'
        full_set_filepath = os.path.join(UNPROCESSED_FOLDER, full_set_file)
        if not os.path.exists(full_set_filepath):
            for corp in CORPORA:
                filename = os.path.join(UNPROCESSED_FOLDER, f'{corp}.{lang}')
                print(f'Loading {corp}')
                with open(filename, 'r', encoding='utf-8') as f:
                    data[lang] += f.readlines()
            print(f'Saving to {full_set_file}')
            with open(full_set_filepath, 'w', encoding='utf-8') as f:
                f.writelines(data[lang])
        else:
            with open(full_set_filepath, 'r', encoding='utf-8') as f:
                data[lang] = f.readlines()
    print('Tokenizing')
    en, de = tokenize('en', data['en']), tokenize('de', data['de'])
    print('Cleaning')
    zipped = clean_corpus(en, de)
    # Byte-pair encoding
    print('Byte-pair encodings')
    subword(zipped, ['en', 'de'], args)
    print('Done')


def load_data(filename, fields, batch_size, device):
    path = os.path.join(PROCESSED_FOLDER, filename)
    if not isinstance(fields[0], (tuple, list)):
        fields = [('src', fields[0]), ('trg', fields[1])]
    def example_generator():
        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in ('.en', '.de'))
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    yield Example.fromlist([src_line, trg_line], fields)
    examples = example_generator()
    data = BucketIterator(
        Dataset(
            examples,
            fields
        ),
        batch_size=batch_size,
        device=device,
        sort=False,
        shuffle=False,
    )
    return data


def load_vocab(lang_extension):
    filename = os.path.join(PROCESSED_FOLDER, f'vocab.50k.{lang_extension}')
    vocab = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            token, number = line.split()
            number = int(number)
            vocab[token] = number
    return Counter(vocab)


def load_data_dict(opts, device):
    en_field = Field(
        tokenize=str.split,
        pad_token=PAD_WORD,
        unk_token=UNK_WORD,
        eos_token=EOS_WORD,
    )
    de_field = Field(
        tokenize=str.split,
        pad_token=PAD_WORD,
        unk_token=UNK_WORD,
        eos_token=EOS_WORD,
    )
    fields = (en_field, de_field)
    print('Loading en vocab')
    en_vocab = load_vocab('en')
    en_field.vocab = en_field.vocab_cls(en_vocab, specials=[PAD_WORD, UNK_WORD, EOS_WORD])
    print('Loading de vocab')
    de_vocab = load_vocab('de')
    de_field.vocab = de_field.vocab_cls(de_vocab, specials=[PAD_WORD, UNK_WORD, EOS_WORD])
    print('Loading data')
    training = load_data(opts.train_data, fields, opts.batch_size, device)
    print('Training data loaded')
    val = load_data(opts.val_data, fields, opts.batch_size, device)
    print('Validation data loaded')
    opts.max_token_seq_len = 80
    opts.src_pad_idx = en_field.vocab.stoi[PAD_WORD]
    opts.trg_pad_idx = de_field.vocab.stoi[PAD_WORD]
    opts.src_vocab_size = len(en_field.vocab)
    opts.trg_vocab_size = len(de_field.vocab)
    return training, val


# en-fr: 32000 word-piece vocab: Google’s neural machine translation system: Bridging the gap between human and machine translation (2016)
# https://arxiv.org/pdf/1609.08144.pdf

# Use sentencepiece

def enfrpreprocessing():
    pass


