import os
import fileinput
import numpy as np


DATA_FOLDER = './data'
DOWNLOADS_FOLDER = os.path.join(DATA_FOLDER, 'downloads')
UNPROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'unprocessed')
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'processed')
TOKENIZED_FOLDER = os.path.join(UNPROCESSED_FOLDER, 'tokenized')
CLEANED_FOLDER = os.path.join(UNPROCESSED_FOLDER, 'cleaned')


if not os.path.exists(DOWNLOADS_FOLDER):
    os.mkdir(DOWNLOADS_FOLDER)
if not os.path.exists(UNPROCESSED_FOLDER):
    os.mkdir(UNPROCESSED_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.mkdir(PROCESSED_FOLDER)
if not os.path.exists(TOKENIZED_FOLDER):
    os.mkdir(TOKENIZED_FOLDER)
if not os.path.exists(CLEANED_FOLDER):
    os.mkdir(CLEANED_FOLDER)


def get_data_urls(langs):
    if langs == ['en', 'de']:
        return [
            "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz",
            "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz",
            "http://statmt.org/wmt14/training-parallel-nc-v9.tgz",
            "http://www.statmt.org/wmt14/dev.tgz",
            "http://statmt.org/wmt14/test-full.tgz",
        ]
    if langs == ['en', 'fr']:
        return [
            "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz",
            "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz",
            "http://statmt.org/wmt13/training-parallel-un.tgz",
            "http://statmt.org/wmt14/training-parallel-nc-v9.tgz",
            "http://statmt.org/wmt10/training-giga-fren.tar",
            "http://www.statmt.org/wmt14/dev.tgz",
            "http://statmt.org/wmt14/test-full.tgz",
        ]


def get_data_files(langs):
    if langs == ['en', 'de']:
        files = [
            "training-parallel-europarl-v7.tgz",
            "training-parallel-commoncrawl.tgz",
            "training-parallel-nc-v9.tgz",
            "dev.tgz",
            "test-full.tgz",
        ]
    elif langs == ['en', 'fr']:
        files = [
            "training-parallel-europarl-v7.tgz",
            "training-parallel-commoncrawl.tgz",
            "training-parallel-un.tgz",
            "training-parallel-nc-v9.tgz",
            "training-giga-fren.tar",
            "dev.tgz",
            "test-full.tgz",
        ]
    return [os.path.join(DOWNLOADS_FOLDER, f'{fi}') for fi in files]


def get_training_data_corpora(langs):
    if langs == ['en', 'de']:
        corpora = [
            "training/europarl-v7.de-en",
            "commoncrawl.de-en",
            "training/news-commentary-v9.de-en",
        ]
    elif langs == ['en', 'fr']:
        corpora = [
            "training/europarl-v7.fr-en",
            "commoncrawl.fr-en",
            "un/undoc.2000.fr-en",
            "training/news-commentary-v9.fr-en",
            "giga-fren.release2.fixed",
        ]
    FULL_DATA_NAME = f'{langs[0]}{langs[1]}_full'
    full_data_paths = [os.path.join(UNPROCESSED_FOLDER, f'{FULL_DATA_NAME}.{lang}') for lang in langs]
    for i, lang in enumerate(langs):
        if os.path.exists(full_data_paths[i]):
            continue
        with open(full_data_paths[i], 'w', encoding='utf-8') as f:
            filepaths = [os.path.join(UNPROCESSED_FOLDER, f'{c}.{lang}') for c in corpora]
            for filepath in filepaths:
                with open(filepath, 'r', encoding='utf-8') as fin:
                    for line in fin:
                        f.write(line)
    return full_data_paths


def get_dev_data_corpora(langs):
    if langs == ['en', 'de']:
        corpora = "test-full/newstest2014-deen-src"
    elif langs == ['en', 'fr']:
        corpora = "test-full/newstest2014-fren-src"
    filepaths = [os.path.join(UNPROCESSED_FOLDER, f'{corpora}.{lang}.sgm') for lang in langs]
    return filepaths


def get_test_data_corpora(langs):
    if langs == ['en', 'de']:
        corpora = "dev/newstest2013"
    elif langs == ['en', 'fr']:
        corpora = "dev/newstest2013"
    filepaths = [os.path.join(UNPROCESSED_FOLDER, f'{corpora}.{lang}') for lang in langs]
    return filepaths


def get_cleaned_path(prefix, lang):
    return os.path.join(CLEANED_FOLDER, f'{prefix}.{lang}')

def get_tokenized_path(prefix, lang):
    return os.path.join(TOKENIZED_FOLDER, f'{prefix}.{lang}')

def get_concat_path(prefix):
    return os.path.join(CLEANED_FOLDER, f'{prefix}_concat')

def get_bpe_path(prefix, merge_ops):
    return os.path.join(PROCESSED_FOLDER, f'{prefix}.{merge_ops}')

def get_vocab_path(experiment, lang):
    return os.path.join(PROCESSED_FOLDER, f'{experiment}_vocab.{lang}')

def get_processed_data_path(experiment, corpora_type, lang):
    return os.path.join(PROCESSED_FOLDER, f'{experiment}_{corpora_type}.{lang}')

def get_reduced_data_path(experiment, corpora_type, size, lang):
    return os.path.join(PROCESSED_FOLDER, f'{experiment}_{corpora_type}_reduced_{size}.{lang}')

def reduce_dataset(experiment, corpora_type, langs, size):
    # Reduces a dataset by a factor of keep_every
    for i, lang in enumerate(langs):
        infile = get_processed_data_path(experiment, corpora_type, lang)
        outfile = get_reduced_data_path(experiment, corpora_type, size, lang)
        if (os.path.exists(outfile)):
            print(outfile, 'exists, skipping.')
            break
        if i == 0:
            with open(infile, 'r', encoding='utf-8') as orig:
                total_size = -1
                for total_size, _ in enumerate(orig):
                    pass
                total_size += 1
            if total_size <= size:
                continue
            keep = np.random.choice(total_size, size=size, replace=False)
            keep = np.sort(keep)
        with open(infile, 'r', encoding='utf-8') as orig, open(outfile, 'w', encoding='utf-8') as new:
            n = -1
            for k in keep:
                found = False
                while not found:
                    line = orig.readline()
                    n += 1
                    if n == k:
                        new.write(line)
                        found = True
    return get_reduced_data_path(experiment, corpora_type, size, '')