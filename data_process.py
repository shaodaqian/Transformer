
from torchtext.data import Field, BucketIterator, Example, Dataset
from torchtext.datasets import TranslationDataset
from sacremoses import MosesTokenizer
from subword_nmt import learn_bpe, apply_bpe, get_vocab
import sentencepiece as spm
import re
import os
from collections import Counter
import io
from numpy import random
import shutil

from special_tokens import PAD_WORD, UNK_WORD, EOS_WORD, BOS_WORD
from data_files import reduce_dataset, get_training_data_corpora, get_dev_data_corpora, get_test_data_corpora, get_bpe_path, get_cleaned_path, get_concat_path, get_processed_data_path, get_tokenized_path, get_vocab_path, get_reduced_data_path


class DataCleaner:
    """
    Cleans given data, and saves it to a file
    data: Dictionary mapping language name to iterable of strings.
    cleaned_name: The name to call the cleaned corpuses. Will be stored as cleaned_name.languagename for each language.
    """
    def __init__(self, tok_filepaths, cleaned_filepaths, overwrite):
        assert (len(tok_filepaths) == 2)
        def get_corpuses():
            with open(tok_filepaths[0], 'r', encoding='utf-8') as f0, open(tok_filepaths[1], 'r', encoding='utf-8') as f1:
                for line0, line1 in zip(f0.readlines(), f1.readlines()):
                    yield line0.split(), line1.split()
        print('Cleaning')
        corpuses = get_corpuses()
        corpuses = self.remove_empty_lines(corpuses)
        corpuses = self.remove_redundant_spaces(corpuses)
        corpuses = self.drop_lines(corpuses)
        self.save(corpuses, cleaned_filepaths, overwrite)

    def save(self, corpuses, cleaned_filepaths, overwrite):
        file_lang0 = cleaned_filepaths[0]
        file_lang1 = cleaned_filepaths[1]
        if overwrite == False and os.path.exists(file_lang0) and os.path.exists(file_lang1):
            print(file_lang0, file_lang1, 'already exist')
            return
        with open(file_lang0, 'w', encoding='utf-8') as f0, open(file_lang1, 'w', encoding='utf-8') as f1:
            for l0, l1 in corpuses:
                f0.write(' '.join(l0))
                f0.write('\n')
                f1.write(' '.join(l1))
                f1.write('\n')

    def remove_empty_lines(self, corpuses):
        for l1, l2 in corpuses:
            if not (l1 == '' or l2 == '' or l1 == '\n' or l2 == '\n'):
                yield l1, l2

    def remove_redundant_spaces(self, corpuses):
        regex = r' +' # 1 or more spaces
        for l1, l2 in corpuses:
            l1 = [re.sub(regex, ' ', word) for word in l1]
            l2 = [re.sub(regex, ' ', word) for word in l2]
            yield l1, l2

    # drops lines (and their corresponding lines), that are empty, too short, too long or violate the 9-1 sentence ratio limit of GIZA++
    def drop_lines(self, corpuses, min_length=1, max_length=80):
        for l1, l2 in corpuses:
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


def concatenate_files(filepaths, concat_path, overwrite):
    if os.path.exists(concat_path) and overwrite == False:
        print(concat_path, 'already exists')
        return
    with open(concat_path, 'w', encoding='utf-8') as f:
        for filepath in filepaths:
            with open(filepath, 'r', encoding='utf-8') as l:
                shutil.copyfileobj(l, f)
                f.write('\n')


class BytePairPipeline:
    def __init__(self, langs, experiment_name, corpora_type, overwrite=False, merge_ops=32000):
        self.langs = langs
        self.merge_ops = merge_ops
        self.experiment_name = experiment_name
        self.corpora_type = corpora_type
        self.file_prefix = f'{experiment_name}_{corpora_type}'
        if corpora_type == 'training':
            datapaths = get_training_data_corpora(langs)
        elif corpora_type == 'dev':
            datapaths = get_dev_data_corpora(langs)
        elif corpora_type == 'test':
            datapaths = get_test_data_corpora(langs)
        tokenized_filepaths = [get_tokenized_path(self.file_prefix, lang) for lang in langs]
        for i, lang in enumerate(langs):
            self.moses_tokenize(lang, datapaths[i], tokenized_filepaths[i], overwrite=overwrite)
        cleaned_filepaths = [get_cleaned_path(self.file_prefix, lang) for lang in langs]
        DataCleaner(tok_filepaths=tokenized_filepaths, cleaned_filepaths=cleaned_filepaths, overwrite=overwrite)
        self.subword(cleaned_filepaths=cleaned_filepaths, overwrite=overwrite)

    def moses_tokenize(self, lang, datapath, tokpath, overwrite):
        if os.path.exists(tokpath) and overwrite == False:
            print(tokpath, 'already exists.')
            return
        print('Tokenizing', tokpath)
        if lang in ['en', 'de', 'fr']:
            mosestok = MosesTokenizer(lang=lang)
            with open(datapath, 'r', encoding='utf-8') as corpus, open(tokpath, 'w', encoding='utf-8') as tokfile:
                for line in corpus.readlines():
                    tokd = mosestok.tokenize(line)
                    tokfile.write(' '.join(tokd))
                    tokfile.write('\n')
        else:
            print(f'Unrecognised language: {lang}')
            raise ValueError()

    # Byte-pair encoding (aka subword)
    def subword(self, cleaned_filepaths, overwrite):
        bpe_filepath = get_bpe_path(self.experiment_name, self.merge_ops)
        if self.corpora_type == 'training':
            # Concatenated file necessary for BPE learning
            concatenated_filepath = get_concat_path(self.file_prefix)
            concatenate_files(cleaned_filepaths, concatenated_filepath, overwrite=overwrite)
            if os.path.exists(bpe_filepath) and overwrite == False:
                print(bpe_filepath, 'already exists')
            else:
                print('Learning BPE encoding. This may take a while.')
                with open(concatenated_filepath, 'r', encoding='utf-8') as infile, open(bpe_filepath, 'w', encoding='utf-8') as outfile:
                    learn_bpe.learn_bpe(infile, outfile, num_symbols=self.merge_ops) # Get codecs, write codecs to outfile
        print('Applying')
        with open(bpe_filepath, 'r', encoding='utf-8') as codec:
            bpe = apply_bpe.BPE(codec)
        print('Writing bpe')
        for i, lang in enumerate(self.langs):
            lang_filepath = cleaned_filepaths[i]
            processed_filepath = get_processed_data_path(self.experiment_name, self.corpora_type, lang)
            if overwrite == False and os.path.exists(processed_filepath):
                continue
            with open(lang_filepath, 'r', encoding='utf-8') as f1, open(processed_filepath, 'w', encoding='utf-8') as f2:
                for line in f1:
                    f2.write(bpe.process_line(line))
            if self.corpora_type == 'training':
                vocab_filepath = get_vocab_path(self.experiment_name, lang)
                with open(processed_filepath, 'r', encoding='utf-8') as train_file, open(vocab_filepath, 'w', encoding='utf-8') as vocab_file:
                    get_vocab.get_vocab(train_file, vocab_file)
            


def load_data(experiment_name, fields, langs, batch_size, device, corpora_type, reduce_size):
    # if not isinstance(fields[0], (tuple, list)):
    #     fields = [('src', fields[0]), ('trg', fields[1])]
    # def example_generator():
    #     src_path, trg_path = tuple(os.path.expanduser(path + x) for x in ('.en', '.de'))
    #     with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
    #             io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
    #         for src_line, trg_line in zip(src_file, trg_file):
    #             src_line, trg_line = src_line.strip(), trg_line.strip()
    #             if src_line != '' and trg_line != '':
    #                 yield Example.fromlist([src_line, trg_line], fields)
    # examples = list(example_generator())
    # data = BucketIterator(
    #     Dataset(
    #         examples,
    #         fields
    #     ),
    #     batch_size=batch_size,
    #     device=device,
    #     train=train,
    #     sort=False,
    #     shuffle=False,
    # )
    if corpora_type == 'training':
        train = True
    else:
        train = False

    if not train or reduce_size == -1:
        fp = get_processed_data_path(experiment_name, corpora_type, '')
    elif train:
        fp = reduce_dataset(
            experiment_name,
            corpora_type,
            langs=langs,
            size=reduce_size
        )
    
    total_tokens = 0
    for lang in langs:
        with open(f'{fp}{lang}', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                total_tokens += len(line.split())

    def batch_size_fn(example, current_count, current_size):
        current_size += len(example.src)
        current_size += len(example.trg)
        return current_size
    data = BucketIterator(
        TranslationDataset(
            exts=langs,
            fields=fields,
            path=fp
        ),
        batch_size=batch_size,
        batch_size_fn=batch_size_fn,
        device=device,
        train=train,
        sort=True,
        shuffle=False,
    )
    return data, total_tokens


def load_vocab(vocab_filepath):
    vocab = {}
    with open(vocab_filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            token, number = line.split()
            number = int(number)
            vocab[token] = number
    return Counter(vocab)


def load_data_dict(experiment_name, langs, corpora_type, args, device, src_field=None, trg_field=None):
    if src_field == None or trg_field == None:
        src_field = Field(
            tokenize=str.split,
            unk_token=UNK_WORD,
            pad_token=PAD_WORD,
            init_token=BOS_WORD,
            eos_token=EOS_WORD
        )
        trg_field = Field(
            tokenize=str.split,
            unk_token=UNK_WORD,
            pad_token=PAD_WORD,
            init_token=BOS_WORD,
            eos_token=EOS_WORD
        )
        fields = (src_field, trg_field)
        print('Loading src vocab')
        src_vocab = load_vocab(get_vocab_path(experiment_name, langs[0]))
        src_field.vocab = src_field.vocab_cls(src_vocab, specials=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])
        print('Loading trg vocab')
        trg_vocab = load_vocab(get_vocab_path(experiment_name, langs[1]))
        trg_field.vocab = trg_field.vocab_cls(trg_vocab, specials=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])
        args.src_pad_idx = src_field.vocab.stoi[PAD_WORD]
        args.trg_pad_idx = trg_field.vocab.stoi[PAD_WORD]
        args.trg_bos_idx = trg_field.vocab.stoi[BOS_WORD]
        args.trg_eos_idx = trg_field.vocab.stoi[EOS_WORD]
        args.src_vocab_size = len(src_field.vocab)
        args.trg_vocab_size = len(trg_field.vocab)

    print('Loading data')
    data, total_tokens = load_data(
        experiment_name=experiment_name,
        langs=langs,
        fields=fields,
        batch_size=args.batch_size,
        device=device,
        corpora_type=corpora_type,
        reduce_size=args.data_reduce_size
    )
    return data, total_tokens, src_field, trg_field


class SentencePieceWrapper:
    def __init__(self, experiment_name, langs, overwrite):
        MODEL_LOCATION = './data/processed/sp'
        if not os.path.exists(MODEL_LOCATION):
            os.makedirs(MODEL_LOCATION)
        self.experiment_name = experiment_name
        self.model_prefix = os.path.join(MODEL_LOCATION, experiment_name)
        self.langs = langs
        self.train(overwrite)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{self.model_prefix}.model')

    def train(self, overwrite):
        full_data_paths = get_training_data_corpora(self.langs)
        concat_filepath = get_concat_path(self.experiment_name)
        concatenate_files(full_data_paths, concat_filepath, overwrite=overwrite)
        if overwrite == True or not os.path.exists(f'{self.model_prefix}.model'):
            spm.SentencePieceTrainer.train((
                f'--input={concat_filepath} --model_prefix={self.model_prefix} --vocab_size=32000 '
                f'--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
                f'--pad_piece={PAD_WORD} --unk_piece={UNK_WORD} --bos_piece={BOS_WORD} --eos_piece={EOS_WORD} '
                f'--input_sentence_size=5000000 --shuffle_input_sentence=true'
            ))
        print('Trained!')

    def tokenize(self, corpora_type, overwrite):
        if corpora_type == 'training':
            data_paths = get_training_data_corpora(self.langs)
        elif corpora_type == 'test':
            data_paths = get_test_data_corpora(self.langs)
        elif corpora_type == 'dev':
            data_paths = get_dev_data_corpora(self.langs)
        tokenized_filepaths = [get_processed_data_path(self.experiment_name, corpora_type, lang) for lang in self.langs]
        for datapath, tokpath in zip(data_paths, tokenized_filepaths):
            if overwrite == False and os.path.exists(tokpath):
                continue
            print('Tokenizing', datapath)
            with open(datapath, 'r', encoding='utf-8') as f_in, open(tokpath, 'w', encoding='utf-8') as f_out:
                for line in f_in.readlines():
                    tokenized = self.sp.encode_as_pieces(line)
                    f_out.write(' '.join(tokenized))
                    f_out.write('\n')


if __name__ == "__main__":
    BytePairPipeline(
        langs=['en', 'de'],
        experiment_name='bpe_ende',
        corpora_type='training',
        overwrite=False,
        merge_ops=32000
    )
    BytePairPipeline(
        langs=['en', 'de'],
        experiment_name='bpe_ende',
        corpora_type='dev',
        overwrite=True,
        merge_ops=32000
    )
    BytePairPipeline(
        langs=['en', 'de'],
        experiment_name='bpe_ende',
        corpora_type='test',
        overwrite=True,
        merge_ops=32000
    )
    BytePairPipeline(
        langs=['en', 'fr'],
        experiment_name='bpe_enfr',
        corpora_type='training',
        overwrite=False,
        merge_ops=32000
    )
    BytePairPipeline(
        langs=['en', 'fr'],
        experiment_name='bpe_enfr',
        corpora_type='dev',
        overwrite=True,
        merge_ops=32000
    )
    BytePairPipeline(
        langs=['en', 'fr'],
        experiment_name='bpe_enfr',
        corpora_type='test',
        overwrite=True,
        merge_ops=32000
    )
    # ende_sp = SentencePieceWrapper(
    #     langs=['en', 'de'],
    #     experiment_name='sp_ende',
    #     overwrite=True
    # )
    # ende_sp.tokenize(
    #     corpora_type='training',
    #     overwrite=True
    # )
    # ende_sp.tokenize(
    #     corpora_type='test',
    #     overwrite=True
    # )
    # ende_sp.tokenize(
    #     corpora_type='dev',
    #     overwrite=True
    # )
