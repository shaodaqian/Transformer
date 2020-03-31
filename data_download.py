import os
import urllib.request
import tarfile


DATA_FOLDER = './data/unprocessed'


def download_file(response, file_path, file_size):
    with open(file_path, 'wb') as f:
        file_size_dl = 0
        block_sz = 2 ** 19
        loop = True
        print('[0.0]%')
        while loop:
            buffer = response.read(block_sz)
            if not buffer:
                loop = False
                continue
            file_size_dl += len(buffer)
            f.write(buffer)
            progress = file_size_dl * 100. / file_size
            if int(progress * 20) % 20 == 0:
                print(f'{file_size_dl}  [{progress:.0f}%]')


def get_data_urls_and_filenames(task):
    if task == 'en-de':
        URLS = [
            "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz",
            "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz",
            "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz",
            "http://data.statmt.org/wmt17/translation-task/dev.tgz",
            "http://statmt.org/wmt14/test-full.tgz",
        ]
        FILES = [
            "training-parallel-europarl-v7",
            "training-parallel-commoncrawl",
            "training-parallel-nc-v12",
            "dev",
            "test-full",
        ]
        CORPORA = [
            "training/europarl-v7.de-en",
            "commoncrawl.de-en",
            "training/news-commentary-v12.de-en",
        ]
    elif task == 'en-fr':
        URLS = [
            "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz",
            "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz",
            "http://statmt.org/wmt13/training-parallel-un.tgz",
            "http://statmt.org/wmt14/training-parallel-nc-v9.tgz",
            "http://statmt.org/wmt10/training-giga-fren.tar",
            "http://statmt.org/wmt14/test-full.tgz",
        ]
        FILES = [
            "training-parallel-europarl-v7",
            "training-parallel-commoncrawl",
            "training-parallel-un",
            "training-parallel-nc-v9",
            "training-giga-fren",
            "test-full",
        ]
        CORPORA = [
            "training/europarl-v7.fr-en",
            "commoncrawl.fr-en",
            "un/undoc.2000.fr-en",
            "training/news-commentary-v9.fr-en",
            "giga-fren.release2.fixed",
        ]
    else:
        print(f'Unknown task {task}: only en-de and en-fr available.')
        URLS = []
        FILES = []
        CORPORA = []
    return URLS, FILES, CORPORA


if __name__ == '__main__':
    try:
        os.mkdir(DATA_FOLDER)
    except FileExistsError:
        print('Data folder exists')
    URLS, FILES, CORPORA = get_data_urls_and_filenames('en-fr')
    for i, url in enumerate(URLS):
        filename = f'{FILES[i]}.tgz'
        file_path = os.path.join(DATA_FOLDER, filename)
        with urllib.request.urlopen(url) as response:
            meta = response.info()
            file_size = int(meta.get("Content-Length"))
            if os.path.exists(file_path) and int(os.stat(file_path).st_size) == file_size:
                print(f'File {filename} exists, skipping download')
            else:
                print(f'Downloading: {filename} Bytes: {file_size}')
                download_file(response, file_path, file_size)
        # Unpack tgz file
        extract_folder = os.path.join(DATA_FOLDER, FILES[i])
        if not os.path.exists(extract_folder):
            os.mkdir(extract_folder)
            with tarfile.open(file_path) as tf:
                print(f'Extracting into {extract_folder}')
                tf.extractall(extract_folder)
        else:
            print(f'{extract_folder} exists, skipping extraction')
