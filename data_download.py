import os
import urllib.request
import tarfile
from data_files import get_data_files, get_data_urls, DOWNLOADS_FOLDER, UNPROCESSED_FOLDER
import gzip
import shutil


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
                print(f'{file_size_dl//(1024*1024)} MB  [{progress:.0f}%]')


def download_data(task):
    URLS = get_data_urls(task)
    FILES = get_data_files(task)
    for url, filepath in zip(URLS, FILES):
        # Download file
        with urllib.request.urlopen(url) as response:
            meta = response.info()
            file_size = int(meta.get("Content-Length"))
            if os.path.exists(filepath) and int(os.stat(filepath).st_size) == file_size:
                print(f'File {filepath} exists, skipping download')
            else:
                print(f'Downloading: {filepath}, {file_size//(1024*1024)} MB')
                download_file(response, filepath, file_size)
        # Unpack tgz files
        if url == "http://statmt.org/wmt10/training-giga-fren.tar":
            # Special case for tar file instead of tgz file.
            with tarfile.open(filepath) as tar:
                tar.extractall(DOWNLOADS_FOLDER)
            packed = ["giga-fren.release2.fixed.en", "giga-fren.release2.fixed.fr"]
            for filename in packed:
                gz_filepath = os.path.join(DOWNLOADS_FOLDER, f'{filename}.gz')
                extracted_filepath = os.path.join(UNPROCESSED_FOLDER, filename)
                print(f'Extracting {gz_filepath}')
                with gzip.open(gz_filepath, 'rb') as f_in, open(extracted_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f'Extracting {filepath}')
            with tarfile.open(filepath) as tar:
                tar.extractall(UNPROCESSED_FOLDER)


if __name__ == '__main__':
    download_data(['en', 'fr'])
    download_data(['en', 'de'])
