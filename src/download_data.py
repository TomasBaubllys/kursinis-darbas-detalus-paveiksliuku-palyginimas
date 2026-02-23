# Author Tomas Baublys
# File purpose:
# This file creates a folder if it doesnt exists in the root folder called 'data'
# Downloads and unzips the Market1501 dataset from a know google-drive link

import gdown
import zipfile
import os

file_id = '0B8-rUzbwVRk0c054eEozWG9COHM'
url = f'https://drive.google.com/uc?id={file_id}'
output_zip = 'Market-1501.zip'
relative_dir = '../data'

def setup_market1501():
    data_dir = os.path.join(os.getcwd(), relative_dir)
    zip_file_full_path = os.path.join(data_dir, output_zip)
    if not os.path.exists(data_dir):
        print(f"Creating a new directory: {data_dir}")
        os.mkdir(data_dir)
    if not os.path.exists(zip_file_full_path):
        print(f"Downloading the dataset {output_zip} file")
        gdown.download(url, zip_file_full_path, quiet=False)
    

setup_market1501()
# gdown.download(url, output, quiet=False)

