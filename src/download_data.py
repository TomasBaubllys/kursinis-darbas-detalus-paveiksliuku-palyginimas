# Author Tomas Baublys
# File purpose:
# This file creates a folder if it doesnt exists in the root folder called 'data'
# Downloads and unzips the Market1501 dataset from a know google-drive link

import gdown
import zipfile
import os

file_id = '0B8-rUzbwVRk0c054eEozWG9COHM'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'Market-1501.zp'
relative_dir = '../data'

def setup_market1501():
    if not os.path.exists(os.path.join(os.getcwd(), relative_dir)):
        data_dir = os.path.join(os.getcwd(), relative_dir)
        print(f"Create a new directory: {data_dir}")
        os.mkdir(data_dir)

setup_market1501()
# gdown.download(url, output, quiet=False)

